import torch
import time
import pandas as pd
import os
import platform
from thop import profile
import warnings
from types import SimpleNamespace

# --- 1. 环境与可复现性设置 ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
warnings.filterwarnings("ignore")

# --- 2. 导入所有需要测试的模型 ---
from models.DLinear import Model as DLinear
from models.PatchTST import Model as PatchTST
from models.iTransformer import Model as iTransformer
from models.Crossformer import Model as Crossformer
from models.TimesNet import Model as TimesNet
from models.Koopa import Model as Koopa


# --- 3. 定义核心评测函数 ---
def get_model_stats_for_device(model, model_name, config, device, iterations=50):
    stats = {}
    batch_size, seq_len, channels = config['batch_size'], config['seq_len'], config['channels']
    pred_len, label_len = config['pred_len'], config['label_len']

    x_enc = torch.randn(batch_size, seq_len, channels).to(device)
    x_mark_enc = torch.randn(batch_size, seq_len, 4).to(device)
    dec_inp_history = torch.randn(batch_size, label_len, channels).to(device)
    dec_inp_zeros = torch.zeros(batch_size, pred_len, channels).to(device)
    x_dec = torch.cat([dec_inp_history, dec_inp_zeros], dim=1).to(device)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4).to(device)

    model.to(device)

    if device.type == 'cpu':
        with torch.no_grad():
            try:
                macs, params = profile(model, inputs=(x_enc, x_mark_enc, x_dec, x_mark_dec), verbose=False)
            except Exception:
                try:
                    macs, params = profile(model, inputs=(x_enc,), verbose=False)
                except Exception as e:
                    print(f"  [Warning] MACs/FLOPs calculation failed: {e}")
                    macs, params = 0, sum(p.numel() for p in model.parameters() if p.requires_grad)
        stats['Params (M)'] = round(params / 1e6, 4)
        stats['MACs (G)'] = round(macs / 1e9, 4)
        stats['FLOPs (G)'] = round(2 * macs / 1e9, 4)

    with torch.inference_mode():
        if device.type == 'cuda':
            torch.cuda.synchronize()
            for _ in range(20): _ = model(x_enc.clone(), x_mark_enc.clone(), x_dec.clone(), x_mark_dec.clone())
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        else:
            for _ in range(10): _ = model(x_enc.clone(), x_mark_enc.clone(), x_dec.clone(), x_mark_dec.clone())

        start_time = time.time()
        for _ in range(iterations): _ = model(x_enc.clone(), x_mark_enc.clone(), x_dec.clone(), x_mark_dec.clone())
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.time()

    total_time = end_time - start_time
    device_upper = device.type.upper()
    stats[f'Inference Time ({device_upper}) (ms/batch)'] = round((total_time / iterations) * 1000, 4)
    stats[f'Throughput ({device_upper}) (batch/s)'] = round(iterations / total_time, 4)
    stats[f'Throughput ({device_upper}) (samples/s)'] = round(batch_size * iterations / total_time, 4)

    if device.type == 'cuda':
        stats['Max Memory (MB)'] = round(torch.cuda.max_memory_allocated() / 1e6, 4)

    return stats


# --- 4. 定义测试配置 ---
CONFIG = {
    'batch_size': 8, 'seq_len': 96, 'label_len': 48, 'pred_len': 720, 'channels': 32,
}
E_LAYERS_LIST = [1, 2, 3]
D_MODEL_LIST = [24, 48, 64, 128, 256, 512]
MODELS_TO_TEST = {
    "DLinear": {"class": DLinear, "params": {"individual": False}},
    "iTransformer": {"class": iTransformer, "params": {}},
    "PatchTST": {"class": PatchTST, "params": {"patch_len": 16, "stride": 8}},
    "Crossformer": {"class": Crossformer, "params": {"seg_len": 6}},
    "TimesNet": {"class": TimesNet, "params": {"top_k": 5}},
}

# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    devices_to_test = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device('cuda'))

    print(f"{'=' * 60}\nStarting benchmark on devices: {[d.type for d in devices_to_test]}")
    print(f"CPU Threads: {torch.get_num_threads()}")
    if 'cuda' in [d.type for d in devices_to_test]: print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 60}")

    # --- 基础配置模板 (已补全) ---
    BASE_CONFIGS = {
        'task_name': 'long_term_forecast', 'is_training': False, 'moving_avg': 25, 'embed': 'timeF', 'freq': 'h',
        'dropout': 0.1, 'factor': 1, 'n_heads': 8, 'd_layers': 1, 'activation': 'gelu', 'output_attention': False,
        'distil': True, 'num_class': 1, 'p_hidden_dims': [128, 128], 'p_hidden_layers': 2,
        'label_len': CONFIG['label_len'],
        'num_kernels': 6,  # <-- 为TimesNet添加
        'data': 'custom',  # <-- 为Koopa添加
    }

    all_results = []
    for model_name, model_info in MODELS_TO_TEST.items():
        for e_layers in E_LAYERS_LIST:
            for d_model in D_MODEL_LIST:
                d_ff = d_model * 2
                print(f"--- Testing {model_name} | e_layers={e_layers} | d_model={d_model} ---")

                combined_run_info = {'Model': model_name, 'e_layers': e_layers, 'd_model': d_model}
                current_config_dict = BASE_CONFIGS.copy()
                current_config_dict.update({
                    'seq_len': CONFIG['seq_len'], 'pred_len': CONFIG['pred_len'], 'enc_in': CONFIG['channels'],
                    'dec_in': CONFIG['channels'], 'c_out': CONFIG['channels'], 'e_layers': e_layers,
                    'd_model': d_model, 'd_ff': d_ff,
                })
                current_config_dict.update(model_info['params'])

                try:
                    configs_obj = SimpleNamespace(**current_config_dict)
                    model = model_info['class'](configs_obj)

                    for device in devices_to_test:
                        print(f"  > Testing on {device.type.upper()}...")
                        device_stats = get_model_stats_for_device(model, model_name, CONFIG, device)
                        combined_run_info.update(device_stats)
                    all_results.append(combined_run_info)
                except Exception as e:
                    print(f"  [ERROR] Failed: {e}")
                    error_info = {'Model': model_name, 'e_layers': e_layers, 'd_model': d_model, 'Error': str(e)}
                    all_results.append(error_info)

    # --- 6. 保存结果 ---
    if not os.path.exists('results'): os.makedirs('results')
    output_filename = "results/benchmark_results.csv"
    df = pd.DataFrame(all_results)

    fixed_cols = ['Model', 'e_layers', 'd_model', 'Params (M)', 'MACs (G)', 'FLOPs (G)']
    cpu_cols = sorted([col for col in df.columns if '(CPU)' in col])
    gpu_cols = sorted([col for col in df.columns if '(CUDA)' in col or 'Memory' in col])
    error_col = ['Error'] if 'Error' in df.columns else []
    final_cols = [col for col in (fixed_cols + cpu_cols + gpu_cols + error_col) if col in df.columns]
    df = df[final_cols]

    with open(output_filename, 'w') as f:
        f.write(f"# Benchmark Results\n# Platform: {platform.system()} {platform.release()}\n")
        f.write(f"# CPU: {platform.processor()}\n")
        if 'cuda' in [d.type for d in devices_to_test]: f.write(f"# CUDA Device: {torch.cuda.get_device_name(0)}\n")
        f.write("\n")
        df.to_csv(f, index=False)

    print(f"\n\nBenchmark finished.\nAll results saved to {output_filename}")
    print(df.head())