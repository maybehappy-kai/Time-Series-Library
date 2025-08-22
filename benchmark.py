import sys
import os

# --- 关键修正：将项目根目录永久性地添加到系统搜索路径的最顶端 ---
# __file__ 指向当前文件 (benchmark.py)
# os.path.abspath(__file__) 获取该文件的绝对路径
# os.path.dirname() 获取该文件所在的目录，也就是我们的项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# 确保项目根目录在搜索路径的最前面，这样Python总能最先找到主项目的模块
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- 修正结束 ---

import torch
import time
import pandas as pd
import os
import platform
import warnings
from types import SimpleNamespace
import torchinfo

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
from models.TimeFilter.models.TimeFilter import Model as TimeFilter
from models.FourierGNN.model.FourierGNN import FGN as FourierGNN
from models.CONTIME.model_tower import CONTime as CONTIME
from models.linoss.models.LinOSS import LinOSS
from models.S_D_Mamba.model.S_Mamba import Model as S_D_Mamba
from models.TimePro.model.TimePro import Model as TimePro
from models.DeepEDM.models.DeepEDM import Model as DeepEDM
from models.SimpleTM.model.SimpleTM import Model as SimpleTM
from models.TQNet.models.TQNet import Model as TQNet
from models.ModernTCN.ModernTCN_Long_term_forecasting.models.ModernTCN import Model as ModernTCN
from models.FilterNet.models.PaiFilter import Model as FilterNet
from models.NFM.Forecasting.NFM_FC import NFM_FC as NFM
from models.TimeKAN.models.TimeKAN import Model as TimeKAN
from models.SOFTS.models.SOFTS import Model as SOFTS
from pypots.imputation import TimeMixerPP


# --- 3. 定义核心评测函数 (优化稳健版) ---
def get_model_stats_for_device(model, model_name, config, device, iterations=50):
    stats = {}
    batch_size, seq_len, channels = config['batch_size'], config['seq_len'], config['channels']
    pred_len, label_len = config['pred_len'], config['label_len']

    # --- 1. 创建所有可能的输入张量 ---
    x_enc = torch.randn(batch_size, seq_len, channels).to(device)
    x_mark_enc = torch.randn(batch_size, seq_len, 4).to(device)
    dec_inp_history = torch.randn(batch_size, label_len, channels).to(device)
    dec_inp_zeros = torch.zeros(batch_size, pred_len, channels).to(device)
    x_dec = torch.cat([dec_inp_history, dec_inp_zeros], dim=1).to(device)
    x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4).to(device)

    # --- 核心修复：从模型配置中动态获取 patch_len ---
    # 为 TimeFilter 创建 masks
    try:
        # 尝试从模型实例的配置中获取 patch_len
        patch_len = model.configs.patch_len
    except AttributeError:
        # 如果模型没有 configs.patch_len (例如非 TimeFilter 模型)，
        # 提供一个默认值以避免在尝试调用 TimeFilter 时出错。
        patch_len = 16

    L = seq_len * channels // patch_len
    N = seq_len // patch_len
    masks = []
    for k in range(L):
        S = ((torch.arange(L) % N == k % N) & (torch.arange(L) != k)).to(torch.float32).to(device)
        T = ((torch.arange(L) >= k // N * N) & (torch.arange(L) < k // N * N + N) & (torch.arange(L) != k)).to(
            torch.float32).to(device)
        ST = torch.ones(L).to(torch.float32).to(device) - S - T
        ST[k] = 0.0
        masks.append(torch.stack([S, T, ST], dim=0))
    masks = torch.stack(masks, dim=0)

    model.to(device)
    model.eval()

    # --- 2. 智能检测并确定正确的模型调用方式 ---
    # 为 TimeMixerPP 创建虚拟的 missing_mask
    missing_mask = torch.zeros_like(x_enc).to(device)
    model_call = None
    profile_inputs = None

    try:
        # 尝试 TimeMixerPP 的调用方式
        # PyPOTS 模型通常接受一个字典作为输入
        _ = model({
            "X": x_enc,
            "missing_mask": missing_mask
        })

        def call_timemixerpp():
            return model({
                "X": x_enc.clone(),
                "missing_mask": missing_mask.clone()
            })

        model_call = call_timemixerpp
        # torchinfo 不支持字典输入，所以我们用元组传递单个张量进行分析
        profile_inputs = ({"X": x_enc, "missing_mask": missing_mask},)
        print("  [Info] Using TimeMixerPP (PyPOTS) call style")

    except Exception:
        try:
            # 尝试 FourierGNN 的调用方式 (B, N, L)
            _ = model(x_enc.permute(0, 2, 1))

            def call_fouriergnn():
                return model(x_enc.permute(0, 2, 1).clone())

            model_call = call_fouriergnn
            profile_inputs = (x_enc.permute(0, 2, 1),)
            print("  [Info] Using FourierGNN call style (x_enc permuted)")

        except Exception:
            try:
                # 尝试 TimeFilter 的调用方式
                _ = model(x_enc, masks, is_training=False)

                def call_timefilter():
                    return model(x_enc.clone(), masks.clone(), is_training=False)

                model_call = call_timefilter
                profile_inputs = (x_enc, masks, False)
                print("  [Info] Using TimeFilter call style")

            except Exception:
                try:
                    # 尝试使用完整的4个参数，适用于Transformer类模型
                    _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                    def call_full():
                        return model(x_enc.clone(), x_mark_enc.clone(), x_dec.clone(), x_mark_dec.clone())

                    model_call = call_full
                    profile_inputs = (x_enc, x_mark_enc, x_dec, x_mark_dec)
                    print("  [Info] Using full 4-param call style")
                except TypeError:
                    # 如果失败，则回退到只使用x_enc，适用于DLinear等简单模型
                    _ = model(x_enc)  # 验证此调用是否可行

                    def call_simple():
                        return model(x_enc.clone())

                    model_call = call_simple
                    profile_inputs = (x_enc,)
                    print("  [Info] Using simple x_enc call style")

    # --- 3. 使用确定的调用方式进行所有测试 ---
    if device.type == 'cpu':
        with torch.no_grad():
            try:
                # 使用 torchinfo 来分析模型
                # 注意：profile_inputs 是一个元组，需要用 * 解包
                summary = torchinfo.summary(model, input_data=profile_inputs, verbose=0)
                params = summary.total_params
                macs = summary.total_mult_adds
            except Exception as e:
                print(f"  [Warning] torchinfo failed: {e}")
                macs = 0
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        stats['Params (M)'] = round(params / 1e6, 4)
        stats['MACs (G)'] = round(macs / 1e9, 4)
        stats['FLOPs (G)'] = round(2 * macs / 1e9, 4)

    with torch.inference_mode():
        warmup_iters = 20 if device.type == 'cuda' else 10
        for _ in range(warmup_iters):
            _ = model_call()

        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        for _ in range(iterations):
            _ = model_call()
        if device.type == 'cuda':
            torch.cuda.synchronize()
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
    "Koopa": {"class": Koopa, "params": {}},
    "TimeFilter": {"class": TimeFilter, "params": {}},
    "FourierGNN": {"class": FourierGNN, "params": {}},
    "CONTIME": {"class": CONTIME, "params": {}},
    "LinOSS": {"class": LinOSS, "params": {}},
    "S_D_Mamba": {"class": S_D_Mamba, "params": {}},
    "TimePro": {"class": TimePro, "params": {}},
    "DeepEDM": {"class": DeepEDM, "params": {}},
    "SimpleTM": {"class": SimpleTM, "params": {}},
    "TQNet": {"class": TQNet, "params": {}},
    "ModernTCN": {"class": ModernTCN, "params": {}},
    "FilterNet": {"class": FilterNet, "params": {}},
    "NFM": {"class": NFM, "params": {}},
    "TimeKAN": {"class": TimeKAN, "params": {}},
    "TimeMixerPP": {"class": TimeMixerPP, "params": {}},
    "SOFTS": {"class": SOFTS, "params": {}},
}

# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    # +++ 新增代码 +++
    # --- 从用户输入获取要测试的模型名称 ---
    available_models = list(MODELS_TO_TEST.keys())
    print("Available models to test:", ", ".join(available_models))
    target_model_name = input("请输入您想要测试的模型名称: ")

    # --- 筛选要测试的模型 ---
    if target_model_name in MODELS_TO_TEST:
        MODELS_TO_TEST = {target_model_name: MODELS_TO_TEST[target_model_name]}
    else:
        print(f"错误：模型 '{target_model_name}' 不在可测试的模型列表中。程序将退出。")
        exit()
    # +++ 新增代码结束 +++
    # +++ 使用 pandas 生成正确的时间序列 +++
    # 1. 定义起始时间和数据点数量
    start_date = "2023-01-01 00:00:00"
    num_rows = 1166  # 确保训练集足够大的最小行数

    # 2. 'h' 表示每小时一个数据点，可以根据需要更改为 'min'（分钟）或 'S'（秒）
    dates = pd.to_datetime(pd.date_range(start=start_date, periods=num_rows, freq='h'))

    # 3. 创建 DataFrame 并保存到 dummy.csv
    dummy_df = pd.DataFrame({
        'date': dates,
        'OT': range(num_rows)  # 使用简单递增的数值
    })
    dummy_df.to_csv('dummy.csv', index=False)
    devices_to_test = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device('cuda'))

    print(f"{'=' * 60}\nStarting benchmark on devices: {[d.type for d in devices_to_test]}")
    print(f"CPU Threads: {torch.get_num_threads()}")
    if 'cuda' in [d.type for d in devices_to_test]: print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"{'=' * 60}")

    # +++ 修改后 (最终的、保证运行的版本) +++
    BASE_CONFIGS = {
        # --- 核心任务参数 ---
        'task_name': 'long_term_forecast',
        'is_training': False,

        # --- 模型结构与超参数 (大部分模型通用) ---
        'n_heads': 8,
        'd_layers': 1,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False,
        'distil': True,
        'moving_avg': 25,
        'p_hidden_dims': [128, 128],  # for TiDE
        'p_hidden_layers': 2,  # for TiDE
        'num_kernels': 6,  # for TimesNet

        # --- 数据与路径相关的虚拟参数 (为满足模型初始化需求) ---
        'data': 'custom',
        'root_path': './',
        'data_path': 'dummy.csv',
        'features': 'M',
        'target': 'OT',
        'embed': 'timeF',
        'freq': 'h',
        'checkpoints': './checkpoints/',
        'label_len': CONFIG['label_len'],

        # --- 解决本次报错的最终关键参数 ---
        'seasonal_patterns': 'Monthly',
        'augmentation_ratio': 0,
        'num_workers': 0,
        'num_class': 1,
        'factor': 3,

        # --- 为 TimeFilter 模型添加的缺失参数 ---
        'patch_len': 16,
        'stride': 8,
        'revin': True,
        'alpha': 0.25,  # 这是TimeFilter的alpha，CONTIME有自己的alpha
        'top_p': 1.0,
        'pos': 0,

        # +++ 为 CONTIME 模型新增的参数 (最终修复) +++
        'beta': 0.01,  # CONTIME 需要
        'interpolation': 'cubic'  # CONTIME 需要
    }

    all_results = []
    for model_name, model_info in MODELS_TO_TEST.items():
        for e_layers in E_LAYERS_LIST:
            for d_model in D_MODEL_LIST:
                d_ff = d_model * 2
                print(f"--- Testing {model_name} | e_layers={e_layers} | d_model={d_model} ---")

                combined_run_info = {'Model': model_name, 'e_layers': e_layers, 'd_model': d_model}
                current_config_dict = BASE_CONFIGS.copy()

                # *** 核心修复 1: 将batch_size添加到模型配置中 ***
                current_config_dict.update({
                    'batch_size': CONFIG['batch_size'],
                    'seq_len': CONFIG['seq_len'], 'pred_len': CONFIG['pred_len'], 'enc_in': CONFIG['channels'],
                    'dec_in': CONFIG['channels'], 'c_out': CONFIG['channels'], 'e_layers': e_layers,
                    'd_model': d_model, 'd_ff': d_ff,
                })
                current_config_dict.update(model_info['params'])

                try:
                    # --- 核心修改：为不同构造函数的模型添加特殊处理 ---
                    if model_name == "FourierGNN":
                        # FourierGNN 需要独立的参数
                        model = model_info['class'](
                            pre_length=current_config_dict['pred_len'],
                            embed_size=current_config_dict['d_model'],
                            feature_size=current_config_dict['enc_in'],
                            seq_length=current_config_dict['seq_len'],
                            hidden_size=current_config_dict['d_ff']
                        )
                    elif model_name == "LinOSS":
                        # LinOSS 需要独立的参数
                        model = model_info['class'](
                            input_dim=current_config_dict['enc_in'],
                            hidden_dim=current_config_dict['d_model'],
                            output_dim=current_config_dict['c_out'],
                            vf_depth=2,  # 使用默认值
                            vf_width=128,  # 使用默认值
                            num_blocks=current_config_dict['e_layers'],
                            ssm_dim=64,  # 使用默认值
                            ssm_blocks=2,  # 使用默认值
                            solver='rk4',  # 使用默认值
                            step_size=0.1,  # 使用默认值
                            use_imex=False,  # 使用默认值
                            imex_ssm_blocks=0,  # 使用默认值
                            lambd=0.0,  # 使用默认值
                            scale=1.0,  # 使用默认值
                            t_final=current_config_dict['seq_len'],  # 将序列长度作为t_final
                        )
                    elif model_name == "NFM":
                        # NFM 需要独立的参数
                        model = model_info['class'](
                            seq_len=current_config_dict['seq_len'],
                            pred_len=current_config_dict['pred_len'],
                            enc_in=current_config_dict['enc_in'],
                            d_model=current_config_dict['d_model'],
                            e_layers=current_config_dict['e_layers'],
                            d_ff=current_config_dict['d_ff'],
                            # 以下为NFM的特有参数，使用默认值
                            M=1024,
                            out_channel=16,
                            r=4,
                            ll=1,
                            attn='softmax',
                            num_L=3,
                            num_S=3,
                            num_heads=4
                        )
                    elif model_name == "TimeMixerPP":
                        # TimeMixerPP 需要独立的参数
                        model = model_info['class'](
                            n_steps=current_config_dict['seq_len'],
                            n_features=current_config_dict['enc_in'],
                            n_layers=current_config_dict['e_layers'],
                            d_model=current_config_dict['d_model'],
                            d_ffn=current_config_dict['d_ff'],
                            top_k=current_config_dict.get('top_k', 5),  # 从配置获取或使用默认值
                            n_heads=current_config_dict.get('n_heads', 8),
                            n_kernels=current_config_dict.get('num_kernels', 6),
                            dropout=current_config_dict.get('dropout', 0.1),
                            # 使用PyPOTS库的默认值
                            channel_mixing=True,
                            channel_independence=True,
                            downsampling_layers=3,
                            downsampling_window=2,
                            apply_nonstationary_norm=False,
                            # 评测脚本需要的参数
                            epochs=1  # 设为1，因为我们不在这里训练
                        )
                    elif model_name == "CONTIME":
                        # 1. 将模型名称转为小写，以匹配其内部的 'contime' 判断
                        current_config_dict['model'] = model_name.lower()

                        # 2. 添加其必需的 h_channels 和 dataset 属性
                        current_config_dict['h_channels'] = current_config_dict['d_model']
                        current_config_dict['dataset'] = current_config_dict['data']

                        # 3. CONTIME 有一个独立的 alpha 参数，这里设置为其默认值
                        current_config_dict['alpha'] = 0.8

                        configs_obj = SimpleNamespace(**current_config_dict)
                        model = model_info['class'](
                            configs_obj,  # 第一个参数是 args 对象
                            input_channels=current_config_dict['enc_in'],
                            output_channels=current_config_dict['c_out'],
                            device=torch.device('cpu')  # 先在CPU上初始化
                        )
                    else:
                        # 其他模型使用通用的 configs 对象
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
    # +++ 修改下面这行代码 +++
    output_filename = f"results/benchmark_results_{target_model_name}.csv"
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
    print(df.head().to_string())
