import torch
import time
import pandas as pd
import os
from thop import profile
import warnings

# --- 忽略一些不影响结果的警告 ---
warnings.filterwarnings("ignore")

# --- 1. 导入需要测试的模型 ---
# 很多模型可以直接从TSLib中导入。
# 我们先以TSLib自带的DLinear, PatchTST, iTransformer为例。
from models.DLinear import Model as DLinear
from models.PatchTST import Model as PatchTST
from models.iTransformer import Model as iTransformer


# --- 2. 定义核心评测函数 ---
def get_model_stats(model, model_name, config, input_tensor, iterations=50):
    """
    计算模型的各项计算性能指标。

    Args:
        model: 已实例化的PyTorch模型。
        model_name: 模型名称（字符串）。
        config: 包含模型超参数的字典。
        input_tensor: 用于测试的输入张量。
        iterations: 推理时间测试的迭代次数。

    Returns:
        一个包含所有指标的字典。
    """
    stats = {}
    device = input_tensor.device

    # === 指标1: 参数量 (Params) ===
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    stats['Params (M)'] = round(params_m, 4)

    # === 指标2 & 3: MACs 和 FLOPs ===
    # 注意：需要根据模型的具体forward参数来调整这里的输入
    # 我们将尝试适配TSLib中常见的Transformer类和非Transformer类模型
    model.eval()
    with torch.no_grad():
        try:
            # 尝试非Transformer类模型的forward签名
            # 例如 DLinear(self, x)
            macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        except Exception:
            try:
                # 尝试Transformer类模型的forward签名
                # 例如 iTransformer(self, x_enc, x_mark_enc, x_dec, x_mark_dec)
                # 我们创建符合要求的dummy inputs
                batch_size, pred_len, channels = config['batch_size'], config['pred_len'], config['channels']
                dec_input = torch.zeros((batch_size, pred_len, channels)).to(device)
                enc_mark = torch.zeros((batch_size, input_tensor.shape[1], 0)).to(device)  # mark_enc/dec通常不需要
                dec_mark = torch.zeros((batch_size, pred_len, 0)).to(device)

                macs, _ = profile(model, inputs=(input_tensor, enc_mark, dec_input, dec_mark), verbose=False)
            except Exception as e:
                print(f"  [Warning] MACs/FLOPs calculation failed for {model_name}: {e}")
                macs = 0

    stats['MACs (G)'] = round(macs / 1e9, 4)
    stats['FLOPs (G)'] = round(2 * macs / 1e9, 4)  # 理论上 FLOPs ≈ 2 * MACs

    # === 指标4 & 5: 推理时间 (Inference Time) 和 吞吐量 (Throughput) ===
    # 在CPU上测试
    model.eval()
    with torch.no_grad():
        # 预热(Warm-up) - 让CPU进入稳定工作状态
        for _ in range(10):
            _ = model(input_tensor.clone())

        # 正式测试
        start_time = time.time()
        for _ in range(iterations):
            _ = model(input_tensor.clone())
        end_time = time.time()

    total_time = end_time - start_time
    avg_inference_time_ms = (total_time / iterations) * 1000
    throughput = iterations / total_time
    stats['Inference Time (ms/batch)'] = round(avg_inference_time_ms, 4)
    stats['Throughput (batch/s)'] = round(throughput, 4)

    # === 指标6: 峰值内存 (Max Memory) ===
    # 这个指标在CPU上监控比较复杂且不准确，主要用于GPU。
    # 按照学长CPU测试的要求，我们暂时将其报告为0或N/A。
    stats['Max Memory (MB)'] = 0.0

    return stats


# --- 3. 定义测试配置 ---
# 输入张量参数
CONFIG = {
    'batch_size': 8,
    'seq_len': 96,  # timestamps
    'pred_len': 720,
    'channels': 32,
    'device': torch.device("cpu")  # 明确在CPU上测试
}

# 模型超参数组合
E_LAYERS_LIST = [1, 2, 3]
D_MODEL_LIST = [24, 48, 64, 128, 256, 512]

# 需要测试的模型列表
# Key是模型名称，Value是模型类和它的初始化参数字典
MODELS_TO_TEST = {
    "DLinear": {
        "class": DLinear,
        "params": {"individual": False}  # DLinear的特定参数
    },
    "iTransformer": {
        "class": iTransformer,
        "params": {}  # iTransformer没有额外的固定参数
    },
    "PatchTST": {
        "class": PatchTST,
        "params": {"patch_len": 16, "stride": 8}  # PatchTST的特定参数
    },
    # 后续我们会在这里添加更多模型
}

# --- 4. 主执行逻辑 ---
if __name__ == '__main__':
    results = []

    # 创建一个固定的输入张量
    input_tensor = torch.randn(
        CONFIG['batch_size'], CONFIG['seq_len'], CONFIG['channels']
    ).to(CONFIG['device'])

    for model_name, model_info in MODELS_TO_TEST.items():
        for e_layers in E_LAYERS_LIST:
            for d_model in D_MODEL_LIST:
                d_ff = d_model * 4  # TSLib中d_ff通常是d_model的4倍

                print(f"--- Testing {model_name} | e_layers={e_layers} | d_model={d_model} ---")

                # 构建模型配置字典
                model_config = {
                    'seq_len': CONFIG['seq_len'],
                    'pred_len': CONFIG['pred_len'],
                    'enc_in': CONFIG['channels'],
                    'c_out': CONFIG['channels'],  # DLinear等模型使用c_out
                    'e_layers': e_layers,
                    'd_model': d_model,
                    'd_ff': d_ff,
                }
                model_config.update(model_info['params'])  # 添加模型的特定参数

                try:
                    # 筛选出模型类__init__需要的参数
                    model_class = model_info['class']
                    valid_keys = model_class.__init__.__code__.co_varnames
                    filtered_config = {k: v for k, v in model_config.items() if k in valid_keys}

                    # 实例化模型
                    model = model_class(**filtered_config).to(CONFIG['device'])

                    # 获取性能指标
                    stats = get_model_stats(model, model_name, CONFIG, input_tensor)

                    # 记录结果
                    run_info = {
                        'Model': model_name,
                        'e_layers': e_layers,
                        'd_model': d_model,
                    }
                    run_info.update(stats)
                    results.append(run_info)

                except Exception as e:
                    print(f"  [ERROR] Failed for {model_name} with config {e_layers, d_model}. Error: {e}")
                    # 记录失败信息
                    results.append({
                        'Model': model_name,
                        'e_layers': e_layers,
                        'd_model': d_model,
                        'Error': str(e)
                    })

    # --- 5. 保存结果 ---
    if not os.path.exists('results'):
        os.makedirs('results')

    df = pd.DataFrame(results)
    df.to_csv("results/benchmark_results.csv", index=False)

    print("\n\nBenchmark finished. Results saved to results/benchmark_results.csv")
    print(df.head())