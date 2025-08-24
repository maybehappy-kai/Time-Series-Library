import subprocess
import time
import sys
import json
import pandas as pd

# 从 benchmark.py 脚本中复制模型列表，确保一致性
MODELS_TO_TEST = [
    "TimePro", "ModernTCN",
    "NFM", "TimeMixerPP"
]

def run_profiling():
    """
    为每个模型运行 benchmark.py 的调试模式，并记录耗时。
    """
    print("--- 开始对所有模型进行性能分析（调试模式） ---")
    timings = {}

    for model_name in MODELS_TO_TEST:
        print(f"\n[INFO] 正在测试模型: {model_name}...")
        start_time = time.time()

        # 准备运行 benchmark.py 的命令
        command = [sys.executable, 'benchmark.py']

        try:
            # --- 关键修正：直接传递字符串给 input 参数 ---
            # 当 text=True 时，subprocess 会自动处理编码
            process = subprocess.run(
                command,
                input=model_name,  # <--- 已修正此行
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 设置10分钟超时
            )

            end_time = time.time()
            duration = end_time - start_time
            timings[model_name] = duration
            print(f"[SUCCESS] {model_name} 测试完成，耗时: {duration:.2f} 秒")

            # 打印 benchmark.py 输出的最后几行，以确认它成功生成了文件
            output_lines = process.stdout.strip().split('\n')
            if len(output_lines) > 3:
                 print("  [Output] " + "\n  [Output] ".join(output_lines[-3:]))

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 模型 {model_name} 测试失败！")
            print("  错误信息 (stdout):")
            print(e.stdout)
            print("  错误信息 (stderr):")
            print(e.stderr)
            timings[model_name] = -1 # 标记为失败
        except subprocess.TimeoutExpired:
            print(f"[ERROR] 模型 {model_name} 测试超时！")
            timings[model_name] = -1 # 标记为失败
        except Exception as e:
            print(f"[ERROR] 运行 {model_name} 时发生未知错误: {e}")
            timings[model_name] = -1 # 标记为失败

    return timings

def save_timings(timings):
    """
    将耗时结果保存到 JSON 文件中，并打印一个漂亮的表格。
    """
    print("\n--- 性能分析结果汇总 ---")

    # 过滤掉失败的测试
    successful_timings = {k: v for k, v in timings.items() if v != -1}
    failed_models = [k for k, v in timings.items() if v == -1]

    if successful_timings:
        # 使用 pandas 创建和显示表格
        df = pd.DataFrame(list(successful_timings.items()), columns=['Model', 'Time (s)'])
        df = df.sort_values(by='Time (s)', ascending=False).reset_index(drop=True)
        print(df.to_string())

        # 保存为 JSON 文件，用于下一步
        output_file = 'model_timings.json'
        with open(output_file, 'w') as f:
            json.dump(successful_timings, f, indent=4)
        print(f"\n[INFO] 耗时数据已成功保存到 {output_file}")
    else:
        print("没有模型测试成功。")

    if failed_models:
        print("\n--- 失败的模型 ---")
        for model in failed_models:
            print(f"- {model}")

if __name__ == '__main__':
    all_timings = run_profiling()
    save_timings(all_timings)