import os
import re

# --- 配置区 ---
# 我们要处理的所有新集成模型的文件夹名称列表
MODEL_NAMES = [
    "TimeFilter", "FourierGNN", "CONTIME", "linoss", "S_D_Mamba",
    "TimePro", "DeepEDM", "SimpleTM", "TQNet", "ModernTCN",
    "FilterNet", "NFM", "TimeKAN", "SOFTS"
]
# 存放这些模型的主目录
MODELS_BASE_DIR = "models"


# --- 配置区结束 ---

def fix_imports_in_project(model_name, project_root):
    """
    修复单个模型项目中的所有Python文件的绝对导入问题。
    """
    print(f"\n--- Processing model: {model_name} ---")

    # 1. 自动查找该项目下的所有顶级模块（文件夹和.py文件）
    top_level_modules = set()
    try:
        for item in os.listdir(project_root):
            if item.endswith('.py'):
                top_level_modules.add(item.replace('.py', ''))
            elif os.path.isdir(os.path.join(project_root, item)) and \
                    any(f.endswith('.py') for f in os.listdir(os.path.join(project_root, item))):
                top_level_modules.add(item)
    except FileNotFoundError:
        print(f"[ERROR] Project root not found: {project_root}")
        return

    if not top_level_modules:
        print(f"[Warning] No top-level Python modules found in {project_root}. Skipping.")
        return

    print(f"Found top-level modules: {', '.join(sorted(list(top_level_modules)))}")

    # 2. 遍历项目中的所有 .py 文件
    for dirpath, _, filenames in os.walk(project_root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(dirpath, filename)

            # 3. 计算当前文件相对于项目根的深度
            relative_dir = os.path.relpath(dirpath, project_root)
            depth = 0 if relative_dir == '.' else len(relative_dir.split(os.sep))

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"  [Error] Could not read {filepath}: {e}")
                continue

            original_content = content

            # 4. 构造正则表达式，用于匹配该项目的所有顶级模块
            # 例如: from (moduleA|moduleB|moduleC)...
            modules_pattern = '|'.join(re.escape(m) for m in top_level_modules)
            # 正则表达式：匹配以 'from' 开头，后跟一个顶级模块名，再跟一个点或空格的行
            pattern = re.compile(r"^(from\s+)(" + modules_pattern + r")([.\s])", re.MULTILINE)

            # 5. 定义替换函数
            def replacer(match):
                # match.group(1) is "from "
                # match.group(2) is the module name (e.g., "utils")
                # match.group(3) is the character after (space or dot)

                # 计算需要的前缀点
                # depth 0 (root): one dot '.'
                # depth 1 (subfolder): two dots '..'
                dots = "." * (depth + 1)

                # 返回替换后的字符串
                return f"{match.group(1)}{dots}{match.group(2)}{match.group(3)}"

            # 6. 执行替换
            content = pattern.sub(replacer, content)

            # 7. 如果文件内容有变化，则写回文件
            if content != original_content:
                print(f"  -> Modifying imports in: {os.path.relpath(filepath, MODELS_BASE_DIR)}")
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                except Exception as e:
                    print(f"  [Error] Could not write to {filepath}: {e}")


if __name__ == "__main__":
    print(f"{'=' * 50}\nStarting script to convert absolute imports to relative imports.\n{'=' * 50}")

    for model_name in MODEL_NAMES:
        project_root_path = os.path.join(MODELS_BASE_DIR, model_name)
        if os.path.isdir(project_root_path):
            fix_imports_in_project(model_name, project_root_path)
        else:
            print(f"\n[Warning] Directory for model '{model_name}' not found at '{project_root_path}'. Skipping.")

    print(f"\n{'=' * 50}\nScript finished. All model imports have been processed.\n{'=' * 50}")