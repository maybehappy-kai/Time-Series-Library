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
    此函数现在可以处理 'from module import ...' 和 'import module' 两种情况。
    """
    print(f"\n--- Processing model: {model_name} ---")

    # 1. 自动查找该项目下的所有顶级模块（文件夹和.py文件）
    top_level_modules = set()
    try:
        for item in os.listdir(project_root):
            # 忽略隐藏文件和__pycache__
            if item.startswith('.') or item == '__pycache__':
                continue
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
    modules_pattern = '|'.join(re.escape(m) for m in top_level_modules)

    # 2. 遍历项目中的所有 .py 文件
    for dirpath, _, filenames in os.walk(project_root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(dirpath, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"  [Error] Could not read {filepath}: {e}")
                continue

            original_content = content
            modified = False

            # 3. 计算当前文件相对于项目根的深度，并生成相应的前置点
            relative_dir = os.path.relpath(dirpath, project_root)
            depth = 0 if relative_dir == '.' else len(relative_dir.split(os.sep))
            dots = "." * (depth + 1)

            # --- 修正第一步: 处理 'from project_module ...' ---
            # 正则表达式：匹配以 'from' 开头，后跟一个顶级模块名，再跟一个点或空格的行
            pattern_from = re.compile(r"^(from\s+)(" + modules_pattern + r")([.\s])", re.MULTILINE)

            def replacer_from(match):
                # group(1)="from ", group(2)=模块名, group(3)=分隔符(. 或 空格)
                return f"{match.group(1)}{dots}{match.group(2)}{match.group(3)}"

            new_content = pattern_from.sub(replacer_from, content)
            if new_content != content:
                modified = True
                content = new_content

            # --- 修正第二步 (新增): 处理 'import project_module' ---
            # 正则表达式：匹配以 'import' 开头，后跟一个顶级模块名，且该行结尾没有其他字符
            pattern_import = re.compile(r"^(import\s+)(" + modules_pattern + r")(\s*)$", re.MULTILINE)

            def replacer_import(match):
                # group(1)="import ", group(2)=模块名
                # 将 'import my_module' 转换为 'from . import my_module'
                return f"from {dots} import {match.group(2)}"

            new_content = pattern_import.sub(replacer_import, content)
            if new_content != content: # 与上一步的content比较
                modified = True
                content = new_content


            # 7. 如果文件内容有变化，则写回文件
            if modified:
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