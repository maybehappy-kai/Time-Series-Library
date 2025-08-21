import sys
import os

# =====================================================================================
# ISOLATED SUBMODULE IMPORTER (v4 - Proactive Cache Seeding & Hard Reset)
# =====================================================================================

def clear_module_cache(module_name):
    """
    搜索并强制删除 sys.modules 中所有以 module_name 开头的缓存键。
    这是确保为具有冲突名称的子模块（如 'layers'）提供干净导入环境的最稳健方法。
    """
    keys_to_delete = [key for key in sys.modules if key.startswith(module_name)]
    for key in keys_to_delete:
        try:
            del sys.modules[key]
        except KeyError:
            pass

# --- Wrapper for TimeFilter ---
original_path_tf = list(sys.path)
clear_module_cache('layers')  # 1. 导入前强制清理
try:
    submodule_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TimeFilter')
    sys.path.insert(0, submodule_root)
    import layers  # 2. 关键一步：主动导入正确的模块以抢占缓存
    from models.TimeFilter.models.TimeFilter import Model as TimeFilter
finally:
    sys.path[:] = original_path_tf
    clear_module_cache('layers')  # 3. 导入后再次清理，确保环境干净

# --- Wrapper for CONTIME ---
# (这个模块没有 'layers' 冲突, 所以不需要特殊处理)
original_path_ct = list(sys.path)
try:
    submodule_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CONTIME')
    sys.path.insert(0, submodule_root)
    from control_tower import Model_selection_part as CONTIME
finally:
    sys.path[:] = original_path_ct

# --- Wrapper for S_D_Mamba ---
original_path_sdm = list(sys.path)
clear_module_cache('layers')
try:
    submodule_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'S_D_Mamba')
    sys.path.insert(0, submodule_root)
    import layers
    from models.S_D_Mamba.model.S_Mamba import Model as S_D_Mamba
finally:
    sys.path[:] = original_path_sdm
    clear_module_cache('layers')

# --- Wrapper for TimePro ---
original_path_tp = list(sys.path)
try:
    submodule_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TimePro')
    sys.path.insert(0, submodule_root)
    from models.TimePro.model.TimePro import Model as TimePro
finally:
    sys.path[:] = original_path_tp

# --- Wrapper for DeepEDM ---
original_path_dedm = list(sys.path)
clear_module_cache('layers')
try:
    submodule_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DeepEDM')
    sys.path.insert(0, submodule_root)
    import layers
    from models.DeepEDM.models.DeepEDM import Model as DeepEDM
finally:
    sys.path[:] = original_path_dedm
    clear_module_cache('layers')

# --- Wrapper for SimpleTM ---
original_path_stm = list(sys.path)
clear_module_cache('layers')
try:
    submodule_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SimpleTM')
    sys.path.insert(0, submodule_root)
    import layers
    from models.SimpleTM.model.SimpleTM import Model as SimpleTM
finally:
    sys.path[:] = original_path_stm
    clear_module_cache('layers')