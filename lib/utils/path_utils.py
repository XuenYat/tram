"""
路径工具函数，用于动态查找项目根目录
"""
import os


def bootstrap_tram_path(caller_file=None):
    """
    引导函数：找到TRAM根目录并添加到Python路径
    这是一个特殊的函数，用于在导入其他模块之前设置Python路径
    
    Args:
        caller_file: 调用者的文件路径，如果为None则自动检测
    """
    import sys
    
    if caller_file is None:
        import inspect
        # 获取调用者的目录
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
    
    current_dir = os.path.dirname(os.path.abspath(caller_file))
    
    # 向上搜索，直到找到包含train.py的目录
    search_dir = current_dir
    for _ in range(10):
        if os.path.exists(os.path.join(search_dir, 'train.py')):
            if search_dir not in sys.path:
                sys.path.insert(0, search_dir)
            return search_dir
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:
            break
        search_dir = parent_dir
    
    raise ValueError(f"Could not find TRAM root directory starting from {current_dir}")


def simple_bootstrap_tram_path(caller_file_path):
    """
    简化的引导函数，直接接受调用者文件路径
    
    Args:
        caller_file_path: 调用者的 __file__ 路径
        
    Returns:
        str: TRAM项目根目录路径
    """
    import sys
    current_dir = os.path.dirname(os.path.abspath(caller_file_path))
    
    # 向上搜索，直到找到包含train.py的目录
    search_dir = current_dir
    for _ in range(10):
        if os.path.exists(os.path.join(search_dir, 'train.py')):
            if search_dir not in sys.path:
                sys.path.insert(0, search_dir)
            return search_dir
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:
            break
        search_dir = parent_dir
    
    raise ValueError(f"Could not find TRAM root directory starting from {current_dir}")


def find_tram_root(start_dir=None):
    """
    找到TRAM项目根目录（包含train.py的目录）
    
    Args:
        start_dir: 开始搜索的目录，如果None则使用调用者的目录
    
    Returns:
        str: TRAM项目根目录的绝对路径
    """
    if start_dir is None:
        # 获取调用者的目录
        import inspect
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
        start_dir = os.path.dirname(os.path.abspath(caller_file))
    
    # 向上搜索，直到找到包含train.py的目录
    search_dir = start_dir
    for _ in range(10):  # 最多向上搜索10层
        if os.path.exists(os.path.join(search_dir, 'train.py')):
            return search_dir
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:  # 到达根目录
            break
        search_dir = parent_dir
    
    # 如果没找到，抛出异常
    raise ValueError(f"Could not find TRAM root directory (with train.py) starting from {start_dir}")


def find_vggt_root(start_dir=None):
    """
    找到VGGT项目根目录（包含vggt包的目录）
    
    Args:
        start_dir: 开始搜索的目录，如果None则使用调用者的目录
    
    Returns:
        str: VGGT项目根目录的绝对路径
    """
    try:
        tram_root = find_tram_root(start_dir)
        
        # 优先查找thirdparty目录下的vggt
        vggt_in_thirdparty = os.path.join(tram_root, 'thirdparty', 'vggt')
        if os.path.exists(vggt_in_thirdparty):
            return vggt_in_thirdparty
        
        # 备用位置：workspace级别的vggt
        workspace_root = os.path.dirname(tram_root)  # /workspace
        vggt_in_workspace = os.path.join(workspace_root, 'vggt')
        if os.path.exists(vggt_in_workspace):
            return vggt_in_workspace
        
        # 如果没找到，尝试其他可能的位置
        possible_paths = [
            os.path.join(tram_root, 'vggt'),  # 直接在tram根目录下
            os.path.join(workspace_root, '..', 'vggt'),  # 上级目录
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        raise ValueError("Could not find VGGT directory")
        
    except Exception as e:
        raise ValueError(f"Could not find VGGT directory: {e}")


def get_pretrain_path(filename):
    """
    获取预训练模型文件的绝对路径
    
    Args:
        filename: 预训练文件名
    
    Returns:
        str: 预训练文件的绝对路径
    """
    tram_root = find_tram_root()
    return os.path.join(tram_root, 'data', 'pretrain', filename)


def ensure_tram_in_path():
    """
    确保TRAM项目根目录在Python路径中
    """
    import sys
    tram_root = find_tram_root()
    if tram_root not in sys.path:
        sys.path.insert(0, tram_root)
    return tram_root


def ensure_vggt_in_path():
    """
    确保VGGT项目根目录在Python路径中
    """
    import sys
    vggt_root = find_vggt_root()
    if vggt_root not in sys.path:
        sys.path.insert(0, vggt_root)
    return vggt_root


if __name__ == "__main__":
    # 测试
    try:
        tram_root = find_tram_root()
        print(f"✅ TRAM root found: {tram_root}")
        
        vggt_root = find_vggt_root()
        print(f"✅ VGGT root found: {vggt_root}")
        
        # 测试一些预训练文件路径
        test_files = [
            'DEVA-propagation.pth',
            'sam_vit_h_4b8939.pth',
            'camcalib_sa_biased_l2.ckpt'
        ]
        
        for filename in test_files:
            try:
                path = get_pretrain_path(filename)
                exists = os.path.exists(path)
                print(f"{'✅' if exists else '❌'} {filename}: {path}")
            except Exception as e:
                print(f"❌ {filename}: Error - {e}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
