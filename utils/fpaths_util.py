import os


def find_pdb_files(directory, ext_key=".pdb"):
    pdb_files = []
    
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for fpath in files:
            if 'temp' in fpath:  continue
            # 检查文件是否以 ".pdb" 为后缀
            if fpath.endswith(ext_key):
                # 构建文件的绝对路径并添加到列表
                pdb_files.append(os.path.abspath(os.path.join(root, fpath)))
    return pdb_files


def search_files_by_string(directoryls, search_string, ext=None):
    matching_files = []

    # 遍历目录及子目录
    for directory in directoryls:
        for root, dirs, files in os.walk(directory):
            for fpath in files:
                # 检查文件名是否包含搜索字符串
                if search_string in fpath:
                    # 获取文件的绝对路径
                    file_path = os.path.join(root, fpath)
                    matching_files.append(file_path)
    if ext is not None:  matching_files = [fpath for fpath in matching_files if fpath.endswith(ext)]
    return matching_files


def basename_all_digits(file_path):
    # 获取文件的基本名称和后缀
    base_name, extension = os.path.splitext(os.path.basename(file_path))
    # 检查基本名称是否由数字组成
    return base_name.isdigit()


def check_string_inpath(file_path, keystring='af_oas_paired'):
    return (keystring in file_path)


def check_objls_inbasename(file_path, pdbobj_ls):
    for pdbobj in pdbobj_ls:
        if pdbobj in file_path:
            return True
    return False


def get_pdbobjs_ofDepthDir(directory, extension=".pdb"):
    pdb_files, pdb_objs = [], []
    # 递归遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for fpath in files:
            if fpath.endswith(extension) and not fpath.startswith('.'):
                if 'temp' not in os.path.join(root, fpath):
                    pdb_files.append(os.path.join(root, fpath))
                    pdb_objs.append(fpath.split('.')[0])
    return pdb_files, pdb_objs
