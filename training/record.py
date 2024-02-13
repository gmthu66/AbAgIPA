import os
import json
import os.path as osp
import numpy as np
from typing import List


class ListAverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val=None, batch_size=1, no_batch=False, n=1):
        batch_size = 1 if no_batch else batch_size
        if isinstance(val, list):
            self.sum += sum(val) / batch_size
        else:
            self.sum += val / batch_size
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class CSVWriter(object):

    def __init__(self, csv_fpath, columns, overwrite):
        self.csv_fpath = csv_fpath
        self.columns = columns

        if os.path.isfile(self.csv_fpath) and overwrite:
            os.remove(csv_fpath)

        if not os.path.isfile(self.csv_fpath):
            # write columns
            with open(self.csv_fpath, 'w') as handles:
                handles.write(','.join(self.columns) + '\n')

        self.values = {key: '' for key in self.columns}

    def add_scalar(self, name, value):
        assert name in self.columns
        self.values[name] = value

    def write(self):
        with open(self.csv_fpath, 'a') as handles:
            handles.write(
                ','.join([str(self.values[key])
                          for key in self.columns]) + '\n')
        self.values = {key: '' for key in self.columns}


class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def clean_config(config, user_config=None):
    repeat_keys, new_config = [], {}
    masked_keys = ['h5', 'bert', 'token']

    def listNotInStr(ls, ss):
        for l_ in ls:
            if l_ in ss:
                return False
        return True
    if not isinstance(config, dict):  config = vars(config)
    if not isinstance(user_config, dict):  user_config = vars(user_config)

    if user_config is not None:
        for k_ in config.keys():
            if k_ in user_config.keys():
                repeat_keys.append(k_)
                new_config[k_] = user_config[k_]
            else:
                if listNotInStr(masked_keys, k_):
                    new_config[k_] = config[k_]
    else:
        for k_ in config.keys():
            if listNotInStr(masked_keys, k_): new_config[k_] = config[k_]

    for kk_ in list(set(user_config.keys() - config.keys())):
        if listNotInStr(masked_keys, kk_) and ('__' not in kk_):  new_config[kk_] = user_config[kk_]
    if isinstance(new_config, dict):  new_config = Namespace(**new_config)
    return new_config


def save_dict(dic, fpath):
    js_obj = json.dumps(dic)
    with open(fpath, 'w') as w:
        w.write(js_obj)


def load_dict(fpath):
    js_obj = json.load(open(fpath))
    return js_obj


def remove_sbatch_logs():
    home_dir = os.environ['HOME']
    for f_ in [ff for ff in os.listdir(home_dir) if osp.isfile(ff) and (ff.startswith('slurm') and ff.endswith('.out'))]:
        os.remove(osp.join(home_dir, f_))
    print(f'clean sbatch log files in {home_dir}')


def save_curves_array(curve_dict, npz_fpath):
    new_dict = {}
    for k_ in curve_dict:
        if isinstance(curve_dict[k_], np.ndarray):
            new_dict[k_] = curve_dict[k_]
    np.savez(npz_fpath, **new_dict)


def delete_files_with_prefix(folder_path, prefix):
    # 获取文件夹中的文件列表
    if osp.exists(folder_path):
        files = os.listdir(folder_path)        
        # 删除以指定前缀开头的文件
        for fpath in files:
            if fpath.startswith(prefix):
                file_path = os.path.join(folder_path, fpath)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
