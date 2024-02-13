import os
import torch
import numpy as np
import os.path as osp
from model.interface import read_dict
from torch.utils.data.sampler import Sampler
from database.inter_data import insight_inter_noninterDf, get_fpaths_databaseDf
from os.path import basename
# torch.manual_seed(22)
# random.seed(22)


class SamplerDataFrame(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None, partion=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples, self.partion = num_samples, partion
        if partion == 'train':
            self.data_df = getattr(self.data_source, 'data_df')
        elif partion == 'valid':
            self.data_df = getattr(self.data_source, 'data_df')
        elif partion == 'test':
            self.data_df = getattr(self.data_source, 'data_df')
        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = self.data_df.shape[0]
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        print('now is SamplerDataFrame iter')
        if self.partion == 'train':
            if not self.replacement:
                return iter(list(np.random.permutation(self.data_df.index.to_list())))
            else:
                return iter(list(np.random.choice(self.data_df.index.to_list(), size=self.num_samples, replace=True)))
        elif self.partion in ['test', 'valid']:
            return iter(self.data_df.index.to_list())

    def __len__(self):
        return len(self.data_df.shape[0])


class CustomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None, partion=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples, self.partion = num_samples, partion
        self.idx_dict = getattr(self.data_source, 'idx_dict') if 'idx_dict' in dir(self.data_source) else None
        t_, dict_t, self.weights = 0, {}, []
        if (len(self.idx_dict.keys()) > 1) and (partion != 'test') and (partion != 'valid'):
            for step_, key_ in enumerate(self.idx_dict.keys()):
                dict_t[step_] = len(self.idx_dict[key_])
                t_ += len(self.idx_dict[key_])
            for k_ in dict_t.keys():
                dict_t[k_] = [1 / (dict_t[k_] / t_)] * len(self.idx_dict[k_])
                self.weights.extend(dict_t[k_])
            self.weights = torch.as_tensor(self.weights, dtype=torch.double)
        else:
            dict_t = None
        self.dict_t = dict_t
        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if (self.idx_dict is not None) and self.partion != 'test':
            if len(self.idx_dict.keys()) < 2:
                if self.replacement:
                    return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64,).tolist())
                return iter(torch.randperm(n).tolist())
            else:
                return iter(torch.multinomial(self.weights, self.num_samples, self.replacement,).tolist())  # 进行一个加权随机采样
        elif self.partion in ['test', 'valid']:
            return iter(range(n))

    def __len__(self):
        return len(self.data_source)


def save_pre_embed(embed_data, edge_data, fpath, data_dict={}):
    # data_dict.update({'embed': embed_data, 'edge': edge_data})
    # torch.save(data_dict, fpath)
    if isinstance(embed_data, list):
        fpath = fpath.replace(".pt", ".npz")
        if len(embed_data) == 1:
            np.savez(fpath, embed0=embed_data[0].cpu().numpy(), edge0=edge_data[0].cpu().numpy())
        elif len(embed_data) == 2:
            np.savez(fpath, embed0=embed_data[0].cpu().numpy(), embed1=embed_data[1].cpu().numpy(), edge0=edge_data[0].cpu().numpy(), edge1=edge_data[1].cpu().numpy())


def load_pre_embed(fpath, device=None):
    device = torch.device('cpu') if device is None else device
    def np2tensor(mt, in_device=device):
        return torch.tensor(data=mt, dtype=torch.float32, device=in_device)

    data = np.load(fpath)
    if len(data.files) == 4:
        return [np2tensor(data['embed0']), np2tensor(data['embed1'])], [np2tensor(data['edge0']), np2tensor(data['edge1'])]
    elif len(data.files) == 2:
        return [np2tensor(data['embed0'])], [np2tensor(data['edge0'])]


def rightpdb_size(fpath, trunc_value=30):
    # trunc_value是KB
    return trunc_value <= osp.getsize(fpath) / 1024


def compare_dicts_with_lists(dict1, dict2):
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    # 检查每个关键字对应的值是否一致（考虑列表顺序）
    for key in dict1.keys():
        if sorted(dict1[key]) != sorted(dict2[key]):
            return False
    # 如果关键字和对应值都一致，则返回 True
    return True


def are_lists_equal(list1, list2):
    # 判断列表长度是否相等
    list1, list2 = sorted(list1), sorted(list2)
    if len(list1) != len(list2):
        return False
    # 逐个比较列表元素
    for elem1, elem2 in zip(list1, list2):
        # 判断每个元素是否为字符串并且相等
        if not (isinstance(elem1, str) and isinstance(elem2, str) and elem1 == elem2):
            return False
    # 如果上面的条件都满足，则列表完全一致
    return True


def get_train_valid_test_datadf(data_dirs, valid_ids = None, ddp = None):
    """获取train valid test对应的datadf的字典"""
    fpaths = []
    idx_dict = {}
    if 'valid' in data_dirs.keys():
        if are_lists_equal(data_dirs['train'], data_dirs['valid']):
            assert valid_ids is not None
    inter_pdbobj_ls, noninter_pdbobj_ls, inter_noninter_dict, sabdab_interdf = insight_inter_noninterDf(ddp.inter_noninter_dict_info['inter'], ddp.inter_noninter_dict_info['noninter'], ddp)
    # sabdab_interdf中是包含所有数据信息的
    for step, _dir in enumerate(data_dirs):  # 依据only_struct, 获取oas_paired、oas_unpaired、sabdab以及test数据的index字典, 便于计算每个数据集的大小, 以便于进行数据采样
        sub_fs = [f_ for f_ in os.listdir(_dir) if (('.pdb' in f_) or ('.fasta' in f_)) and not f_.startswith('.')]
        if osp.exists(ddp.error_dict_path) and osp.exists(ddp.right_dict_path):
            sub_fs, valid_ids = _filter_errorfile(_dir, sub_fs, valid_ids)
        # self.check_pdb(_dir, sub_fs)
        if (valid_ids is not None) and (not valid_flag) and (not test_flag):
            sub_fs = sorted(list(set([s_.split('.')[0] for s_ in sub_fs]).difference(valid_ids)))
        elif valid_flag:
            test_flag = False
            sub_fs = sorted(list(set([s_.split('.')[0] for s_ in sub_fs]).intersection(valid_ids)))  # 这里都进行sorted是为了进行datasampler时可复现
        elif test_flag:
            valid_flag = False
            sub_fs = sorted(list(set([s_.split('.')[0] for s_ in sub_fs]).difference(valid_ids)))
        sub_fs = [(f_ + '.pdb') for f_ in sub_fs]
        sub_fs = [osp.join(_dir, f_) for f_ in sub_fs if (f_.endswith('.pdb') and rightpdb_size(osp.join(_dir, f_)))]
        fpaths.extend(sub_fs)
        if len(sub_fs) > 0:
            idx_dict[step] = list(range(st_, st_ + len(sub_fs)))
        if len(sub_fs) > 0:
            st_ = (idx_dict[step][-1] + 1)

    total_inter_noninter_fpath = osp.join(osp.dirname(osp.dirname(__file__)), 'database', 'total_inter_noninter.tsv')
    total_df = get_fpaths_databaseDf(fpaths, ddp, valid_ids, opath=total_inter_noninter_fpath,)  # 需要基于inter pdb_objls以及only_struct objls产生一些noninter的负作用数据



def _filter_errorfile(self, dir_path, f_ls, valid_ids):
    error_dict, right_dict, err_ls, rm_validids = read_dict(self.ddp.error_dict_path), read_dict(self.ddp.right_dict_path), [], []
    for k_ in error_dict.keys():  # error_pdb_dict集中于af_oas_paired子集合中
        err_ls += error_dict[k_]
    temp_fls = [osp.join(dir_path, f_) for f_ in f_ls]
    for f_ in temp_fls:
        if f_ in err_ls:
            f_ls.remove(basename(f_))
            f_ls.remove(basename(f_).replace('.pdb', '.fasta'))
    for _id in valid_ids:
        for e_ in err_ls:
            if _id in e_:
                rm_validids.append(_id)
                valid_ids.remove(_id)
                break
    return f_ls, valid_ids
