# -*- coding:utf-8 -*-
"""存储了一些constant数据, 用于获取特征"""
import re
import os
import pandas as pd
import numpy as np


DICT = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLN': 'Q',
    'GLU': 'E',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V',
    'ASX': 'N',
    'GLX': 'Q',
    'UNK': 'G',
    'HSD': 'H',
}


def normlize_feature(feature_dict):
    values = list(feature_dict.values())
    min_, max_ = min(values), max(values)
    r_ = max_ - min_
    new_dict = {}
    for key_ in feature_dict.keys():
        new_dict[DICT[key_.upper()]] = (feature_dict[key_] - min_) / r_
    return new_dict


class PRO_property(object):
    def __init__(self, uniform=True):
        self.uniform = uniform
        self.DICT_kdhydro = {
            'ILE': 4.5,
            'VAL': 4.2,
            'LEU': 3.8,
            'PHE': 2.8,
            'CYS': 2.5,
            'MET': 1.9,
            'ALA': 1.8,
            'GLY': -0.4,
            'THR': -0.7,
            'SER': -0.8,
            'TRP': -0.9,
            'TYR': -1.3,
            'PRO': -1.6,
            'HIS': -3.2,
            'GLU': -3.5,
            'GLN': -3.5,
            'ASP': -3.5,
            'ASN': -3.5,
            'LYS': -3.9,
            'ARG': -4.5
        }
        # self.DICT_kdhydro = normlize_feature(self.DICT_kdhydro)
        self.DICT_iep = {
            'ALA': 6.00,
            'ARG': 11.15,
            'ASN': 5.41,
            'ASP': 2.77,
            'CYS': 5.02,
            'GLN': 5.65,
            'GLU': 3.22,
            'GLY': 5.97,
            'HIS': 7.47,
            'ILE': 5.94,
            'LEU': 5.98,
            'LYS': 9.59,
            'MET': 5.74,
            'PHE': 5.48,
            'PRO': 6.30,
            'SER': 5.68,
            'THR': 5.64,
            'TRP': 5.89,
            'TYR': 5.66,
            'VAL': 5.96}
        # self.DICT_iep = normlize_feature(self.DICT_iep)
        self.DICT_molw = {
            'ALA': 89.09,
            'ARG': 174.2,
            'ASN': 132.12,
            'ASP': 133.1,
            'CYS': 121.16,
            'GLN': 146.15,
            'GLU': 147.13,
            'GLY': 75.07,
            'HIS': 155.16,
            'ILE': 131.17,
            'LEU': 131.17,
            'LYS': 146.19,
            'MET': 149.21,
            'PHE': 165.19,
            'PRO': 115.13,
            'SER': 105.09,
            'THR': 119.12,
            'TRP': 204.23,
            'TYR': 181.19,
            'VAL': 117.15}
        # self.DICT_molw = normlize_feature(self.DICT_molw)
        self.DICT_vdW = {
            'ALA': 67,
            'ARG': 148,
            'ASN': 96,
            'ASP': 91,
            'CYS': 86,
            'GLN': 114,
            'GLU': 109,
            'GLY': 48,
            'HIS': 118,
            'ILE': 124,
            'LEU': 124,
            'LYS': 135,
            'MET': 124,
            'PHE': 135,
            'PRO': 90,
            'SER': 73,
            'THR': 93,
            'TRP': 163,
            'TYR': 141,
            'VAL': 105}
        # self.DICT_vdW = normlize_feature(self.DICT_vdW)
        self.DICT_sidetype = {
            'ALA': 0,
            'ARG': 3,
            'ASN': 2,
            'ASP': 4,
            'CYS': 2,
            'GLN': 2,
            'GLU': 4,
            'GLY': 5,
            'HIS': 3,
            'ILE': 0,
            'LEU': 0,
            'LYS': 3,
            'MET': 2,
            'PHE': 1,
            'PRO': 5,
            'SER': 2,
            'THR': 2,
            'TRP': 1,
            'TYR': 1,
            'VAL': 0}

    @property
    def dict_kdhydro(self):
        # 疏水性(值有正有负)
        DICT_kdhydro_o = {}
        for key_i in self.DICT_kdhydro.keys():
            DICT_kdhydro_o[DICT[key_i.upper()]] = self.DICT_kdhydro[key_i]
        DICT_kdhydro_o['-'] = 0
        return DICT_kdhydro_o

    @property
    def dict_iep(self):
        # 等电点(与极性相关)  等电点进行了标准化(是不是应该用归一化)
        if self.uniform:
            DICT_iep = normlize_feature(self.DICT_iep)
            DICT_iep['-'] = 0
        else:
            DICT_iep = {}
            mean_v = np.mean(np.array(list(self.DICT_iep.values())))
            std_v = np.std(np.array(list(self.DICT_iep.values())), ddof=1)
            for key_i in self.DICT_iep.keys():
                DICT_iep[DICT[key_i.upper()]] = (self.DICT_iep[key_i] - mean_v) / (std_v + 1e-8)  # 进行了标准化
            DICT_iep['-'] = (0 - mean_v) / (std_v + 1e-8)
        return DICT_iep

    @property
    def dict_mol_weight(self):
        # 分子量的大小
        if self.uniform:
            dict_molw = normlize_feature(self.DICT_molw)
            dict_molw['-'] = 0
        else:
            dict_molw = {}
            mean_v = np.mean(np.array(list(self.DICT_molw.values())))
            std_v = np.std(np.array(list(self.DICT_molw.values())), ddof=1)
            for key_i in self.DICT_molw.keys():
                dict_molw[DICT[key_i.upper()]] = (self.DICT_molw[key_i] - mean_v) / (std_v + 1e-8)  # 进行了标准化
            dict_molw['-'] = (0 - mean_v) / (std_v + 1e-8)
        return dict_molw

    @property
    def dict_vdW(self):
        # 范德华体积大小
        if self.uniform:
            _dict_vdW = normlize_feature(self.DICT_vdW)
            _dict_vdW['-'] = 0
        else:
            _dict_vdW = {}
            mean_v = np.mean(np.array(list(self.DICT_vdW.values())))
            std_v = np.std(np.array(list(self.DICT_vdW.values())), ddof=1)
            for key_i in self.DICT_vdW.keys():
                _dict_vdW[DICT[key_i.upper()]] = (self.DICT_vdW[key_i] - mean_v) / (std_v + 1e-8)  # 进行了标准化
            _dict_vdW['-'] = (0 - mean_v) / (std_v + 1e-8)
        return _dict_vdW

    @property
    def dict_sidetype(self):
        # 侧链是脂肪族 芳香族  中性侧链  正电荷侧链  负电荷侧链  特殊氨基酸侧链
        # one-hot向量 数目为5 加上gap则数目为6
        _dict_sidetype = {}
        for key_i in self.DICT_sidetype.keys():
            _dict_sidetype[DICT[key_i.upper()]] = int(self.DICT_sidetype[key_i])
        _dict_sidetype['-'] = int(6)  # 加上了特殊氨基酸类型
        return _dict_sidetype


def sortkey_fun(s):
    if s:
        try:
            c = re.findall('\d+', s)[0]
            # print(c)
        except:
            c = -1
        return int(c)


def str_list_sort(l_str):
    l_str.sort(key=sortkey_fun)  # key是作用于list中每个元素上的函数
    return l_str


def getnum_record_db(file_name):
    """返回记录文件中的条目"""
    rec = pd.read_csv(file_name, index_col=0)
    print(rec.columns)
    num_samps = len(rec)
    return num_samps


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功") 


def fixer_pdbf(pdbf):
    """对原子缺失的pdb文件进行原子补充"""
    fix_python_env = '/home/user/anaconda3/envs/pt14/bin/python'
    fix_python_f = '/home/user/protein/ppi_new/interface_pred/code_for_feature_AbTotal/fix_pdbfile.py'
    os.system(f'{fix_python_env} {fix_python_f} --pdb_file {pdbf}')
