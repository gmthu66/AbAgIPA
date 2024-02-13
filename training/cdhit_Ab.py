import numpy as np
import os.path as osp
import pandas as pd
from Bio import Phylo
from model.interface import DataDirPath
ddp = DataDirPath()
temp_dir = osp.join(osp.dirname(ddp.init_pdbdir), 'ClusterSAbDab/AbwithNbcluster_dir')


class CLS_dict(object):
    def __init__(self, data):
        self.cls_dict = data
        self.train_cls, self.test_cls = {}, {}
        self.seed = 123

    def getFold_data(self, fold_n):
        np.random.seed(self.seed)
        for i_ in range(fold_n):
            self.train_cls[i_], self.test_cls[i_] = {}, {}
        for key_ in self.cls_dict:
            N_ = len(self.cls_dict[key_])
            unit_ = int(N_ / fold_n)
            np.random.shuffle(self.cls_dict[key_])
            # self.train_cls[key_], self.test_cls[key_] = {}, {}
            for i_ in range(fold_n):
                if i_ != range(fold_n)[-1]:
                    test_loc = list(range(unit_ * i_, unit_ * (i_ + 1)))
                else:
                    test_loc = list(range(unit_ * i_, N_))
                self.test_cls[i_][key_] = self.cls_dict[key_][min(test_loc):max(test_loc) + 1]
                self.train_cls[i_][key_] = list(set(self.cls_dict[key_]) - set(self.test_cls[i_][key_]))


def get_tree_TaxonListCLS(tree_file=osp.join(osp.dirname(temp_dir), 'Cov-AbDab/build_tree_rename.nwk')):
    """获得进化树中的数据, 并进行子树的划分"""
    tree = Phylo.read(tree_file, 'newick')
    # 对tree中的分支clades进行分析
    print(tree)
    term_namels, subtree_dict = unfold_tree_ByLevel(tree, level_in=2)
    subtree_df = clsdict2df(subtree_dict)
    subtree_df = subtree_df.reset_index(drop=True)
    subtree_df_f = osp.join(osp.dirname(tree_file), 'build_tree_rename_CLS.csv')
    subtree_df.to_csv(subtree_df_f)
    return subtree_dict


def clsdict2df(dict_in):
    df_ = None
    for key_ in dict_in:
        pd_i = pd.DataFrame()
        pd_i['id_name'] = dict_in[key_]
        pd_i['id_cls'] = [key_] * len(dict_in[key_])
        if df_ is None:
            df_ = pd_i
        else:
            df_ = pd.concat([df_, pd_i], ignore_index=False)
    return df_


def unfold_tree_ByLevel(tree, level_in=1):
    subtree_dict, temp_f, term_namels, cls_ = {}, osp.join(temp_dir, 'tree_clade_temp.txt'), [], 0
    tree.get_terminals()
    for j_ in tree.root.clades:
        if level_in == 1:
            j_list = j_.get_terminals()
            j_list = get_idnames(j_list)
            term_namels = term_namels + j_list
            subtree_dict[cls_] = j_list
            cls_ += 1
            print(f'cluster level {level_in}\t', len(j_.clades), 'terminals:', len(j_list))
        elif level_in == 2:
            for j_2 in j_.clades:
                j_list2 = j_2.get_terminals()
                j_list2 = get_idnames(j_list2)
                term_namels = term_namels + j_list2
                subtree_dict[cls_] = j_list2
                cls_ += 1
                print(f'cluster level {level_in}\t', len(j_2.clades), 'terminals:', len(j_list2))
    print('unfold_tree_ByLevel END')
    return term_namels, subtree_dict


def get_idnames(obj_list, key_='name', add_line=True):
    if not add_line:
        return [obj_.__dict__[key_] for obj_ in obj_list]
    else:
        name_list = []
        for obj_ in obj_list:
            name_ = obj_.__dict__[key_][0:4] + '_' + '_'.join(list(obj_.__dict__[key_][4:]))
            name_list.append(name_)
        return name_list
