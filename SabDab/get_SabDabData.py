import sys
import os.path as osp
import pandas as pd
import argparse, os
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
project_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(project_dir)
from model.interface import DataDirPath
from database.parse_utils import series_values_byKeys
from database.preprocess.collect_utils import movepdb_from_sabdab
from database.inter_pdb import save_residuels2pdb
from database.inter_embed import physicEmbed_ab

from spec_At.data_code.Embedding import Embeder
from spec_At.HER2.inter_utils import get_pos_data
from spec_At.utils_df import remove_unnamed_columns, modify_posdataid
from spec_At.parsePDB import get_residues_BychNotAlterChainId, clean_pdbfile
from spec_At.parse_seqs import add_aaseqDict2fasta
from training_ClusterSAbDab.cluster_db import clustrByAtseq, family_with_subset
from training_ClusterSAbDab.cdhit_Ab import get_tree_TaxonListCLS
from utils.general import exists
from utils.folding import process_template, process_MutantTemplate
from spec_At.utils_df import partition_into_groups
import random
data_dir = osp.join(osp.dirname(__file__), 'SabDabdatabase')
ddp = DataDirPath()
random.seed(22)
np.random.seed(22)


def getPosPairFromInitTxt():
    pos_txt = osp.join(data_dir, 'positive_samples.txt')
    comp_df = pd.read_csv(pos_txt, sep='\t', header=None)
    # comp_df.columns = ['sudo_ab', 'sudo_at']  # 这里是固定坐标, 对右边采样使得抗体表位variable一些
    comp_df.columns = ['sudo_at', 'sudo_ab']  # debug, 这里是固定抗原, 让抗体variable一些
    comp_df.to_csv(osp.join(data_dir, 'positive_samples.csv'))
    return comp_df


def str2ab_at(ss, dtype='Ab'):
    if dtype == 'Ab':
        return ss.split('_')[0], ss.split('_')[1], ss.split('_')[2], ss.split('_')[3]
    else:
        return ss.split('_')[0], ss.split('_')[1], ss.split('_')[2]


def get_summ_index(pdb_obj, ab_Hch, ab_Lch, Tch, df_summ):
    df_ = df_summ[df_summ['pdb'].str.contains(pdb_obj)]
    df_ = df_[df_['Hchain'].str.contains(ab_Hch)]
    df_ = df_[df_['Lchain'].str.contains(ab_Lch)]
    df_ = df_[df_['antigen_chain'].str.contains(Tch)]
    if df_.shape[0] > 0:
        i_ = df_.iloc[0]._name
        return int(i_)
    else:
        return -1


def get_summ_Nbindex(pdb_obj, ab_Hch, Tch, df_summ):
    df_ = df_summ[df_summ['pdb'].str.contains(pdb_obj)]
    df_ = df_.dropna(subset=['Hchain'])
    df_ = df_[df_['Hchain'].str.contains(ab_Hch)]
    df_ = df_[df_['antigen_chain'].str.contains(Tch)]
    if df_.shape[0] > 0:
        i_ = df_.iloc[0]._name
        return int(i_)
    else:
        return -1


def add_summary_index(df_in, df_summ, df_in_file, dtype='Ab'):
    df_out_file = "".join(df_in_file.split('.')[0] + 'WithSummindex' + df_in_file.split('.')[1])
    df_in = df_in.dropna(subset=['sudo_ab', 'sudo_at'])  # 删除为空的数据
    if osp.exists(df_out_file):
        df_out = pd.read_csv(df_out_file, index_col=False)
        if 'summary_index' in df_out.columns:
            return df_in
    df_in['summary_index'] = [0] * df_in.shape[0]
    print('add_summary_index...')
    for i_ in tqdm(df_in.index):
        if dtype == 'Ab':
            ab_pdb_obj, ab_Hch, ab_Lch, ab_Tch = str2ab_at(df_in.loc[i_]['sudo_ab'], dtype)
            at_pdb_obj, at_Hch, at_Lch, at_Tch = str2ab_at(df_in.loc[i_]['sudo_at'], dtype)
            ab_summ_index = get_summ_index(ab_pdb_obj, ab_Hch, ab_Lch, ab_Tch, df_summ)
            at_summ_index = get_summ_index(at_pdb_obj, at_Hch, at_Lch, at_Tch, df_summ)
        else:
            ab_pdb_obj, ab_Hch, ab_Tch = str2ab_at(df_in.loc[i_]['sudo_ab'], dtype)
            at_pdb_obj, at_Hch, at_Tch = str2ab_at(df_in.loc[i_]['sudo_at'], dtype)
            ab_summ_index = get_summ_Nbindex(ab_pdb_obj, ab_Hch, ab_Tch, df_summ)
            at_summ_index = get_summ_Nbindex(at_pdb_obj, at_Hch, at_Tch, df_summ)
        at_pdb_obj = df_in.loc[i_]['sudo_at'].split('_')[0]
        df_in.at[i_, 'summary_index'] = ab_summ_index
        df_in.at[i_, 'ab_summary_index'] = ab_summ_index
        df_in.at[i_, 'at_summary_index'] = at_summ_index
    df_in.to_csv(df_out_file, index=False)
    return df_in


def filter_nan_ls(ls_in):
    new_ls = []
    for i_ in ls_in:
        if isinstance(i_, str):
            new_ls.append(i_)
    return new_ls


def get_at_ab_infos(df_in):
    dict_info = {}
    ab_ls = df_in['sudo_ab'].to_list()
    ab_ls = filter_nan_ls(ab_ls)
    at_ls = df_in['sudo_at'].to_list()
    at_ls = filter_nan_ls(at_ls)
    ab_ls.extend(at_ls)
    total_ls = sorted(list(set(ab_ls)))
    info_ls = []
    for item_i in total_ls:
        assert len(item_i.split('_')) == 4
        pdb_id = item_i.split('_')[0]
        ab_at = '_'.join(item_i.split('_')[1:])
        if pdb_id not in dict_info.keys():
            dict_info[pdb_id] = [ab_at]
            info_ls.append(item_i)
        else:
            if ab_at not in dict_info[pdb_id]:
                dict_info[pdb_id].append(ab_at)
                info_ls.append(item_i)
    list(set(df_in.loc[:, 'sudo_ab'].to_list() + df_in.loc[:, 'sudo_at'].to_list()))
    return dict_info, info_ls


def get_rec_property(pdbf_name_dict, pdbf_name_with=None):
    summary_info = pd.read_csv(ddp.sabdab_summtsv, sep='\t', index_col=False).dropna(subset=['antigen_chain'])
    AbAtId_ls = []
    rec_property, rec_property_ls = {}, []

    for i_, pdbf in enumerate(pdbf_name_dict.keys()):

        rec_property = {'AtId_ls': list(pdbf_name_dict.keys())[i_], 'Pos': True, 'InitIsTotal': True}
        item_name = pdbf_name_dict[pdbf].split('.')[0]
        pdb_obj = item_name[:4]
        df_info = summary_info[summary_info['pdb'].str.contains(pdb_obj)]
        if pdbf_name_with == 'At':
            df_info = df_info[df_info['antigen_chain'].str.contains(item_name[-1])]  # 利用at找到数据
        if df_info.shape[0] > 1: df_info = df_info.iloc[0].to_frame().transpose()
        assert df_info.shape[0] == 1

        At_chs = series_values_byKeys(df_info, 'antigen_chain')
        HL_chs = series_values_byKeys(df_info, ['Hchain', 'Lchain'])
        AbAtId = '{}_{}_{}'.format(pdb_obj, '_'.join(HL_chs), ''.join(At_chs))
        rec_property['AbAtId'] = AbAtId
        AbAtId_ls.append(AbAtId)
        rec_property_ls.append(rec_property)    

    return rec_property_ls


def process_SingleAt(embeder, rec_property=None, database_args=None, atseq=False):
    AbAtId, InitIsTotal, Pos = rec_property['AbAtId'], rec_property['InitIsTotal'], rec_property['Pos']
    AbAtId_ls, pdb_dir = parseAbAtId(AbAtId)
    At_datals, Ab_datals = get_pos_data(AbAtId_ls, pdb_dir, embeder=embeder, atseq=atseq)
    return At_datals, Ab_datals


def save_merge_pdb(AbAtId_string, merge_pdb_path):
    pdb_dir = osp.join(osp.dirname(osp.dirname(merge_pdb_path)), 'pdbs')
    temp_dir, db_dir = osp.join(pdb_dir, 'temp'), osp.dirname(merge_pdb_path)
    os.makedirs(temp_dir, exist_ok=True)
    obj_pdb_path = osp.join(pdb_dir, AbAtId_string[:4] + '.pdb')
    movepdb_from_sabdab(AbAtId_string[:4], obj_pdb_path)
    clean_pdbfile(obj_pdb_path)
    HL_chls, at_chls = AbAtId_string.split('_')[1:3], [AbAtId_string.split('_')[-1]]
    Ab_resls = get_residues_BychNotAlterChainId(obj_pdb_path, HL_chls)
    At_resls = get_residues_BychNotAlterChainId(obj_pdb_path, at_chls)
    save_residuels2pdb(Ab_resls + At_resls, merge_pdb_path)
    return merge_pdb_path


def get_AtAbDataDict(info_dict, embeder, reci=None, atseq=True):
    At_datals, Ab_datals, AbAtPair_dict = [], [], {}
    pdb_dir = osp.join(osp.dirname(data_dir), 'pdbs')
    print('get AtAbDataDict...')
    for AbAt_ids in tqdm(info_dict.keys()):
        # if '1mlc' not in AbAt_ids:  continue
        for abatch in info_dict[AbAt_ids]:
            abat_id_datadict = {}
            AbAtId_string = f"{AbAt_ids}_{abatch}"
            # if '4od2' not in AbAtId_string:  continue  # debug
            if AbAtId_string in ['3ixx_G_H_A', '3j3p_H_L_1', '3j8d_D_D_F', '4c2i_H_L_C', '4cau_E_e_A', '4uif_K_L_C', '5d71_H_L_A', '5kov_Q_q_N', '5kov_W_w_S', '5y0a_D_E_A', '6idi_M_N_F', '6kn9_E_E_B', '7bue_J_K_C']:  continue
            # if '6urh' not in AbAtId_string:  continue
            merge_pdb_path = osp.join(data_dir, AbAtId_string + '.pdb')
            if not osp.exists(merge_pdb_path):
                merge_pdb_path = save_merge_pdb(AbAtId_string, merge_pdb_path)
            At_datals, Ab_datals = get_pos_data([AbAtId_string], pdb_dir, embeder=embeder, atseq=True, ab_pre=True, keep_initId=True)
            At_data, Ab_data = At_datals[0], Ab_datals[0]
            At_data['init_AbAtId'], Ab_data['init_AbAtId'] = AbAtId_string, AbAtId_string
            abat_id_datadict['merge_pdb_path'] = hasattr(At_data, AbAtId_string)
            abat_id_datadict['At_data'] = At_data
            abat_id_datadict['Ab_data'] = Ab_data
            AbAtPair_dict[AbAtId_string] = abat_id_datadict
    return AbAtPair_dict


def abseq_FromDict(Pair_dict):
    abseq_dict = Pair_dict['Ab_data']['seq_dict']
    Hseq = abseq_dict['H'] if 'H' in abseq_dict.keys() else None
    Lseq = abseq_dict['L'] if 'L' in abseq_dict.keys() else None
    total_abseq = ''.join([seq for seq in [Hseq, Lseq] if seq is not None])
    return Hseq, Lseq, total_abseq


def atseq_FromDict(Pair_dict):
    return Pair_dict['At_data']['atseq']


def abrecord2standard(df_in, AbAtPair_dict, rec_path):
    pos_std = deepcopy(df_in)
    pos_dataid = 0
    print('abrecord standardring...')
    if osp.exists(rec_path):
        pos_std = pd.read_csv(rec_path, index_col=False)
    else:
        for i_ in tqdm(pos_std.index):
            Babat_id, Tabat_id = df_in.loc[i_]['sudo_ab'], df_in.loc[i_]['sudo_at']
            # if '5ken' not in Babat_id:  continue
            if Babat_id in AbAtPair_dict.keys() and Tabat_id in AbAtPair_dict.keys():
                at_pdb_obj, at_Hch, at_Lch, at_Tch = str2ab_at(Tabat_id, dtype='Ab')  # 原本正样本增强弄成了抗原表位的正样本增强, 然而抗原正样本间相似性是非常大的没必要增强
                ab_pdb_obj, ab_Hch, ab_Lch, ab_Tch = str2ab_at(Babat_id, dtype='Ab')
                at_chname = "{}_{}".format(at_pdb_obj, at_Tch)
                ab_chname = "{}_{}".format(ab_pdb_obj, ''.join([ab_Hch, ab_Lch]))
                pos_std.at[i_, 'At_name'] = at_chname
                pos_std.at[i_, 'Ab_name'] = ab_chname

                Hseq, Lseq, ab_seq = abseq_FromDict(AbAtPair_dict[Babat_id])
                pos_std.at[i_, 'Hseq'], pos_std.at[i_, 'Lseq'], pos_std.at[i_, 'Abseq'] = Hseq, Lseq, ab_seq
                pos_std.at[i_, 'len_0'] = len(Hseq) if isinstance(Hseq, str) else 0
                pos_std.at[i_, 'len_1'] = len(Lseq) if isinstance(Lseq, str) else 0
                pos_std.at[i_, 'cdr_fr_info'] = str(AbAtPair_dict[Babat_id]['Ab_data']['cdr_fr_dict'])

                pos_std.at[i_, 'init_ab_mergeobj'] = Babat_id
                pos_std.at[i_, 'ab_mergeobj'] = AbAtPair_dict[Babat_id]['At_data']['init_AbAtId']
                pos_std.at[i_, 'ab_preDir'] = osp.dirname(AbAtPair_dict[Babat_id]['At_data'][Babat_id])

                pos_std.at[i_, 'init_at_mergeobj'] = Tabat_id
                pos_std.at[i_, 'Atseq'] = atseq_FromDict(AbAtPair_dict[Tabat_id])  # 这里的Atseq给错了?
                pos_std.at[i_, 'at_mergeobj'] = AbAtPair_dict[Tabat_id]['At_data']['init_AbAtId']
                pos_std.at[i_, 'at_preDir'] = osp.dirname(AbAtPair_dict[Tabat_id]['At_data'][Tabat_id])
                # pos_std.at[i_, 'at_preDir'] = osp.dirname(AbAtPair_dict[Tabat_id]['At_data'][Tabat_id])
                pos_std.at[i_, 'pos_dataid'] = pos_dataid
                pos_dataid += 1
    pos_std.to_csv(rec_path, index=False)
    pos_std = pos_std.dropna(subset=['Ab_name'])
    return pos_std


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody and Nanobody Structure predict by IPA')
    parser.add_argument('--id_fold', default=0, type=int)
    parser.add_argument('--embed_type', default='Physic', type=str)
    args = parser.parse_args()
    return args


def check_duplic_dict(info_dict):
    def check_intersection(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if set(lst[i]) & set(lst[j]):
                    return True
        return False
    ll = []
    for k_ in info_dict.keys():
        ll.append(list(info_dict[k_].keys()))
    return check_intersection(ll)


def flatten_dict_values(data_dict):
    # 使用列表生成式将字典中的子列表连接在一起
    dict_flattened_list = {k_ : [] for k_ in data_dict.keys()}
    for k_, dicti in data_dict.items():
        flattened_listi = [value for values in dicti.values() for value in values]
        dict_flattened_list[k_].extend(flattened_listi)
    return dict_flattened_list


def abatid_2_clsid(cls_dict_subset):
    abatid_affine_clsid = {}
    for k_ in cls_dict_subset.keys():
        for abatid in cls_dict_subset[k_]:
            abatid_affine_clsid[abatid] = k_
    return abatid_affine_clsid


def k_fold_cross_validation(data_dict, k=5):
    """获取对所有subset平均分配的k_fold字典"""
    valid_folds, train_folds = {i: [] for i in range(k)}, {i: [] for i in range(k)}

    for key, values in data_dict.items():
        # 将当前关键字的内容进行随机打乱
        shuffled_values = random.sample(values, len(values))
        # 将打乱后的内容划分为k份
        fold_size = len(shuffled_values) // k
        for i in range(k):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < k - 1 else len(shuffled_values)
            fold_data = shuffled_values[start_idx:end_idx]
            valid_folds[i].extend(fold_data)
    for key, values in data_dict.items():
        shuffled_values = random.sample(values, len(values))
        for i in range(k):
            train_folds[i].extend(list(set(shuffled_values) - set(valid_folds[i])))
    return train_folds, valid_folds


def k_fold_noRedunSubsetTrianVal(data_dict, all_abatids, k=5):
    # 输入的应该是家族信息
    print('construct kFold trainval by k_fold_noRedunSubsetTrianVal')
    subset_family_dict = {key: {} for key in range(len(data_dict.keys()))}
    train_folds = {key: [] for key in range(k)}
    val_folds = {key: [] for key in range(k)}
    for i_ in data_dict.keys():  # 一共有6个家族其实
        # 分别对每个家族下的subset id进行 5组分组
        subset_family_dicti = {ii: [] for ii in range(k)}
        familyi = data_dict[i_]
        numls_of_elem = [len(v_) for v_ in familyi.values()]
        subset_indexls = [k_ for k_ in familyi.keys()]

        # 对set进行分割, 使得分组后各组内元素数目尽量一致, 且不同组之间subset不重复
        familyi_set_partion = partition_into_groups(numls_of_elem, k)
        familyi_set_partion = {index: value for index, value in enumerate(familyi_set_partion)}
        for kk in familyi_set_partion.keys():
            subset_family_dicti[kk] = [subset_indexls[j_] for j_ in familyi_set_partion[kk]]

        shuffle_ls, id_ls = list(subset_family_dicti.keys()), [[] for _ in range(k)]
        np.random.shuffle(shuffle_ls)
        for _foldi in shuffle_ls:
            for _subid in subset_family_dicti[_foldi]:
                id_ls[_foldi].extend(data_dict[i_][_subid])
        for _foldi in range(k):
            subset_family_dicti[_foldi] = id_ls[_foldi]
        subset_family_dict[i_] = subset_family_dicti
    for _foldi in range(k):
        for _, subset_family_dicti in subset_family_dict.items():
            val_folds[_foldi].extend(subset_family_dicti[_foldi])
        train_folds[_foldi] = sorted(list(set(all_abatids) - set(val_folds[_foldi])))
    return train_folds, val_folds


def get_data(embeder, non_redun='Atseq', atcluster_trunc=0.90, negRatioPos=1, dataSetName='SabDab', poskey='sudo_at', kfold=5):
    df_summary = pd.read_csv(ddp.sabdab_summtsv, sep='\t', index_col=False)
    abrecord_df = getPosPairFromInitTxt()
    abrecord_df['merged_index'] = range(abrecord_df.shape[0])
    abrecord_df = add_summary_index(abrecord_df, df_summary, osp.join(data_dir, 'positive_samples.csv'))
    info_dict, info_ls = get_at_ab_infos(abrecord_df)
    pdb_idls = list(set([p_id.split('_')[0] for p_id in info_ls]))
    # 首先获得Hseq Lseq 以及 cdr信息
    AbAtPair_dict = get_AtAbDataDict(info_dict, embeder, atseq=True)

    save_dir = osp.join(osp.dirname(data_dir), 'kfold_data')
    trainval_df_dict = try_read_trainvaldf(save_dir, negRatioPos, dataSetName, kfold)
    # trainval_df_dict = None  # debug

    if trainval_df_dict is None:
        pos_stdDf_path = osp.join(data_dir, 'positive_StdRecord.csv')
        pos_stdDf = abrecord2standard(abrecord_df, AbAtPair_dict, pos_stdDf_path)
        pos_stdDf = modify_posdataid(pos_stdDf, pos_dataid_byAt=True, at_name='sudo_at')
        # df_no_duplicates = pos_stdDf.drop_duplicates(subset=['Ab_name'], keep='first')
        subset_contain_abatid, pdbid2datanum = clustrByAtseq(pos_stdDf, data_dir=data_dir, merge_trunc=atcluster_trunc, non_redun=non_redun, ab_type='Ab', abatid_flg=True)
        family_contain_abatid = get_tree_TaxonListCLS()  # '5dfv_CD_B', '6ejg_Dd_A'
        # cls_dict_family = CLS_dict(cls_dict)
        abatid_affine_fammid = abatid_2_clsid(family_contain_abatid)
        family_subset_info = family_with_subset(subset_contain_abatid, family_contain_abatid)  # 得到一个含有6个家族与subset对应关系的字典
        # assert not check_duplic_dict(family_subset_info)  # 检查family 相互之间是否有重复的subset
        print(f'\tclsutering at ids by merge_trunc: {atcluster_trunc} End\t')
        # train_folds, valid_folds = k_fold_cross_validation(family_contain_abatid)
        allpdb_id = [element for sublist in subset_contain_abatid.values() for element in sublist]
        train_folds, valid_folds = k_fold_noRedunSubsetTrianVal(family_subset_info, allpdb_id, kfold)
        abatid_affine_subsetid = abatid_2_clsid(subset_contain_abatid)
        trainval_df_dict = construct_trainval_df(pos_stdDf, train_folds, valid_folds, subset_contain_abatid, abatid_affine_subsetid, family_subset_info, family_contain_abatid, abatid_affine_fammid, negRatioPos, poskey=poskey)
        save_kfold_df(save_dir, trainval_df_dict, negRatioPos, dataSetName=dataSetName)
    return trainval_df_dict, AbAtPair_dict


def kfold_df_init():
    [('/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/tranval_1in5Fold_1posaug_1_negsample1_onlyAb_AbAtSplitTrue.csv', '/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/test_1in5Fold_1posaug_0_negsample1_onlyAb_AbAtSplitTrue.csv'), ('/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/tranval_2in5Fold_1posaug_1_negsample1_onlyAb_AbAtSplitTrue.csv', '/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/test_2in5Fold_1posaug_0_negsample1_onlyAb_AbAtSplitTrue.csv'), ('/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/tranval_3in5Fold_1posaug_1_negsample1_onlyAb_AbAtSplitTrue.csv', '/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/test_3in5Fold_1posaug_0_negsample1_onlyAb_AbAtSplitTrue.csv'), ('/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/tranval_4in5Fold_1posaug_1_negsample1_onlyAb_AbAtSplitTrue.csv', '/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/test_4in5Fold_1posaug_0_negsample1_onlyAb_AbAtSplitTrue.csv'), ('/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/tranval_5in5Fold_1posaug_1_negsample1_onlyAb_AbAtSplitTrue.csv', '/home/user/protein/ppi_new/structAware_AbAtPred/database/Ab/trainval/processed/test_5in5Fold_1posaug_0_negsample1_onlyAb_AbAtSplitTrue.csv')]


def save_kfold_df(save_dir, trainval_df_dict, negRatioPos, dataSetName='SabDab'):
    os.makedirs(save_dir, exist_ok=True)
    for k_ in trainval_df_dict:
        for n_, datadf in trainval_df_dict[k_].items():
            datadf.to_csv(osp.join(save_dir, f'{dataSetName}_{n_}_{k_}_negRatioPos{negRatioPos}.csv'), index=False)


def try_read_trainvaldf(save_dir, negRatioPos, dataSetName='SabDab', kfold=5):
    trainval_df_dict = {k:{} for k in range(kfold)}
    try:
        for k_ in range(kfold):
            for n_ in ['train', 'val']:
                fpath = osp.join(save_dir, f'{dataSetName}_{n_}_{k_}_negRatioPos{negRatioPos}.csv')
                trainval_df_dict[k_][n_] = remove_unnamed_columns(pd.read_csv(fpath, index_col=False))
        return trainval_df_dict
    except:
        return None


def construct_trainval_df(pos_df, train_folds, valid_folds, subset_contain_abatid, abatid_affine_subsetid, family_subset_info, family_contain_abatid, abatid_affine_fammid, negRatioPos=1, poskey='sudo_at'):
    kfold = len(train_folds.keys())
    trainval_df_dict = {k:{} for k in range(kfold)}
    for k_i in range(kfold):
        train_fold, valid_fold = train_folds[k_i], valid_folds[k_i]
        train_pos_df = split_posdf_trainval(pos_df, train_fold, poskey)
        valid_pos_df = split_posdf_trainval(pos_df, valid_fold, poskey)  # 负样本取同一个family但是不同subset的数据来构成?
        print(f'get negdf by Ab_abatid samefamily but not samesubset... kfold:{k_i} / {kfold}')
        train_neg_df = negdf_by_samefamily_nosamesubset(pos_df, train_pos_df, subset_contain_abatid, abatid_affine_subsetid, family_subset_info, family_contain_abatid, abatid_affine_fammid, negRatioPos, poskey)
        valid_neg_df = negdf_by_samefamily_nosamesubset(pos_df, valid_pos_df, subset_contain_abatid, abatid_affine_subsetid, family_subset_info, family_contain_abatid, abatid_affine_fammid, negRatioPos, poskey)

        train_pos_df['AgClass'] = [1] * train_pos_df.shape[0]
        valid_pos_df['AgClass'] = [1] * valid_pos_df.shape[0]
        train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True)
        val_df = pd.concat([valid_pos_df, valid_neg_df], ignore_index=True)
        trainval_df_dict[k_i] = {'train': train_df, 'val': val_df}
        print(f'get negdf by Ab_abatid samefamily but not samesubset End, kfold:{k_i} / {kfold}\n')
    return trainval_df_dict


def negdf_by_samefamily_nosamesubset(total_posdf, subpos_df, subset_contain_abatid, abatid_affine_subsetid, family_subset_info, family_contain_abatid, abatid_affine_fammid, negRatioPos=1, poskey='sudo_ab'):
    negdf = pd.DataFrame(columns=subpos_df.columns)
    # neg_keepcols = ['sudo_at', 'At_name', 'Atseq', 'init_at_mergeobj', 'init_at_mergeobj', 'at_mergeobj', 'at_preDir']  # 这里错了, 现在应该反过来 固定了At, 而且原本sudo_at也设置错了
    # total_At_abatids = total_posdf['sudo_at'].to_list()
    neg_changeAb_keepcols = ['sudo_ab', 'merged_index', 'Ab_name', 'Hseq', 'Lseq', 'Abseq', 'len_0', 'len_1', 'cdr_fr_info', 'init_ab_mergeobj', 'ab_mergeobj', 'ab_preDir']
    # neg_keepcols = neg_changeAb_keepcols
    total_At_abatids = total_posdf['sudo_at'].to_list()
    # pos_abatids = subpos_df['sudo_ab'].to_list()
    pos_abatids = sorted(list(set(subpos_df['sudo_at'].to_list())))
    excused_abatids = []
    for pos_abatid in tqdm(pos_abatids):
        if pos_abatid in excused_abatids:  continue

        if pos_abatid in abatid_affine_fammid:
            same_famm_abatids = sorted(list(set(family_contain_abatid[abatid_affine_fammid[pos_abatid]]) - set(subset_contain_abatid[abatid_affine_subsetid[pos_abatid]])))  # 获取同家族但是不同subset的数据
            candi_neg_abatids = sorted(list(set(total_At_abatids).intersection(same_famm_abatids)))  # 选择同家族但是不同subset的情况作为负样本
        else:
            print(f'{pos_abatid} not in family cluster')
            candi_neg_abatids = sorted(list(set(total_At_abatids) - set(subset_contain_abatid[abatid_affine_subsetid[pos_abatid]])))
        pos_dfi = subpos_df[subpos_df[poskey] == pos_abatid]  # pos中是对抗原进行固定的

        neg_abatidnum = pos_dfi.shape[0] * negRatioPos
        neg_abatids = random.sample(candi_neg_abatids, neg_abatidnum)
        neg_dfi = pd.concat([pos_dfi] * negRatioPos, ignore_index=True).reset_index(drop=True)
        for j_, neg_abatid in enumerate(neg_abatids):
            neg_dfij = total_posdf[total_posdf['sudo_at'] == neg_abatid].iloc[0]  # 一开始的初始化值是pos Ab端的数据
            # neg_dfij = total_posdf[total_posdf['sudo_ab'] == neg_abatid].iloc[0]  # 一开始的初始化值是pos Ab端的数据
            # for coli in neg_keepcols:
            for coli in neg_changeAb_keepcols:
                neg_dfi.at[j_, coli] = neg_dfij[coli]
        negdf = pd.concat([negdf, neg_dfi], ignore_index=True)
        excused_abatids.append(pos_abatid)
    negdf['merged_index'] = [-1] * negdf.shape[0]
    negdf['AgClass'] = [0] * negdf.shape[0]
    return negdf


def split_posdf_trainval(pos_df, folds_abatids, poskey):
    # fold_posdf = pos_df[pos_df['ab_mergeobj'].isin(folds_abatids)]
    fold_posdf = pos_df[pos_df[poskey].isin(folds_abatids)]
    return fold_posdf


def process_Abdata(Ab_data, db_args=None, template_pdb=None, ignore_cdrs=None, ignore_chain=None, tmp_flg=False, mutate_range=None, map_cpu=False):

    label_coords = Ab_data['label_coords']
    cdr_fr_dict = Ab_data['cdr_fr_dict']
    if 'Ab_phisicEmbed' not in Ab_data.keys() or 'Ab_embeddings' not in Ab_data.keys():
        aaseq_dict = Ab_data['seq_dict']
        ab_rei_distAt, ab_fasta_file = Ab_data['ab_rei_distAt'], Ab_data['ab_fasta_file']

        Ab_embeddings, Ab_attentions = db_args['Ab_embed_model'].embed(aaseq_dict.values(), return_attention=True,)
        if not map_cpu:
            Ab_embeddings = [e[1:-1].unsqueeze(0) for e in Ab_embeddings]
            Ab_attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0) for a in Ab_attentions]
        else:
            Ab_embeddings = [e[1:-1].unsqueeze(0).to(torch.device('cpu')) for e in Ab_embeddings]
            Ab_attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0).to(torch.device('cpu')) for a in Ab_attentions]
        if 'Ab_phisicEmbed' in Ab_data:
            Ab_phisicEmbed = Ab_data['Ab_phisicEmbed']
        else:
            if not exists(db_args['At_embed_model']):  # 若At没有embed_model
                Ab_phisicEmbed, Ab_phisicAttention = physicEmbed_ab(aaseq_dict)
        # ab_fasta_file = osp.join(osp.dirname(data_dir), 'pdbs', 'temp', 'db_collect.fasta')
        # aaseqDict2fasta(aaseq_dict, ab_fasta_file)
        add_aaseqDict2fasta(aaseq_dict, ab_fasta_file)
        if not tmp_flg:
            Ab_temp_coords, Ab_temp_mask = process_template(
                template_pdb,
                ab_fasta_file,
                ignore_cdrs=ignore_cdrs,
                ignore_chain=ignore_chain,
            )  # 获取的其实是无模板信息
        else:  # 看作template
            Ab_temp_coords, Ab_temp_mask = process_MutantTemplate(label_coords, cdr_fr_dict, mutate_range, ignore_cdrs=ignore_cdrs, ignore_chain=ignore_chain,)  # mut_range范围内的氨基酸不会作为模板信息
            # Ab_temp_coords是可以在__get_item时再做的
        Ab_data = {'label_coords': label_coords, 'cdr_fr_dict': cdr_fr_dict, 'ab_rei_distAt': None,
                    'Ab_embeddings': Ab_embeddings, 'Ab_attentions': Ab_attentions, 'Ab_temp_coords': Ab_temp_coords,
                    'Ab_temp_mask': Ab_temp_mask, 'ab_fasta_file': ab_fasta_file, 'mutate_range': mutate_range,}
        if not exists(db_args['At_embed_model']):
            init_dict = {'Ab_batch_mask': None, 'Ab_align_mask': None, 'Ab_phisicEmbed': Ab_phisicEmbed, 'Ab_phisicAttention': Ab_phisicAttention}
        else:
            init_dict = {'Ab_batch_mask': None, 'Ab_align_mask':None}
        Ab_data.update(init_dict)
    return Ab_data


def parseAbAtId(AbAtId, data_pardir):
    if not isinstance(AbAtId, list):
        AbAtId = [AbAtId]
    pdb_dir = osp.join(data_pardir, 'pdbs')
    if not osp.exists(pdb_dir):
        os.makedirs(pdb_dir, exist_ok=True)
    for id_i in AbAtId:
        if len(id_i.split('_')[0]) == 4:
            pdb_id = id_i.split('_')[0].lower()
            f_path = osp.join(pdb_dir, pdb_id + '.pdb')
            if not osp.exists(f_path):
                movepdb_from_sabdab(pdb_id, f_path)
    return AbAtId, pdb_dir


if __name__ == "__main__":
    config = parse_args()
    embeder = Embeder(config)
    get_data(embeder)
    print()
