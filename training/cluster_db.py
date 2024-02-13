import os
import os.path as osp
import sys
import pandas as pd
from os.path import basename
from tqdm import tqdm
project_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(project_dir)

from model.interface import DataDirPath, read_dict
from database.inter_data import insight_inter_noninterDf, get_fpaths_databaseDf
from database.parse_utils import series_values_byKeys, inter_noninter_split
from training.dataset import rightpdb_size
from utils.fasta import get_chainseq_FromPDB
from database.preprocess.nonRedun import NonRedun, get_nonRedun_ByAbSeq, dataframe2fasta
from training.cdhit_Ab import get_tree_TaxonListCLS, CLS_dict
ddp = DataDirPath()
temp_dir = osp.join(osp.dirname(ddp.init_pdbdir), 'ClusterSAbDab/AbwithNbcluster_dir')


def add_aaseq2Df(df_in, init_path, atseq_trunc=40):
    o_path = osp.join(osp.dirname(init_path), basename(init_path).split('.')[0] + '_WithAAseq.' + basename(init_path).split('.')[1])
    if osp.exists(o_path):
        df_in = pd.read_csv(o_path, index_col=False)
        df_in = remove_unnamed_columns(df_in)
    else:
        for i_ in tqdm(df_in.index.to_list()):
            pdb_path = df_in.loc[i_]['merge_pdbpath']
            # 获得HL对应的aaseq并更新
            HL_chs = series_values_byKeys(df_in.loc[i_], ['Hchain', 'Lchain'])
            assert len(set(HL_chs) - set(['H', 'L'])) < 1
            for ch_ in HL_chs:
                df_in.at[i_, f'{ch_}seq'] = get_chainseq_FromPDB(pdb_path, ch_)
            # 获得AtChain对应的aaseq并更新
            if isinstance(df_in.loc[i_, 'antigen_chain'], str):
                atch_ls = [ch_.strip(' ') for ch_ in df_in.loc[i_, 'antigen_chain'].split('|')]
                atch_seqls, num_len = [], 0
                for atch in sorted(atch_ls):
                    atseq = get_chainseq_FromPDB(pdb_path, atch)
                    num_len += len(atseq)
                    atch_seqls.append(atseq)
                atseq = ','.join(atch_seqls) if num_len >= atseq_trunc else pd.NA
                df_in.at[i_, 'Atseq'] = atseq
        df_in.to_csv(o_path, index=False)
    return df_in


def clsutering_sabdab(sabdab_dirs=[ddp.sabdab_only_Ab, ddp.test_dirs], valid_ids=[]):
    fpaths = []
    inter_pdbobj_ls, noninter_pdbobj_ls, inter_noninter_dict, sabdab_interdf = insight_inter_noninterDf(ddp.inter_noninter_dict_info['inter'], ddp.inter_noninter_dict_info['noninter'], ddp)
    for step, _dir in enumerate(sabdab_dirs):  # 依据only_struct, 获取oas_paired、oas_unpaired、sabdab以及test数据的index字典, 便于计算每个数据集的大小, 以便于进行数据采样
        sub_fs = [f_ for f_ in os.listdir(_dir) if (('.pdb' in f_) or ('.fasta' in f_)) and not f_.startswith('.')]
        if osp.exists(ddp.error_dict_path) and osp.exists(ddp.right_dict_path):
            sub_fs, valid_ids = _filter_errorfile(_dir, sub_fs, valid_ids)
        sub_fs = [(f_.split('.')[0] + '.pdb') for f_ in sub_fs]
        sub_fs = [osp.join(_dir, f_) for f_ in sub_fs if (f_.endswith('.pdb') and rightpdb_size(osp.join(_dir, f_)))]
        fpaths.extend(sub_fs)
    total_inter_noninter_fpath = osp.join(project_dir, 'database', 'ClusterSAbDabCsv', 'sabdab_inter_noninter.tsv')
    temp_dir = osp.join(osp.join(project_dir, 'database', 'ClusterSAbDabCsv', 'temp'))
    os.makedirs(temp_dir, exist_ok=True)
    sabdab_df = get_fpaths_databaseDf(fpaths, ddp, valid_ids, opath=total_inter_noninter_fpath,)  # 需要基于inter pdb_objls以及only_struct objls产生一些noninter的负作用数据
    sabdab_df = add_aaseq2Df(sabdab_df, total_inter_noninter_fpath)

    # 利用单抗/纳米抗体序列一致性小于0.98来筛选数据
    nonRedunAb_df, nonRedunNb_df = get_nonRedun_ByAbSeq(sabdab_df, non_redun_trunc=0.98, temp_dir=osp.dirname(total_inter_noninter_fpath))
    sabdab_df = pd.concat([nonRedunAb_df, nonRedunNb_df], ignore_index=True)

    # 一共1636条具有相互作用标签且抗体序列一致性低于98%的数据
    inter_sabdab_df, noninter_sabdab_df = inter_noninter_split(sabdab_df)
    inter_sabdab_df['redun_tempIndex'] = range(inter_sabdab_df.shape[0])
    temp_inter_sabdab_AtSeqf = dataframe2fasta(inter_sabdab_df, temp_fasta=osp.join(temp_dir, 'At_fasta_InterSabdabDf.fasta'), seq_keys='At', fasta_seqIndexKey='redun_tempIndex', index_first=False)
    cls_dict, pdbid2datanum = cluster_DataByAntigen(rec_df=inter_sabdab_df)  # 根据抗原的90%序列一致性进行筛选, 最终得到980个subgroups
    cls_dict = get_tree_TaxonListCLS()
    cls_dict = CLS_dict(cls_dict)  # 将原本At中的
    # cls_dict.getFold_data(fold_n=Fold_number)  # 获得五折验证的数据
    print('\tclsutering_sabdab end\t')


def cluster_DataByAntigen(rec_df):
    if isinstance(rec_df, str):
        rec_df = pd.read_csv(rec_df, index_col=0)
    cls_dict, pdbid2datanum = clustrByAtseq(rec_df, merge_trunc=0.90, non_redun='sudo_atseq', ab_type='AbwithNb', abatid_flg=True)
    cls_dict_byRec = cluster_DataByPosRecord()  # 这里返回的是原文献中的数据集
    return cls_dict, pdbid2datanum


def get_abat_subset(cls_dict_subset, abatid):
    subset_dict = {}
    for k_ in cls_dict_subset:
        if abatid in cls_dict_subset[k_]:
            subset_dict[k_] = cls_dict_subset[k_]
    return subset_dict


def alter_abatid_subset(cls_dict_subset):
    def update_string(abatid):
        HLchs = list(abatid.split('_')[1])
        Atchs = abatid.split('_')[-1]
        return "{}_{}_{}".format(abatid.split('_')[0], '_'.join(HLchs), Atchs)

    for k_ in cls_dict_subset.keys():
        ls = cls_dict_subset[k_]
        cls_dict_subset[k_] = [update_string(ss) for ss in ls]
    return cls_dict_subset


def family_with_subset(cls_dict_subset, cls_dict_family):
    cls_dict_subset = alter_abatid_subset(cls_dict_subset)
    family_WithSub_dict = {k:{} for k in cls_dict_family.keys()}
    for fkey in cls_dict_family.keys():
        for abatid in cls_dict_family[fkey]:
            subset_dict = get_abat_subset(cls_dict_subset, abatid)
            subset_id = list(subset_dict.keys())
            if len(subset_id) < 1:  continue
            assert len(subset_id) == 1
            if subset_id[0] not in family_WithSub_dict[fkey].keys():
                family_WithSub_dict[fkey][subset_id[0]] = [abatid]
            else:
                if abatid not in family_WithSub_dict[fkey][subset_id[0]]:
                    family_WithSub_dict[fkey][subset_id[0]].append(abatid)
            print('')
    return family_WithSub_dict


def clustrByAtseq(df_in, data_dir=None, merge_trunc=0.9, non_redun='TotalAt', ab_type='Nb', task='get_neg', abatid_flg=False):
    vhvl_atls = []
    trunc_process_dir = osp.join(osp.dirname(ddp.init_pdbdir), 'ClusterSAbDab', ) if data_dir is None else osp.join(data_dir, 'ClusterSAbDab')
    cluster_dir = osp.join(trunc_process_dir, f'{ab_type}cluster_dir')
    os.makedirs(cluster_dir, exist_ok=True)

    nr_label = str(merge_trunc).split('.')[1]
    merge_db_fasta = osp.join(cluster_dir, f'ClusterByAtseq_nr{nr_label}.fasta')
    pdbid2datanum = {}
    # merge_usedf = df_in[df_in['sudo_at'] == df_in['sudo_ab']]
    merge_usedf = df_in
    with open(merge_db_fasta, 'w') as w:
        for i_ in merge_usedf.index.to_list():
            info_i = merge_usedf.loc[i_].to_dict()
            if not abatid_flg:
                if 'comp_id' not in info_i.keys():
                    VHVL_At = VHVL_At_chain(info_i)
                else:
                    VHVL_At = info_i['comp_id']
            else:
                VHVL_At = info_i['sudo_at'][:6] + info_i['sudo_at'][7:]
            if VHVL_At not in vhvl_atls:
                vhvl_atls.append(VHVL_At)
                if not abatid_flg:
                    pdbid2datanum[VHVL_At] = info_i['dataid_num'] if 'dataid_num' in info_i else i_
                    atseqs = info_i['Atseq'].split(',')
                    for chi, at_ch in enumerate(atchsFromSeries(info_i)):
                        w.write('>{0}_{1}\n'.format(VHVL_At, pdbid2datanum[VHVL_At]))
                        w.write('{0}\n'.format(atseqs[chi]))
                else:
                    # pdbid2datanum[VHVL_At] = info_i['pos_dataid'] if 'pos_dataid' in info_i else i_
                    pdbid2datanum[VHVL_At] = info_i['pos_dataid'] if 'pos_dataid' in info_i else i_
                    atseq = info_i['Atseq']
                    w.write('>{0}_{1}\n'.format(VHVL_At, pdbid2datanum[VHVL_At]))
                    w.write('{0}\n'.format(atseq))
    print('fasta for non-redundant --merge_trunc:{} get'.format(merge_trunc))
    nonreduner = NonRedun(merge_trunc=merge_trunc, non_redun=non_redun, task=task)
    data_num_idls, pdb_id_indexls = nonreduner.get_nr_index(merge_db_fasta, extract_fasta=None)
    print(f'index get, number is {len(data_num_idls)}')
    cls_dict = parse_clstr_file(nonreduner.cluster_fasta)  # 447个家族
    print(f'cls by {non_redun}, cls number is {max(cls_dict.keys())}')
    return cls_dict, pdbid2datanum  # 根据抗原相似性得到一个聚类的结果


def cluster_DataByPosRecord(rec_df=osp.join(osp.dirname(project_dir), 'init_data/ClusterSAbDab/positive_samples.txt')):
    if osp.isfile(rec_df):
        rec_df = pd.read_csv(rec_df, index_col=False, sep='\t', header=None)
    cls_dict, comp_ls = {0: []}, list(set(rec_df.iloc[:, 0].to_list() + rec_df.iloc[:, 1].to_list()))
    j_ = 0
    for i_ in range(rec_df.shape[0]):
        id1, id2 = rec_df.iloc[i_, 0], rec_df.iloc[i_, 1]
        flag_1, key_1 = check_idInValues(cls_dict, id1)
        flag_2, key_2 = check_idInValues(cls_dict, id2)
        if flag_2:
            cls_dict[key_2] = list(set(cls_dict[key_2] + [id2, id1]))
        if flag_1:
            cls_dict[key_1] = list(set(cls_dict[key_1] + [id2, id1]))
        if (not flag_1) and (not flag_2):
            cls_dict[j_] = list(set([] + [id2, id1]))
            j_ += 1
    cls_dict = remove_sub_dict(cls_dict)
    return cls_dict


def VHVL_At_chain(info_i):
    """输入Series数据或者对应的dict数据, 返回VHVL_At信息"""
    if 'Hchain' in info_i.keys():
        ab_chs = ''.join(series_values_byKeys(info_i, ['Hchain', 'Lchain']))
        atch_ls = [ch_.strip(' ') for ch_ in info_i['antigen_chain'].split('|')]
        at_chs = ''.join(atch_ls)
        pdb_obj = info_i['pdb']
    if not isinstance(info_i, dict):
        info_i = info_i.to_dict()
        ab_chs = ''.join(eval(info_i['sudo_abchains'])).replace('chain ', '')
        at_chs = ''.join(eval(info_i['sudo_atchains'])).replace('chain ', '')
        pdb_obj = info_i['parent_id']
    rec_i = f'{pdb_obj}_{ab_chs}_{at_chs}'
    return rec_i


def atchsFromSeries(si):
    atch_ls = [ch_.strip(' ') for ch_ in si['antigen_chain'].split('|')]
    return atch_ls


def parse_clstr_file(cls_f):
    cls_dict = {}
    with open(cls_f, 'r') as r:
        context = r.readlines()
    for line_i in context:
        if line_i.startswith('>Cluster'):
            cls_i = eval(line_i.strip('\n').split(' ')[-1])
            cls_dict[cls_i] = []
        elif 'aa,' in line_i:
            pdbid_i = line_i.split('>')[-1].split('...')[0]
            cls_dict[cls_i].append('_'.join(pdbid_i.split('_')[:-1]))
    return cls_dict


def _filter_errorfile(dir_path, f_ls, valid_ids=[]):
    error_dict, right_dict, err_ls, rm_validids = read_dict(ddp.error_dict_path), read_dict(ddp.right_dict_path), [], []
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


def check_idInValues(dict_in, id_in):
    flag_, key_o = False, None
    for key_ in list(dict_in.keys()):
        # print(key_)
        if id_in in dict_in[key_]:
            flag_, key_o = True, key_
            break
    return flag_, key_o


def remove_sub_dict(dict_in):
    for key_i in dict_in:
        for key_j in dict_in:
            if key_i != key_j:
                if len(set(dict_in[key_i]) - set(dict_in[key_j])) == 0:
                    dict_in.pop(key_i)
    return dict_in


def remove_unnamed_columns(df):
    # 使用 filter 函数选择不含有 "Unnamed" 的列
    filtered_columns = df.filter(regex='^(?!Unnamed).*$')    
    # 创建一个新的 DataFrame，包含不含有 "Unnamed" 的列
    new_df = df[filtered_columns.columns]
    return new_df


if __name__ == "__main__":
    clsutering_sabdab()
