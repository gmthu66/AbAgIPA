import sys
import torch
import os
import os.path as osp
import pandas as pd
data_pardir = osp.dirname(__file__)
project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)
from tqdm import tqdm
from database.parse_utils import series_values_byKeys
from spec_At.data_code.Embedding import Embeder
from model.interface import DataDirPath
from database.preprocess.collect_utils import movepdb_from_sabdab
from database.numbering import get_cdrfr_info
from database.inter_embed import physicEmbed_ab
from spec_At.HER2.inter_utils import get_pos_data
from spec_At.parse_seqs import aaseqDict2fasta, add_aaseqDict2fasta
from utils.folding import process_template, process_MutantTemplate
from utils.general import exists
pdbf_name = {'SARS-CoV1': '2dd8_S.pdb', 'SARS-CoV2': '7tbf_A.pdb'}
ddp = DataDirPath()


def vhvl2total(df_in):
    for i_ in range(df_in.shape[0]):
        if isinstance(list(df_in.index)[0], str):
            i_ = str(i_)
        df_in.loc[i_, 'AASeq'] = df_in.loc[i_, 'VH_seq'] + df_in.loc[i_, 'VL_seq']
        df_in.loc[i_, 'len_0'] = len(df_in.loc[i_, 'VH_seq'])
        df_in.loc[i_, 'len_1'] = len(df_in.loc[i_, 'VL_seq'])
    return df_in


def mixcr_input(file_name, Ag_class, seq_len=None, noRedun=True):
    """
    Read in data from the MiXCR txt output file
    ---
    file_name: file name of the MiXCR txt file to read

    Ag_class: classification of sequences from MiXCR txt file
               (i.e., antigen binder = 1, non-binder = 0)

    seq_len: the length of sequences; other lengths will be
             removed.
    """
    # Read data and rename columns
    x = pd.read_table(file_name)
    x = pd.read_csv(file_name, sep='\t', header=None)

    x = x.rename(index=str, columns={
        0: 'At_name', 1: 'VH_seq',
        2: 'VL_seq'
    })
    x = vhvl2total(x)
    # Select length and drop duplicate sequences
    if seq_len is not None:
        x = x[(x.AASeq.str.len() == seq_len) & (x.Count > 1)]
    if noRedun:
        x = x.drop_duplicates(subset='AASeq')

    # Remove stop codons and incomplete codon sequences (*, _)
    idx = [i for i, aa in enumerate(x['AASeq']) if '*' not in aa]
    x = x.iloc[idx, :]
    idx = [i for i, aa in enumerate(x['AASeq']) if '_' not in aa]  # 删除带有下划线以及星号的CDRH3序列数据
    x = x.iloc[idx, :]
    # x.to_csv(file_name.split('.')[0] + '.csv', index=False)  # 测试

    if Ag_class == 0:
        x['AgClass'] = 0
    if Ag_class == 1:
        x['AgClass'] = 1
    return x


def load_input_data(filenames, Ag_class, noRedun=True):
    # Combine the non-binding sequence data sets.
    # Non-binding data sets include Ab+ data and Ag-
    # sorted data for all 3 libraries
    l_data = []
    for file in filenames:
        l_data.append(
            mixcr_input(osp.join(data_pardir, 'data/' + file), Ag_class, noRedun=noRedun)
        )
    mHER_H3 = pd.concat(l_data)
    # Drop duplicate sequences
    if noRedun:
        mHER_H3 = mHER_H3.drop_duplicates(subset='AASeq')  # 对序列数据进行去冗余
    # Remove 'CAR/CSR' motif and last two amino acids
    # mHER_H3['AASeq'] = [x[3:-2] for x in mHER_H3['AASeq']]
    # Shuffle sequences and reset index
    mHER_H3 = mHER_H3.sample(frac=1).reset_index(drop=True)
    return mHER_H3


def parseAbAtId(AbAtId):
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


def process_SingleAt(embeder, rec_property=None, database_args=None, atseq=False):
    AbAtId, InitIsTotal, Pos = rec_property['AbAtId'], rec_property['InitIsTotal'], rec_property['Pos']
    AbAtId_ls, pdb_dir = parseAbAtId(AbAtId)
    At_datals, Ab_datals = get_pos_data(AbAtId_ls, pdb_dir, embeder=embeder, atseq=atseq)
    return At_datals, Ab_datals


def modify_col(df, before_col, after_col):
    df.columns = [col if col != before_col else after_col for col in df.columns]
    return df


def renew_mutant_Abdf(df_in, rec_property):
    df_in = modify_col(df_in, 'VH_seq', 'Hseq')
    df_in = modify_col(df_in, 'VL_seq', 'Lseq')
    for i_ in tqdm(list(df_in.index)):
        for reci in rec_property:
            if reci['AtId_ls'] in df_in.loc[i_, 'At_name']:
                df_in.at[i_, 'merge_obj'] = reci['AbAtId']
                df_in.at[i_, 'pos_dataid'] = eval(reci['AtId_ls'][-1]) - 1
                hlen = len(df_in.loc[i_, 'Hseq']) if not pd.isna(df_in.loc[i_, 'Hseq']) else len(df_in.loc[i_, 'Lseq'])
                cdr_fr_info = get_cdrfr_info(ab_aaseq=df_in.loc[i_, 'AASeq'], h_len=hlen)
                df_in.loc[i_, 'cdr_fr_info'] = str(cdr_fr_info)  # 添加cdr_fr信息
                # for k_, v_ in cdr_fr_info.items():
                #     df_in.loc[i_, k_] = str(v_)  # 添加cdr_fr信息
    print('renew_mutant_Abdf end...')
    return df_in


def get_data(embeder, sudoAbdist_property=None, AtEmbed_property=None, more_neg=0, noRedun=True, atseq=False):
    if noRedun:
        Covid_AgPos_path, Covid_AgNeg_path = osp.join(data_pardir, 'data/Covid_AgPos.csv'), osp.join(data_pardir, 'data/Covid_AgNeg.csv')
        Covid_df_path = osp.join(data_pardir, 'Covid19database', 'Covid_df.csv')
    else:
        Covid_AgPos_path, Covid_AgNeg_path = osp.join(data_pardir, f'data/Covid_AgPos_{noRedun}.csv'), osp.join(data_pardir, f'data/Covid_AgNeg_{noRedun}.csv')
        Covid_df_path = osp.join(data_pardir, 'Covid19database', f'Covid_df_{noRedun}.csv')
    if osp.exists(Covid_df_path):
        Covid_df = pd.read_csv(Covid_df_path, index_col=False)
        rec_property = {'AtId_ls': list(pdbf_name.keys()), 'Pos': True, 'InitIsTotal': True}
        rec_property = get_rec_property(pdbf_name, pdbf_name_with='At')
    else:
        if osp.exists(Covid_AgPos_path) and osp.exists(Covid_AgNeg_path):
            Covid_AgNeg = pd.read_csv(Covid_AgNeg_path, index_col=False)
            Covid_AgPos = pd.read_csv(Covid_AgPos_path, index_col=False)
        else:
            ab_neg_files = ['negative_dataset.txt']
            Covid_AgNeg = load_input_data(ab_neg_files, Ag_class=0, noRedun=noRedun)
            ab_pos_files = ['positive_dataset.txt']
            Covid_AgPos = load_input_data(ab_pos_files, Ag_class=1, noRedun=noRedun)
            Covid_AgNeg['data_id'], Covid_AgPos['data_id'] = range(Covid_AgNeg.shape[0]), range(Covid_AgPos.shape[0])
            # Save those files
            Covid_AgNeg.to_csv(Covid_AgNeg_path)
            Covid_AgPos.to_csv(Covid_AgPos_path)
        # Covid_AgNeg 数目为1672条, Covid_AgPos数目为7491条
        Covid_df = pd.concat([Covid_AgNeg, Covid_AgPos], ignore_index=True)
        Covid_df = Covid_df.sample(frac=1, random_state=22)  # 将顺序打乱

        rec_property = {'AtId_ls': list(pdbf_name.keys()), 'Pos': True, 'InitIsTotal': True}
        rec_property = get_rec_property(pdbf_name, pdbf_name_with='At')
        Covid_df = renew_mutant_Abdf(Covid_df, rec_property)

    At_datals, Ab_datals = [], []
    for reci in rec_property:
        At_data, Ab_data = process_SingleAt(embeder, rec_property=reci, atseq=atseq)  # 从复合结构pdb文件中提取初始的抗体序列以及抗原口袋
        appen_datals(At_datals, At_data)
        appen_datals(Ab_datals, Ab_data)

    if osp.exists(osp.join(data_pardir, 'pdbs', 'temp', 'db_collect.fasta')):
        os.remove(osp.join(data_pardir, 'pdbs', 'temp', 'db_collect.fasta'))
    return Covid_df, Ab_datals, At_datals 


def get_dfTransf():
    train_df = pd.read_csv('/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/spec_At/Covid19/Covid19database/fold_1_train.csv', index_col=False)
    val_df = pd.read_csv('/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/spec_At/Covid19/Covid19database/fold_1_val.csv', index_col=False)
    data_df = pd.concat([train_df, val_df], axis=0)
    return data_df


def appen_datals(datals, datai):
    if isinstance(datai, list):
        datals.extend(datai)
    else:
        datals.append(datai)


def process_Abdata(aaseq_dict, wild_Abdata_dict=None, db_args=None, pos_index=None, rec_cdrfr=None, template_pdb=None, ignore_cdrs=None, ignore_chain=None, tmp_flg=False, mutate_range=None, map_cpu=False):
    if isinstance(wild_Abdata_dict, list):
        assert pos_index is not None
        wild_Abdata_dict = wild_Abdata_dict[pos_index]  # 若pos data是一个列表, 则传入参数中应当有指定的pos_index
    if wild_Abdata_dict is not None:
        label_coords = wild_Abdata_dict['label_coords'] if mutate_range is not None else None
        cdr_fr_dict = wild_Abdata_dict['cdr_fr_dict'] if mutate_range is not None else rec_cdrfr
    Ab_embeddings, Ab_attentions = db_args['Ab_embed_model'].embed(aaseq_dict.values(), return_attention=True,)
    if not map_cpu:
        Ab_embeddings = [e[1:-1].unsqueeze(0) for e in Ab_embeddings]
        Ab_attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0) for a in Ab_attentions]
    else:
        Ab_embeddings = [e[1:-1].unsqueeze(0).to(torch.device('cpu')) for e in Ab_embeddings]
        Ab_attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0).to(torch.device('cpu')) for a in Ab_attentions]
    if not exists(db_args['At_embed_model']):
        Ab_phisicEmbed, Ab_phisicAttention = physicEmbed_ab(aaseq_dict)

    ab_fasta_file = osp.join(data_pardir, 'pdbs', 'temp', 'db_collect.fasta')
    # aaseqDict2fasta(aaseq_dict, ab_fasta_file)
    add_aaseqDict2fasta(aaseq_dict, ab_fasta_file)
    if not tmp_flg:
        Ab_temp_coords, Ab_temp_mask = process_template(
            template_pdb,
            ab_fasta_file,
            ignore_cdrs=ignore_cdrs,
            ignore_chain=ignore_chain,
        )
    else:
        Ab_temp_coords, Ab_temp_mask = process_MutantTemplate(label_coords, cdr_fr_dict, mutate_range, ignore_cdrs=ignore_cdrs, ignore_chain=ignore_chain,)  # mut_range范围内的氨基酸不会作为模板信息
    Ab_data = {'label_coords': label_coords, 'cdr_fr_dict': cdr_fr_dict, 'ab_rei_distAt': None,
                'Ab_embeddings': Ab_embeddings, 'Ab_attentions': Ab_attentions, 'Ab_temp_coords': Ab_temp_coords,
                'Ab_temp_mask': Ab_temp_mask, 'ab_fasta_file': ab_fasta_file, 'mutate_range': mutate_range,}
    if not exists(db_args['At_embed_model']):
        init_dict = {'Ab_batch_mask': None, 'Ab_align_mask': None, 'Ab_phisicEmbed': Ab_phisicEmbed, 'Ab_phisicAttention': Ab_phisicAttention}
    else:
        init_dict = {'Ab_batch_mask': None, 'Ab_align_mask':None}
    Ab_data.update(init_dict)
    return Ab_data


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
            df_info = df_info[df_info['antigen_chain'].str.contains(item_name[-1])]
        if df_info.shape[0] > 1: df_info = df_info.iloc[0].to_frame().transpose()
        assert df_info.shape[0] == 1

        At_chs = series_values_byKeys(df_info, 'antigen_chain')
        HL_chs = series_values_byKeys(df_info, ['Hchain', 'Lchain'])
        AbAtId = '{}_{}_{}'.format(pdb_obj, '_'.join(HL_chs), ''.join(At_chs))
        rec_property['AbAtId'] = AbAtId
        AbAtId_ls.append(AbAtId)
        rec_property_ls.append(rec_property)    

    return rec_property_ls


if __name__ == "__main__":
    device_in = torch.device('cuda:1')
    embeder = Embeder(device=device_in)
    Covid_df, Ab_datals, At_datals = get_data(embeder)
    Covid_df.to_csv(osp.join(data_pardir, 'Covid19database', 'Covid_df.csv'))
