import os
import torch
import os.path as osp
import numpy as np
import pandas as pd
import sys, shutil
from os.path import basename
project_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(project_dir)
from spec_At.parsePDB import ch_affine, merge_chs2pdb, merge_pdbs

from database.inter_pdb import get_atbox, get_box_coords, atom_Abcoords_ByResls, atom_Atcoords_ByResls, pdbresls2fasta, resls2aaseq
from database.parse_utils import series_values_byKeys, totaldict2
from database.inter_embed import embed_atbox, physicEmbed_atbox, get_pt_files_recursive
from utils.folding import get_sequence_dict, process_template
from esms_util.pred_byESM import predict_antibody_structure
from model.interface import DataDirPath
from esms_util.esm_parse_pdb import esm_pdb2resls
ddp = DataDirPath()
esm_Abpdbs_dir = osp.join(osp.dirname(__file__), 'esm_Abpdbs')


def try_load_atdata(abatid_resdir, dist_trunc=15.0):
    pt_files = get_pt_files_recursive(abatid_resdir)
    if dist_trunc != 15.0:
        flg = False
        for ptf in pt_files:
            if str(int(dist_trunc)) in basename(ptf)[4:]:
                flg = True
        if not flg:  return None, None
    atdata = None
    for fpath in pt_files:
        if 'Ats' in fpath:
            if dist_trunc != 15.0:
                if str(int(dist_trunc)) in basename(fpath)[4:]:
                    atdata = torch.load(fpath, map_location='cpu')
            elif 'A.pt' not in basename(fpath)[4:]:
                atdata = torch.load(fpath, map_location='cpu')
    return atdata


def try_load_abdata(preembed_dir, pdbobj, chls=None):
    if chls is not None:
        findstring = f'{pdbobj}_{"".join(chls)}'
    else:
        findstring = f'{pdbobj}'
    pt_files = []
    # 遍历目录并查找符合条件的文件
    for root, dirs, files in os.walk(preembed_dir):
        for f in files:
            if f.endswith('.pt') and findstring in f:
                pt_files.append(os.path.join(root, f))
    abdata = None
    if len(pt_files) > 0:
        assert len(pt_files) == 1
        ptfile = pt_files[0]
        abdata = torch.load(ptfile, map_location='cpu')
    return abdata


def get_pos_data(AbAtId_ls, pos_pdbs_dir, embeder=None, atseq=False, ab_pre=False, keep_initId=False):
    """AbAtId_ls:  列表, 其中元素为 pdbobj_H_L_At.pdb
    pos_pdbs_dir: 保存还有At以及Ab的pdb文件的文件夹
    db_args: 用于进行At Ab embed的模型以及相关参数"""
    db_args = embeder.database_args
    database_dir = osp.join(osp.dirname(pos_pdbs_dir), '{}database'.format(pos_pdbs_dir.split('/')[-2]))
    # merge_init_pdbs = [fpath for fpath in os.listdir(pos_pdbs_dir) if (fpath.endswith('.pdb') and 'temp' not in fpath)]
    sabdab_df = pd.read_csv(ddp.sabdab_summtsv, sep='\t', index_col=False)
    At_datals, Ab_datals = [], []
    if not osp.exists(esm_Abpdbs_dir):
        os.makedirs(esm_Abpdbs_dir, exist_ok=True)
    # keep_initId = True
    for abatid in AbAtId_ls:
        database_diri = database_dir if (not keep_initId) else osp.join(database_dir, abatid)
        Ab_data, At_data, pdbobj = None, None, abatid.split('_')[0]
        At_chs = list(abatid.split('_')[-1])

        if keep_initId and ab_pre:
            At_data = try_load_atdata(database_diri)
            Ab_data = try_load_abdata(osp.join(osp.dirname(esm_Abpdbs_dir), 'Abs/addattn_pre_embed'), abatid.split('_')[0])
            # Ab_data = try_load_abdata(osp.dirname(esm_Abpdbs_dir), abatid.split('_')[0], HL_chs)
        if (Ab_data is None or At_data is None):
            data_df = sabdab_df[sabdab_df['pdb'] == abatid.split('_')[0]].dropna(subset=['antigen_chain'])
            data_df = data_df[data_df['antigen_chain'].str.contains(' | '.join(At_chs))]
            if data_df.shape[0] > 0:
                HL_chs = series_values_byKeys(data_df, ['Hchain', 'Lchain'])
                At_chs = series_values_byKeys(data_df, 'antigen_chain')
            else:
                HL_chs = abatid.split('_')[1:-1]
            init_merge_file = osp.join(pos_pdbs_dir, f'{pdbobj}.pdb')
            ab_affine_dict, at_affine_dict, Abch_pdbs, Atch_pdbs = ch_affine(HL_chs, At_chs, init_merge_file, temp_dir=osp.join(pos_pdbs_dir, 'temp'))
            ab_affine_dict = sortdict_byKeyValue(ab_affine_dict)
            # Abch_pdbs = sorted(Abch_pdbs)  # 这样merge时让H chian在前, L chain在后
            HL_chs, At_chs = list(ab_affine_dict.values()), list(at_affine_dict.values())

            ab_file = merge_chs2pdb(Abch_pdbs, ab_affine_dict, temp_dir=osp.join(pos_pdbs_dir, 'temp'), nm='Ab')
            at_file = merge_chs2pdb(Atch_pdbs, at_affine_dict, temp_dir=osp.join(pos_pdbs_dir, 'temp'))
            merge_struct_file = osp.join(database_diri, "{}_{}_{}.pdb".format(pdbobj, ''.join(HL_chs), ''.join(At_chs)))
            merge_pdbs([ab_file, at_file], merge_struct_file)
            at_box_resls, atbox_chls, at_residues, ab_residue_ls = get_atbox(HL_chs, At_chs, merge_struct_file, dist_trunc=15.0)

            # -------------------- 获取Ab的序列数据 ----------------------
            pre_Abdata_path = osp.join(osp.dirname(esm_Abpdbs_dir), 'Abs_preEmbed', f'{pdbobj}_{"".join(HL_chs)}_At{"".join(atbox_chls)}PreData.pt') if embeder.embed_type is 'bert' else osp.join(osp.dirname(esm_Abpdbs_dir), 'Abs', 'pre_embed', f'{pdbobj}_{"".join(HL_chs)}_At{"".join(atbox_chls)}PhsicData.pt')
            # pre_Atdata_path = osp.join(osp.join(database_diri, abatid), 'Ats', 'pre_embed', f'{pdbobj}_{"".join(atbox_chls)}_PreData.pt') if embeder.embed_type is 'bert' else osp.join(osp.join(database_diri, abatid), 'Ats', 'pre_embed', f'{pdbobj}_{"".join(atbox_chls)}_PysicData.pt')
            pre_Atdata_path = osp.join(database_diri, 'Ats', 'pre_embed', f'{pdbobj}_{"".join(atbox_chls)}_PreData.pt') if embeder.embed_type is 'bert' else osp.join(database_diri, 'Ats', 'pre_embed', f'{pdbobj}_{"".join(atbox_chls)}_PysicData.pt')
            if ab_pre and osp.exists(pre_Abdata_path):
                Ab_data = torch.load(pre_Abdata_path, map_location='cpu')
            else:
                ab_fasta_file = ab_file.replace(".pdb", ".fasta")
                pdbresls2fasta(ab_residue_ls, ab_fasta_file)
                # output_pdb_file = predict_antibody_structure(ab_fasta_file, esm_Abpdbs_dir)
                esm_pred_pdb = osp.join(esm_Abpdbs_dir, basename(ab_fasta_file).replace(".fasta", ".pdb"))
                ab_residue_ls, right_flg = esm_pdb2resls(esm_pred_pdb, HL_chs, ab_residue_ls)  # 替换为esmfold预测结构的resls
                if right_flg is False:
                    shutil.copy(ab_fasta_file, osp.join(osp.dirname(esm_Abpdbs_dir), 'Abs'))
                    ab_pre = False
                label_coords, r_rangels, cdr_fr_dict, ab_rei_distAt = get_Abcoords_label(ab_residue_ls, ab_fasta_file, HL_chs)
                assert label_coords.size(0) == len(ab_rei_distAt)
                if None in [label_coords, r_rangels]:  pass  # 占位, 需要语句进行处理
                seq_dict = get_sequence_dict(sequences=None, fasta_file=ab_fasta_file)
                Ab_data = {'seq_dict': seq_dict, 'label_coords': label_coords, 'cdr_fr_dict': cdr_fr_dict, 'ab_rei_distAt': ab_rei_distAt, 'ab_fasta_file': ab_fasta_file,}
                if ab_pre:
                    os.makedirs(osp.dirname(pre_Abdata_path), exist_ok=True)
                    torch.save(Ab_data, pre_Abdata_path)
                # -------------------- 获取At数据 -------------------------
            if osp.exists(pre_Atdata_path):
                At_data = torch.load(pre_Atdata_path, map_location='cpu')
            else:
                # if embeder.embed_type is 'bert':
                At_embeddings, chi_attentionsLs, atbox_coords = get_At_data(at_box_resls, atbox_chls, at_residues, At_chs, embeder=embeder, at_fasta_file=None)
                # atbox_coords = torch.FloatTensor(atbox_coords, device=db_args['device'])
                atbox_coords = torch.tensor(data=atbox_coords, dtype=torch.float32, device=db_args['device'])
                At_data = {'atbox_coords': [atbox_coords], 'At_embeddings': At_embeddings, 'chi_attentionsLs': chi_attentionsLs, 'fpath': merge_struct_file}
                if atseq:
                    At_data['atseq'] = resls2aaseq(at_residues)
                os.makedirs(osp.dirname(pre_Atdata_path), exist_ok=True)
                At_data[abatid] = merge_struct_file
                torch.save(At_data, pre_Atdata_path)
        At_datals.append(At_data)
        Ab_datals.append(Ab_data)
    return At_datals, Ab_datals


def get_inter_data(fpath, ab_pdbpath, dfi, template_pdb, ab_fasta_file, ignore_cdrs, ignore_chain, dock_dir, db_args=None, pre_embed=True):
    """对有inter信息也就是有At chains的正样本对数据进行处理"""
    Ab_data, At_data = {}, {}
    if '.fasta' not in ab_pdbpath:
        ab_fasta_file = ab_pdbpath.split('.')[0] + '.fasta'
    pdb_obj = basename(ab_pdbpath).split('.')[0]
    assert isinstance(dfi, pd.DataFrame) or isinstance(dfi, pd.Series)
    # only_struct_file = dfi.iloc[0]['onlystruct_f']  merge_struct_file = dfi['merge_pdbpath']
    merge_struct_file = fpath
    HL_chs = series_values_byKeys(dfi, ['Hchain', 'Lchain'])
    At_chs = series_values_byKeys(dfi, 'antigen_chain')
    at_box_resls, atbox_chls, at_residues, ab_residue_ls = get_atbox(HL_chs, At_chs, merge_struct_file, dist_trunc=15.0)

    pre_Abdata_path = osp.join(osp.dirname(ab_pdbpath), 'pre_embed', f'{pdb_obj}_PreData.pt')
    pre_Atdata_path = osp.join(dock_dir, 'dock_temp', 'Ats', 'pre_embed', f'{pdb_obj}_{"".join(atbox_chls)}_PreData.pt')
    #  --------------------------- 获得Ab方面的数据 ------------------------
    if osp.exists(pre_Abdata_path) and pre_embed:
        Ab_data = torch.load(pre_Abdata_path, map_location='cpu')
    else:
        label_coords, r_rangels, cdr_fr_dict, ab_rei_distAt = get_Abcoords_label(ab_residue_ls, ab_fasta_file, HL_chs)
        assert label_coords.size(0) == len(ab_rei_distAt)
        if None in [label_coords, r_rangels]:  pass  # 占位, 需要语句进行处理
        seq_dict = get_sequence_dict(sequences=None, fasta_file=ab_fasta_file)
        if not check_seqdict(seq_dict):  return None, None, None, None, None, None
        with open(ab_fasta_file, "w") as f:
            for chain, seq in seq_dict.items():
                f.write(">{}\n{}\n".format(chain, seq,))

        Ab_embeddings, Ab_attentions = db_args['Ab_embed_model'].embed(seq_dict.values(), return_attention=True,)

        truncate_sequences = True if torch.any(torch.isnan(label_coords)) else r_rangels[0]
        if not truncate_sequences:
            label_coords = label_coords.unsqueeze(0).to(db_args['device'])
            Ab_embeddings = [e[1:-1].unsqueeze(0) for e in Ab_embeddings]
            Ab_attentions = [a[:, :, 1:-1, 1:-1].unsqueeze(0) for a in Ab_attentions]
        else:
            # 在获取get_coords_label过程中会遇到一些有间隔的氨基酸坐标缺失的情况, 造成计算O坐标时出现nan值
            range_list, range_dict = get_continues_range(r_rangels[1], seq_dict)
            if torch.isnan(label_coords[:, :, :4][range_list]).sum() > 0:
                range_list, range_dict, label_coords = rangeby_nan(range_list, range_dict, label_coords, db_args['device'])
            else:
                label_coords = label_coords[range_list].unsqueeze(0).to(db_args['device'])
            Ab_embeddings = [e[1 + np.array(range_dict[i])].unsqueeze(0) for i, e in enumerate(Ab_embeddings)]
            Ab_attentions = [a[:, :, 1 + np.array(range_dict[i]), :][:, :, :, 1 + np.array(range_dict[i])].unsqueeze(0) for i, a in enumerate(Ab_attentions)]

        # seq_lens, attn_lens = [max(e.shape[1], 0) for e in Ab_embeddings], [attni.size(-1) for attni in Ab_attentions]
        # assert seq_lens == attn_lens
        Ab_temp_coords, Ab_temp_mask = process_template(
            template_pdb,
            ab_fasta_file,
            ignore_cdrs=ignore_cdrs,
            ignore_chain=ignore_chain,
        )
        Ab_data = {'label_coords': label_coords, 'cdr_fr_dict': cdr_fr_dict, 'ab_rei_distAt': ab_rei_distAt,
                'Ab_embeddings': Ab_embeddings, 'Ab_attentions': Ab_attentions, 'Ab_temp_coords': Ab_temp_coords,
                'Ab_temp_mask': Ab_temp_mask, 'ab_fasta_file': ab_fasta_file,}
        torch.save(Ab_data, pre_Abdata_path)

    # ------------------------ 获得At方面的数据 -------------------------
    if osp.exists(pre_Atdata_path) and pre_embed:
        At_data = torch.load(pre_Atdata_path, map_location='cpu')
    else:
        At_embeddings, chi_attentionsLs, atbox_coords = get_At_data(at_box_resls, atbox_chls, at_residues, At_chs, db_args=db_args, at_fasta_file=None)
        # atbox_coords = torch.FloatTensor(atbox_coords, device=db_args['device'])
        atbox_coords = torch.tensor(data=atbox_coords, dtype=torch.float32, device=db_args['device'])
        At_data = {'atbox_coords': [atbox_coords], 'At_embeddings': At_embeddings, 'chi_attentionsLs': chi_attentionsLs, 'fpath': fpath}
        os.makedirs(osp.dirname(pre_Atdata_path), exist_ok=True)
        torch.save(At_data, pre_Atdata_path)
    return Ab_data, At_data


def get_At_data(at_box_resls, atbox_chls, at_residues, At_chs, embeder=None, at_fasta_file=None):
    atbox_coords = None
    atbox_resi_affine, dict_aaseq, dict_coords = atom_Atcoords_ByResls(at_box_resls, atbox_chls, at_residues, fasta_file=at_fasta_file, chls=At_chs)

    db_args = embeder.database_args
    if embeder.embed_type is 'Bert':
        model, alphabet, device = db_args['At_embed_model'], db_args['At_embed_alphabet'], db_args['device']
        embeddings, chi_attentionsLs, affine_dict = embed_atbox(atbox_resi_affine, dict_aaseq, model, alphabet, device)
    elif embeder.embed_type is 'Physic':
        embeddings, chi_attentionsLs, affine_dict = physicEmbed_atbox(atbox_resi_affine, dict_aaseq, db_args['device'])
    if sorted(list(affine_dict.keys())) != sorted(list(dict_coords.keys())):
        drop_keys = list(set(dict_coords.keys()) - set(affine_dict.keys()))
        for k_ in drop_keys:
            del dict_coords[k_]
            del dict_aaseq[k_]
            for ind_, value in enumerate(atbox_resi_affine):
                if value[0] in drop_keys:
                    atbox_resi_affine.remove(value)
    atbox_coords = get_box_coords(atbox_resi_affine, dict_coords)
    return embeddings, chi_attentionsLs, atbox_coords


def get_Abcoords_label(Ab_residues, ab_fastaf, HL_chs):
    label_coords, temp_mask = None, None
    label_coords, r_rangels, cdr_fr_dict, ab_rei_distAt = atom_Abcoords_ByResls(Ab_residues, fasta_file=ab_fastaf, chls=HL_chs)
    if None in [label_coords, r_rangels]:
        return None, None, None, None, None
    for key_ in label_coords.keys():
        label_coords[key_] = label_coords[key_].unsqueeze(-2)
    label_coords = torch.cat([label_coords['N'], label_coords['CA'], label_coords['C'], label_coords['CB'], label_coords['O']], dim=-2)
    return label_coords, r_rangels, cdr_fr_dict, ab_rei_distAt


def check_seqdict(seq_dict):
    ss = ''
    if len(seq_dict.values()) == 0:  return False
    for s_ in seq_dict.values():
        if isinstance(s_, str):  ss += s_
    return len(ss) > 0


def get_continues_range(range_dict, seq_dict):
    prim_ch = 'L' if (len(seq_dict.keys()) == 1 and list(seq_dict.keys())[0] == 'L') else 'H'
    range_list, r_stls, range_dict_temp = [], [0, len(seq_dict[prim_ch])], {}
    for key_ in range_dict:
        range_dict_temp[key_] = []
        for (st_i, ed_i) in range_dict[key_]:
            if ed_i is not None:
                range_list = (range_list + list(range(r_stls[key_] + st_i, r_stls[key_] + ed_i)))
                range_dict_temp[key_] = range_dict_temp[key_] + list(range(st_i, ed_i))
            elif key_ == 0:
                range_list = (range_list + list(range(r_stls[0] + st_i, r_stls[1])))
                range_dict_temp[key_] = range_dict_temp[key_] + list(range(r_stls[0] + st_i, r_stls[1]))
            elif key_ == 1:
                range_list = (range_list + list(range(r_stls[1] + st_i, r_stls[1] + len(seq_dict['L']))))
                range_dict_temp[key_] = range_dict_temp[key_] + list(range(st_i, len(seq_dict['L'])))
    return range_list, range_dict_temp


def rangeby_nan(range_list, range_dict, label_coords, device):
    l0 = len(range_dict[0])
    l1 = len(range_dict[1]) if 1 in range_dict.keys() else None
    lenls = [0, l0]

    atom_with_nan = torch.any(torch.isnan(label_coords[:, :4, :]), dim=-1)
    index_nan_ls = sorted(list(set(torch.where(atom_with_nan)[0].cpu().numpy())))
    for nan_id in index_nan_ls:
        if nan_id in range_list:
            range_list.remove(nan_id)
        if nan_id < l0 and nan_id in range_dict[0]:
            range_dict[0].remove(nan_id)
        elif nan_id >= l0:
            if (nan_id - l0) in range_dict[1]:
                range_dict[1].remove(nan_id - l0)
    if len(range_list) == len(totaldict2(range_dict)):
        label_coords = label_coords[range_list].unsqueeze(0).to(device)
    else:
        range_list_temp = []
        for k_, leni in zip(list(range_dict.keys()), lenls):
            range_list_temp = range_list_temp + list(np.array(range_dict[k_]) + leni)
        range_list = range_list_temp
        label_coords = label_coords[range_list].unsqueeze(0).to(device)
    return range_list, range_dict, label_coords


def sortdict_byKeyValue(dict_in):
    sorted_values = sorted(list(dict_in.values()))
    sorted_dict = {}
    for v_ in sorted_values:
        for k, vt in dict_in.items():
            if v_ == vt:
                sorted_dict[k] = vt
    # 创建一个新的有序字典
    # sorted_dict = {key: dict_in[key] for key in sorted_keys}
    return sorted_dict
