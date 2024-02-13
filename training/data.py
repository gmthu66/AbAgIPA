import os
import os.path as osp
import logging
import pandas as pd
import pymol2 as pm2
import sys
import math, esm
import numpy as np
import torch

project_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(project_dir)
from os.path import splitext, basename
from Bio.PDB import PDBParser
from antiberty import AntiBERTyRunner
from tqdm import tqdm
from database.download import Spider
from database.parse_utils import get_chainsseq, PDB_pre, match_chain, return_key, totaldict2
from database.inter_data import insight_inter_noninterDf, get_fpaths_databaseDf, get_inter_data, get_Abs_data
from database.dock_data import get_negpair_info, change_fpath_byPreEmbedDir
from database.batch_data import stack_attentions
from utils.abnumber_ import renumber_pdb, cdrDict2Array, cdrDict2MaskTensor

from model.interface import IgFoldInput, read_dict, DataDirPath
from training.dataset import SamplerDataFrame, get_train_valid_test_datadf, save_pre_embed, load_pre_embed, rightpdb_size
from torch.utils.data import Dataset, DataLoader

ddp = DataDirPath()


class GenDataloader(object):
    def __init__(self, config, valid_ids=None, foldi=None):
        # self.test_flag = split == 'test'
        self.ddp = DataDirPath()
        self.test_dir = [osp.join(self.ddp.test_dirs, d_) for d_ in os.listdir(self.ddp.test_dirs)]
        self.split_dir = self.ddp.train_dirs
        self.batch_size = config.batch_size
        self.num_data_workers = config.num_data_workers
        self.trainval_negs = config.trainval_negs

        self._init_datasets(config, valid_ids, foldi)
        self._init_samplers()
        self._init_loaders()

    def _init_datasets(self, config, valid_ids=None, foldi=None):
        data_dirs = {'train': self.split_dir, 'valid': self.split_dir, 'test': self.test_dir}
        dict_train_valid_test_df = get_train_valid_test_datadf(data_dirs, valid_ids=valid_ids, ddp=self.ddp)  # 修改成了只是以抗体预测结构
        self.train_set = Dataset_3d(self.split_dir, valid_ids=valid_ids, test_flag=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir, foldi=foldi)
        self.valid_set = Dataset_3d(self.split_dir, valid_ids=valid_ids, test_flag=False, trainval_negs=self.trainval_negs, valid_flag=True, device=config.device, ddp=self.ddp, out_dir=config.out_dir, foldi=foldi)
        self.test_set = Dataset_3d(self.test_dir, valid_ids=valid_ids, test_flag=True, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir, foldi=foldi)
        config.is_master = True
        if config.is_master:
            msg = [f'Protein Docking task, train: {len(self.train_set.fpaths)},',
                   f'val: {len(self.valid_set.fpaths)}, test: {len(self.test_set.fpaths)}']
            logging.info(' '.join(msg))

    def _init_samplers(self):
        self.train_sampler = SamplerDataFrame(self.train_set, partion='train')
        self.valid_sampler = SamplerDataFrame(self.valid_set, partion='valid')
        self.test_sampler = SamplerDataFrame(self.test_set, partion='test')

    def _init_loaders(self):
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=self.num_data_workers,
                                       collate_fn=self.train_set.collate_wrapper,
                                       pin_memory=torch.cuda.is_available(),)
        self.valid_loader = DataLoader(self.valid_set,
                                       batch_size=self.batch_size,
                                       sampler=self.valid_sampler,
                                       num_workers=self.num_data_workers,
                                       collate_fn=self.valid_set.collate_wrapper,
                                       pin_memory=torch.cuda.is_available())
        self.test_loader = DataLoader(self.test_set,
                                      batch_size=self.batch_size,
                                      sampler=self.test_sampler,
                                      num_workers=self.num_data_workers,
                                      collate_fn=self.test_set.collate_wrapper,
                                      pin_memory=torch.cuda.is_available())


class Dataset_3d(Dataset):
    def __init__(self, data_dirs, valid_ids=None, test_flag=False, valid_flag=False, trainval_negs=5, device='cpu', ddp=None, out_dir=None, atbox_truc=15.0, foldi=0):
        self.valid_ids, self.ddp, self.out_dir, self.trainval_negs = valid_ids, ddp, out_dir, trainval_negs
        self.fpaths, self.idx_dict, self.no_pdbls, st_ = [], {}, [], 0
        self.test_flag, self.valid_flag, self.device, self.atbox_truc = test_flag, valid_flag, device, atbox_truc

        self.inter_pdbobj_ls, self.noninter_pdbobj_ls, self.inter_noninter_dict, self.sabdab_interdf = insight_inter_noninterDf(ddp.inter_noninter_dict_info['inter'], ddp.inter_noninter_dict_info['noninter'], ddp)
        # sabdab_interdf中是包含所有数据信息的
        for step, _dir in enumerate(data_dirs):  # 依据only_struct, 获取oas_paired、oas_unpaired、sabdab以及test数据的index字典, 便于计算每个数据集的大小, 以便于进行数据采样
            sub_fs = [f_ for f_ in os.listdir(_dir) if (('.pdb' in f_) or ('.fasta' in f_)) and not f_.startswith('.')]
            if osp.exists(self.ddp.error_dict_path) and osp.exists(self.ddp.right_dict_path):
                sub_fs, valid_ids = self._filter_errorfile(_dir, sub_fs, valid_ids)
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
            self.fpaths.extend(sub_fs)
            if len(sub_fs) > 0:
                self.idx_dict[step] = list(range(st_, st_ + len(sub_fs)))
            if len(sub_fs) > 0:
                st_ = (self.idx_dict[step][-1] + 1)

        total_inter_noninter_fpath = osp.join(osp.dirname(osp.dirname(__file__)), 'database', 'total_inter_noninter.tsv')
        self.total_df = get_fpaths_databaseDf(self.fpaths, self.ddp, valid_ids, opath=total_inter_noninter_fpath,)  # 需要基于inter pdb_objls以及only_struct objls产生一些noninter的负作用数据
        # get_negpair_info的过程是赋予了权重的采样过程, 对于sabdab来源以及paired、unpaid来源的数据进行了均衡
        inter_withneg_df, test_withnegdf, valid_withnegdf = osp.join(project_dir, f'database/csvs/inter_withneg_df{foldi}.csv'), osp.join(project_dir, f'database/csvs/valid_withnegdf{foldi}.csv'), osp.join(project_dir, 'database/csvs/test_withnegdf.csv')
        self.posinter_df, self.valid_df, self.test_df, self.total_df = get_negpair_info(self.total_df, atbox_truc, valid_ids, inter_withneg_df, valid_withnegdf, test_withnegdf)
        # self.posinter_df, self.valid_df, self.test_df = map(lambda dfin: change_fpath_byPreEmbedDir(dfin), [self.posinter_df, self.valid_df, self.test_df])
        self.data_df = get_datadf(self, test_flag, valid_flag)
        self.total_df = change_fpath_byPreEmbedDir(self.total_df)
        self.data_df = change_fpath_byPreEmbedDir(self.data_df)

        self.antiberty = AntiBERTyRunner(device=self.device)
        # self.antiberty = AntiBERTyRunner(device=torch.device('cpu'))
        self.antiberty.model.eval()
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.logger = open(osp.join(self.out_dir, 'data_fpath.log'), 'a', encoding='utf-8')
        self.debug_iteri = 0

    def __getitem__(self, idx, template_pdb=None, fasta_file=None, ignore_cdrs=None, ignore_chain=None, rebuild_pre_embed=False, use_fresh_embed=True):
        self.debug_iteri += 1
        data_si = self.data_df.loc[idx]
        fpath, ab_pdbpath, pdbobj = data_si['merge_pdbpath'], data_si['ab_pdbpath'], data_si['pdb']
        # if self.debug_iteri < 2425:  return None, None, None
        # if '7akj' not in fpath:  return None, None, None  #debug
        if not pdbobj in self.inter_pdbobj_ls:  return None, None, None
        # debug
        # fpath = '/data/gm_data/IgFold_data/sabdab/7ce2.pdb'
        print('\t', fpath)
        print(self.debug_iteri, '\t', fpath, file=self.logger, flush=True)
        database_args = {'device': self.device, 'Ab_embed_model': self.antiberty, 'At_embed_model': self.esm_model, 'At_embed_alphabet': self.esm_alphabet, 'atbox_truc': self.atbox_truc}

        # 这里处理的是负样本
        negAb_indexs = list(np.random.choice(eval(data_si['neg_indexls']), self.trainval_negs, replace=False))
        negab_datas = get_Abs_data(negAb_indexs, self.total_df, template_pdb, ignore_cdrs, ignore_chain, db_args=database_args)

        # 这里处理的是正样本
        Ab_data, At_data = get_inter_data(fpath, ab_pdbpath, data_si, template_pdb, fasta_file, ignore_cdrs, ignore_chain, db_args=database_args)
        return Ab_data, At_data, negab_datas

    @staticmethod
    def collate_wrapper(unbatched_list):
        return Batch_InteractData(unbatched_list)

    def __len__(self):
        return len(self.fpaths)

    def _get_continues_range(self, range_dict, seq_dict):
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


    def rangeby_nan(self, range_list, range_dict, label_coords):
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
            label_coords = label_coords[range_list].unsqueeze(0).to(self.device)
        else:
            range_list_temp = []
            for k_, leni in zip(list(range_dict.keys()), lenls):
                range_list_temp = range_list_temp + list(np.array(range_dict[k_]) + leni)
            range_list = range_list_temp
            label_coords = label_coords[range_list].unsqueeze(0).to(self.device)
        return range_list, range_dict, label_coords

    def check_pdb(self, _dir, sub_fs):
        f_ls = [osp.join(_dir, f_) for f_ in sub_fs if ('.pdb' in f_) and (not f_.startswith('.'))]
        p = PDBParser()
        for pdbf in tqdm(f_ls):
            file_name = splitext(basename(pdbf))[0]
            structure = p.get_structure(file_name, pdbf)
            ch_ls = [ch.id for ch in list(structure.get_chains())]
            if len(ch_ls) < 1:
                self.no_pdbls.append(pdbf)

    def check_seqdict(self, seq_dict):
        ss = ''
        if len(seq_dict.values()) == 0:  return False
        for s_ in seq_dict.values():
            if isinstance(s_, str):  ss += s_
        return len(ss) > 0


class Batch_InteractData(object):
    def __init__(self, unbatched_list, cdr2mask=True, paratope=True):
        self.Ab_embeddings = []
        self.Ab_attentions = []
        self.Ab_seqlens, self.Ab_VHLlens = [], []
        self.Ab_coords_label = []
        self.Ab_temp_coords = []
        self.Ab_temp_masks = []
        self.Ab_cdr_indexs = []
        self.Ab_temp_coords_flag, self.model_in, self.inter_label = False, None, []
        unbatched_list = unbatched_list[0]
        pos_ab, pos_at, neg_abs = unbatched_list

        if None not in [pos_ab, pos_at, neg_abs]:
            temp_coord = pos_ab['Ab_temp_coords']
            if temp_coord is not None:
                self.temp_coords_flag = True
            bert_attn_dim = pos_ab['Ab_attentions'][0].size(1) * pos_ab['Ab_attentions'][0].size(2)
            for negab in neg_abs:
                seq_lens = [max(e.shape[1], 0) for e in negab['Ab_embeddings']]
                self.Ab_seqlens.append(sum(seq_lens))
                self.Ab_VHLlens.append(seq_lens)
                self.Ab_attentions.append(stack_attentions(negab['Ab_attentions'], seq_lens, bert_attn_dim))
                self.Ab_coords_label = self.append_sgls(self.Ab_coords_label, negab['label_coords'])
                self.Ab_embeddings = self.append_sgls(self.Ab_embeddings, negab['Ab_embeddings'], type='embeddings')
                if self.Ab_temp_coords_flag:
                    self.Ab_temp_coords = self.append_sgls(self.Ab_temp_coords, temp_coord)
                    self.Ab_temp_masks = self.append_sgls(self.Ab_temp_masks, negab['Ab_temp_mask'])
                self.inter_label.append([1, 0])
                if cdr2mask:  self.Ab_cdr_indexs.append(cdrDict2MaskTensor(negab['cdr_fr_dict'], seqlen=sum(seq_lens)))

            seq_lens = [max(e.shape[1], 0) for e in pos_ab['Ab_embeddings']]
            self.Ab_seqlens.append(sum(seq_lens))
            self.Ab_VHLlens.append(seq_lens)
            self.Ab_attentions.append(stack_attentions(pos_ab['Ab_attentions'], seq_lens, bert_attn_dim))
            self.Ab_coords_label = self.append_sgls(self.Ab_coords_label, pos_ab['label_coords'])
            self.Ab_embeddings = self.append_sgls(self.Ab_embeddings, pos_ab['Ab_embeddings'], type='embeddings')
            self.inter_label.append([0, 1])
            if self.Ab_temp_coords_flag:
                self.Ab_temp_coords = self.append_sgls(self.Ab_temp_coords, temp_coord)
                self.Ab_temp_masks = self.append_sgls(self.Ab_temp_masks, pos_ab['Ab_temp_mask'])

            # 处理Atbox特征
            self.At_coords_label = pos_at['atbox_coords'] * len(self.Ab_embeddings)
            self.At_embeddings = pos_at['At_embeddings'] * len(self.Ab_embeddings)
            self.At_attentions, self.At_len = pos_at['chi_attentionsLs'] * len(self.Ab_embeddings), int(pos_at['At_embeddings'][0].size(0))
            if cdr2mask:  self.Ab_cdr_indexs.append(cdrDict2MaskTensor(negab['cdr_fr_dict'], seqlen=sum(seq_lens)))
            model_in = IgFoldInput(
                Ab_embeddings=self.Ab_embeddings,
                Ab_attentions=self.Ab_attentions,
                Ab_seqlens=self.Ab_seqlens,
                Ab_VHLlens=self.Ab_VHLlens,
                Ab_template_coords=self.Ab_temp_coords,
                Ab_template_mask=self.Ab_temp_masks,
                Ab_coords_label=self.Ab_coords_label,
                return_embeddings=True,
                At_embeddings=self.At_embeddings,
                At_attentions=self.At_attentions,
                At_coords_label=self.At_coords_label,
                Ab_cdr_indexs=self.Ab_cdr_indexs,
                At_len=self.At_len,
                fpaths=pos_at['fpath'],
                inter_label=torch.tensor(data=self.inter_label, dtype=torch.long, device=self.At_embeddings[0].device),
            )
            self.model_in = model_in
        else:
            self.model_in = None

    def __len__(self):
        return len(self.Ab_seqlens)

    def append_sgls(self, data_ls, data, type=None):
        if isinstance(data, list):
            if len(data) == 1:
                if type == 'embeddings':  data = data[0].squeeze(0)
                data_ls.append(data)
            elif len(data) > 1:
                if type == 'embeddings':  data = torch.cat(data, dim=-2).squeeze(0)
                data_ls.append(data)  # 将heavy链和轻链合在一起
        else:
            data_ls.append(data)
        return data_ls


class CustomBatchPro3d(object):
    def __init__(self, unbatched_list):
        self.embeddings = []
        self.attentions = []
        self.temp_coords = []
        self.temp_masks = []
        self.fpaths = []
        self.temp_coords_flag, self.model_in = False, None
        if len(unbatched_list) > 1:
            for (embedding, attention, temp_coord, temp_mask, fpath, label_coords) in unbatched_list:
                self.embeddings = self.append_sgls(self.embeddings, embedding)
                self.attentions = self.append_sgls(self.attentions, attention)
                if temp_coord is not None:
                    self.temp_coords_flag = True
                if self.temp_coords_flag:
                    self.temp_coords = self.append_sgls(self.temp_coords, temp_coord)
                    self.temp_masks = self.append_sgls(self.temp_masks, temp_mask)
                self.fpaths = self.append_sgls(self.fpaths, fpath)
            self.embeddings = torch.cat(self.embeddings, dim=0)
            self.attentions = torch.cat(self.attentions, dim=0)
            if self.temp_coords_flag:
                self.temp_coords = torch.cat(self.temp_coords, dim=0)
                self.temp_masks = torch.cat(self.temp_masks, dim=0)
        else:
            # 获得模型的输入数据
            unbatched_list = unbatched_list[0]
            model_in = IgFoldInput(
                embeddings=unbatched_list[0],
                attentions=unbatched_list[1],
                template_coords=unbatched_list[2],
                template_mask=unbatched_list[3],
                coords_label=unbatched_list[5],
                return_embeddings=True,
                fpaths=[unbatched_list[4]]
            )
            self.fpaths = [unbatched_list[-2]]
            self.model_in = model_in

    def pin_memory(self):
        if self.model_in is None:
            self.embeddings = self.embeddings.pin_memory()
            self.attentions = self.attentions.pin_memory()
            if self.temp_coords_flag:
                self.temp_coords = self.temp_coords.pin_memory()
                self.temp_masks = self.temp_masks.pin_memory()
        return self

    def to(self, device):
        if self.model_in is None:
            self.embeddings = self.embeddings.to(device)
            self.attentions = self.attentions.to(device)
            if self.temp_coords_flag:
                self.temp_coords = self.temp_coords.to(device)
                self.temp_masks = self.temp_masks.to(device)

    def __len__(self):
        return len(self.fpaths)

    def append_sgls(self, data_ls, data):
        if isinstance(data, list):
            if len(data) == 1:
                data_ls.append(data[0])
            elif len(data) > 1:
                data_ls.append(torch.cat(data, dim=-2))  # 将heavy链和轻链合在一起
        else:
            data_ls.append(data)
        return data_ls


def get_datadf(self_obj, test_flag, valid_flag):
    if (not test_flag) and (not valid_flag):
        data_df = getattr(self_obj, 'posinter_df')
    elif valid_flag and (not test_flag):
        data_df = getattr(self_obj, 'valid_df')
    elif test_flag:
        data_df = getattr(self_obj, 'test_df')
    return data_df


def check_exist(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def gethl_info(pdb_id, df_info, temp_dir):
    info_ = df_info[df_info['pdb'] == pdb_id]
    print(info_)
    if info_.shape[0] > 0:
        info_ = info_.iloc[0, :]
        if not osp.exists(osp.join(temp_dir, '%s.pdb' % pdb_id)):
            spider = Spider()
            spider.sabdab(pdb_id, temp_dir)
        return info_
    else:
        return None


def check_test_pdb():
    # 检查benchmark中的数据是否在训练数据中出现过
    # split_dir = '/data/gm_data/IgFold_data/benchmark/nano/IgFold'
    split_dir = '/data/gm_data/IgFold_data/benchmark/paired/IgFold'
    sabdab_dir = '/data/gm_data/IgFold_data/sabdab'
    test_ids = [obj.split('.')[0] for obj in os.listdir(split_dir) if '.pdb' in obj]
    sabdab_ids = [obj.split('.')[0] for obj in os.listdir(sabdab_dir) if '.pdb' in obj]
    print(set(sabdab_ids).intersection(set(test_ids)))
    print('check end')
    test_dir = '/data/gm_data/IgFold_data/test'
    split = split_dir.split('/')[-2]
    test_dir = osp.join(test_dir, split)
    temp_dir = osp.join(test_dir, 'temp')
    check_exist(test_dir)
    check_exist(temp_dir)
    df_ = pd.read_csv(ddp.sabdab_summtsv, sep='\t')
    df_test = pd.DataFrame()
    for pdb_id in tqdm(test_ids):
        info_i = gethl_info(pdb_id, df_, temp_dir)
        if info_i is not None:
            df_test = df_test.append(info_i)
    print('number of items:', df_test.shape[0])
    df_test.to_csv(osp.join(temp_dir, '%s.csv' % split), sep='\t')
    return test_dir


def get_ref_info(ref_dir):
    pdbfs = [osp.join(ref_dir, f_) for f_ in os.listdir(ref_dir) if f_.endswith('.pdb')]
    pdbfs_info = {}
    for pdbf in pdbfs:
        chains_dict, obj = get_chainsseq(pdbf)
        pdbfs_info[obj] = chains_dict
    return pdbfs_info


def gen_testpdb(pdbf, match_dict):
    # if '7bh8' in pdbf:
    #     print('debug')
    obj_ = 'Ab' if len(match_dict.keys()) > 1 else 'Nb'
    pdbpreer = PDB_pre()
    pm = pm2.PyMOL()
    pm.start()
    pm.cmd.load(pdbf)
    chain_handle = pdbpreer.chls2handle(list(match_dict.values()))
    pm.cmd.select(obj_, chain_handle)
    pm.cmd.create(obj_, obj_)
    pm.cmd.save(pdbf, selection=obj_)
    pm.cmd.remove('all')
    pm.stop()

    pm = pm2.PyMOL()
    pm.start()
    pm.cmd.load(pdbf)
    # 有时原本的H chian id为L
    # if 'H' in match_dict.values() or 'L' in match_dict.values():
    first_key = (return_key(match_dict, 'H') + return_key(match_dict, 'L')).strip(' ')
    ch_orders = list(match_dict.keys()) if len(first_key) == 0 else (list(first_key) + list(set(list(match_dict.keys())) - set(first_key)))
    for ch_i in ch_orders:
        if match_dict[ch_i] != ch_i:
            pm.cmd.alter(f'(chain {match_dict[ch_i]})', f"chain='{ch_i}'")
    pm.cmd.save(pdbf, selection='all')


def gen_testseq(test_dir, ref_info):
    for key_ in ref_info.keys():
        with open(osp.join(test_dir, '%s.fasta' % key_), 'w') as w:
            for ch_ in ref_info[key_].keys():
                w.write('>:{}\n{}\n'.format(ch_, ref_info[key_][ch_]))


def trunc_pdbs(test_dir):
    pdb_preer = PDB_pre()
    print('trunc and renumber pdbs ...')
    for pdb_ in tqdm(sorted([osp.join(test_dir, pdbf) for pdbf in os.listdir(test_dir) if pdbf.endswith('.pdb')])):
        # pdb_ = '/data/gm_data/IgFold_data/test/nano/7xqv.pdb'
        pdb_preer.trunc_pdb(pdb_)
        renumber_pdb(pdb_)
    print('truncate and renumber Ab/Nb pdbfile by chothia rule:\n\t', test_dir)


def excuse_test(test_dir, ref_info):
    initpdb_dir = osp.join(test_dir, 'temp')
    pdb_preer = PDB_pre()
    print('matching......')
    for obj in tqdm(ref_info.keys()):
        # obj = '7xqv'
        info_i = ref_info[obj]
        chain_ls = list(info_i.keys())
        in_f, pdbf = osp.join(initpdb_dir, '{}.pdb'.format(obj)), osp.join(test_dir, '{}.pdb'.format(obj))
        pdb_preer.remove_water(in_f, pdbf)
        chains_dict, obj = get_chainsseq(pdbf)
        match_dict = match_chain(chains_dict, info_i)
        if len(info_i.keys()) != len(match_dict.keys()):
            match_dict = match_chain(chains_dict, info_i, total_match=False)
        assert len(info_i.keys()) == len(match_dict.keys())
        gen_testpdb(pdbf, match_dict)
        gen_testseq(test_dir, ref_info)
    trunc_pdbs(test_dir)
    print('gen test dataset end\t', test_dir)  # 需要对抗体进行截短


def gen_testdb(test_dir=None):
    nano_reftest = '/data/gm_data/IgFold_data/benchmark/nano/IgFold'
    paired_reftest = '/data/gm_data/IgFold_data/benchmark/paired/IgFold'
    nano_test = '/data/gm_data/IgFold_data/test/nano'
    paired_test = '/data/gm_data/IgFold_data/test/paired'
    pairedref_info = get_ref_info(paired_reftest)
    excuse_test(paired_test, pairedref_info)
    nanoref_info = get_ref_info(nano_reftest)
    excuse_test(nano_test, nanoref_info)
    print('')


def get_valids(nfold=5, n_splits=10):
    pdbids = sorted(list(set([f_.split('.')[0] for f_ in os.listdir(ddp.train_dirs[-1])])))
    num = len(pdbids)
    pdbids = list(np.random.choice(pdbids, num, replace=False))
    # random.shuffle(pdbids)
    valid_dict, _bin = {}, math.floor(num / n_splits)
    for n_ in range(nfold):
        if n_ + 1 * _bin < num:
            valid_dict[n_] = pdbids[(n_ * _bin) : (n_ + 1) * _bin]
        else:
            valid_dict[n_] = pdbids[n_ * _bin:]
    return valid_dict


if __name__ == "__main__":
    # test_dir = check_test_pdb()  # download test pdbs
    gen_testdb()  # 创建数据集
