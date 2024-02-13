import os
import os.path as osp
import logging
import pandas as pd
import sys
import numpy as np
import torch

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)
from spec_At.data_code.sequence_encoding import returnCKSAAPcode, OneHot_residuetype

from model.interface import DataDirPath
from training.dataset import SamplerDataFrame
from torch.utils.data import Dataset, DataLoader

ddp = DataDirPath()


def split_train_test(pos_df, neg_df, train_set_ratioDict, test_set_PosAndNeg, TrainSetName='A'):
    test_posdf = pos_df.sample(n=test_set_PosAndNeg[0], replace=False, random_state=22)
    pos_df = pos_df.drop(labels=test_posdf.index.to_list())
    test_negdf = neg_df.sample(n=test_set_PosAndNeg[1], replace=False, random_state=22)
    neg_df = neg_df.drop(labels=test_negdf.index.to_list())
    test_df = pd.concat([test_posdf, test_negdf], ignore_index=True)

    train_posdf = pos_df.sample(n=train_set_ratioDict[TrainSetName][0], replace=False, random_state=22)
    train_negdf = neg_df.sample(n=train_set_ratioDict[TrainSetName][1], replace=False, random_state=22)
    val_posratio = train_posdf.shape[0] / train_negdf.shape[0]

    val_posdf = train_posdf.sample(n=test_set_PosAndNeg[0], replace=False, random_state=22)
    val_negdf = train_negdf.sample(n=int(test_set_PosAndNeg[0] / val_posratio), replace=False, random_state=22)
    val_df = pd.concat([val_posdf, val_negdf], ignore_index=True)
    val_df = val_df.sample(frac=1, random_state=22)

    train_posdf = train_posdf.drop(labels=val_posdf.index.to_list())
    train_negdf = train_negdf.drop(labels=val_negdf.index.to_list())
    train_df = pd.concat([train_posdf, train_negdf], ignore_index=True)
    train_val_test_df = {'train': train_df, 'val': val_df, 'test': test_df}
    return train_val_test_df


def modify_dictkey(dict_in, init_key, tgt_key, except_ks=['val']):
    new_dict = {}
    for k_ in dict_in.keys():
        if k_ == init_key:
            new_dict[tgt_key] = dict_in[init_key]
        elif k_ not in except_ks:
            new_dict[k_] = dict_in[k_]
    return new_dict


class GenDataloader(object):
    def __init__(self, config, train_val_test_df, init_pos_data, input_seqtype=None, embed_type=None):

        self.train_val_testflg = True if ('val' in train_val_test_df.keys()) and ('test' in train_val_test_df.keys()) else False
        if isinstance(config.id_fold, int) and (not self.train_val_testflg):
            self.loader_type = 'Nfold'
            train_val_test_df = modify_dictkey(train_val_test_df, 'val', 'test')
        else:
            self.loader_type = 'TrainValTest'

        self.ddp = DataDirPath()
        self.test_dir = [osp.join(self.ddp.test_dirs, d_) for d_ in os.listdir(self.ddp.test_dirs)]
        self.split_dir = self.ddp.train_dirs
        self.batch_size = config.batch_size
        self.num_data_workers = config.num_data_workers
        self.trainval_negs = config.trainval_negs
        self.embed_type = embed_type
        self.input_seqtype = input_seqtype

        self._init_datasets(config, train_val_test_df, init_pos_data)
        self._init_samplers()
        self._init_loaders()

    def _init_datasets(self, config, train_val_test_df=None, init_pos_data=None):

        if self.input_seqtype is 'OnlyMut':
            self.train_set = Dataset_SeqMutant(data_df=train_val_test_df['train'], embed_type=self.embed_type, init_pos_data=init_pos_data, input_seqtype=self.input_seqtype, test_flag=False, device=config.device, out_dir=config.out_dir)
            self.test_set = Dataset_SeqMutant(data_df=train_val_test_df['test'], embed_type=self.embed_type, init_pos_data=init_pos_data, input_seqtype=self.input_seqtype, test_flag=True, device=config.device, out_dir=config.out_dir)
            if (self.loader_type is not 'Nfold') or self.train_val_testflg:
                self.valid_set = Dataset_SeqMutant(data_df=train_val_test_df['val'], embed_type=self.embed_type, init_pos_data=init_pos_data, input_seqtype=self.input_seqtype, test_flag=False, valid_flag=True, device=config.device, out_dir=config.out_dir)

        else:
            self.train_set = Dataset_AbAtSeq(data_df=train_val_test_df['train'], embed_type=self.embed_type, init_pos_data=init_pos_data, test_flag=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
            self.test_set = Dataset_AbAtSeq(data_df=train_val_test_df['test'], embed_type=self.embed_type, init_pos_data=init_pos_data, test_flag=True, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
            if (self.loader_type is not 'Nfold') or self.train_val_testflg:
                self.valid_set = Dataset_AbAtSeq(data_df=train_val_test_df['val'], embed_type=self.embed_type, init_pos_data=init_pos_data, test_flag=False, trainval_negs=self.trainval_negs, valid_flag=True, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
        config.is_master = True

    def _init_samplers(self):
        self.train_sampler = SamplerDataFrame(self.train_set, partion='train')
        if (self.loader_type is not 'Nfold') or self.train_val_testflg:
            self.valid_sampler = SamplerDataFrame(self.valid_set, partion='valid')
        self.test_sampler = SamplerDataFrame(self.test_set, partion='test')

    def _init_loaders(self):
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=self.num_data_workers,
                                       pin_memory=torch.cuda.is_available(),)
        if (self.loader_type is not 'Nfold') or self.train_val_testflg:
            self.valid_loader = DataLoader(self.valid_set,
                                        batch_size=self.batch_size,
                                        sampler=self.valid_sampler,
                                        num_workers=self.num_data_workers,
                                        pin_memory=torch.cuda.is_available())
        self.test_loader = DataLoader(self.test_set,
                                      batch_size=self.batch_size,
                                      sampler=self.test_sampler,
                                      num_workers=self.num_data_workers,
                                      pin_memory=torch.cuda.is_available())


class Dataset_AbAtSeq(Dataset):
    def __init__(self, data_df=None, embed_type='CKSAAP', init_pos_data=None, test_flag=False, valid_flag=False, trainval_negs=5, device='cpu', ddp=None, out_dir=None):
        self.ddp, self.out_dir, self.trainval_negs = ddp, out_dir, trainval_negs
        self.test_flag, self.valid_flag, self.device, = test_flag, valid_flag, device
        self.embed_type = embed_type

        # sabdab_interdf中是包含所有数据信息的
        self.data_df = data_df
        # self.logger = open(osp.join(self.out_dir, 'data_fpath.log'), 'a', encoding='utf-8')
        self.debug_iteri = 0
        self.init_pos_data = init_pos_data

    def __getitem__(self, idx):
        self.debug_iteri += 1
        data_si = self.data_df.loc[idx]
        if 'Hseq' not in data_si.index:
            aaseq_dict = get_aaseqDict_BySeries(data_si)
            ab_seq = ''.join(seq for seq in aaseq_dict.values())
        else:
            ab_seq = ''.join(data_si[k_] for k_ in ['Hseq', 'Lseq'] if isinstance(data_si[k_], str))
        pos_index = int(data_si['pos_dataid']) if 'pos_dataid' in data_si.index else None
        if 'Atseq' in data_si.index:  pos_index = None
        at_seq = data_si['Atseq'] if pos_index is None else self.init_pos_data['At_data'][pos_index]['atseq']
        if self.embed_type is 'CKSAAP':
            ab_data = returnCKSAAPcode(ab_seq, 3)
            at_data = returnCKSAAPcode(at_seq, 3)
            ab_data = torch.tensor(data=ab_data, dtype=torch.float32).view(-1, 20, 20)
            at_data = torch.tensor(data=at_data, dtype=torch.float32).view(-1, 20, 20)
        elif self.embed_type is 'OneHot':
            ab_data = OneHot_residuetype(ab_seq)
            at_data = OneHot_residuetype(at_seq)
        inter_label = int(data_si['AgClass'])
        inter_label = torch.tensor(data=inter_label, dtype=torch.long)
        return ab_data, at_data, inter_label


class Dataset_SeqMutant(Dataset):
    def __init__(self, data_df=None, embed_type='OneHot', input_seqtype='OnlyMut', init_pos_data=None, test_flag=False, valid_flag=False, device='cpu', out_dir=None):
        self.out_dir = out_dir
        self.test_flag, self.valid_flag, self.device, = test_flag, valid_flag, device
        self.embed_type = embed_type
        self.input_seqtype = input_seqtype

        # sabdab_interdf中是包含所有数据信息的
        self.data_df = data_df
        self.logger = open(osp.join(self.out_dir, 'data_fpath.log'), 'a', encoding='utf-8')
        self.debug_iteri = 0
        self.init_pos_data = init_pos_data

    def __getitem__(self, idx):
        self.debug_iteri += 1
        data_si = self.data_df.loc[idx]
        inter_label = int(data_si['AgClass'])
        inter_label = torch.tensor(data=inter_label, dtype=torch.long)
        if self.input_seqtype is 'OnlyMut':
            ab_seq = data_si['AASeq']
            if self.embed_type is 'CKSAAP':
                ab_data = returnCKSAAPcode(ab_seq, 3)
            elif self.embed_type is 'OneHot':
                ab_data = OneHot_residuetype(ab_seq)
            return ab_data, inter_label
        else:
            aaseq_dict = get_aaseqDict_BySeries(data_si)
            ab_seq = ''.join(seq for seq in aaseq_dict.values())
            if self.embed_type is 'CKSAAP':
                ab_data = returnCKSAAPcode(ab_seq, 3)
                at_data = returnCKSAAPcode(data_si['Atseq'], 3)
                ab_data = torch.tensor(data=ab_data, dtype=torch.float32).view(-1, 20, 20)
                at_data = torch.tensor(data=at_data, dtype=torch.float32).view(-1, 20, 20)
            elif self.embed_type is 'OneHot':
                ab_data = OneHot_residuetype(ab_seq)
                at_data = OneHot_residuetype(at_data)
            return ab_data, at_data, inter_label


def get_aaseqDict_BySeries(dfi, hseq_name='Hseq', lseq_name='Lseq'):
    aaseq_dict = {}
    if hseq_name in dfi:
        if isinstance(dfi[hseq_name], str):  aaseq_dict['H'] = dfi[hseq_name]
    if lseq_name in dfi:
        if isinstance(dfi[lseq_name], str):  aaseq_dict['L'] = dfi[lseq_name]
    return aaseq_dict


class Batch_SeqInterData(object):
    def __init__(self, unbatched_list):
        self.len_unbatchedLs = len(unbatched_list)
        ab_embed, at_embed, label = unbatched_list
        self.model_in = unbatched_list[0]

    def __len__(self):
        return self.len_unbatchedLs


class Batch_InterData(object):
    def __init__(self, unbatched_list):
        self.len_unbatchedLs = len(unbatched_list)
        self.model_in = unbatched_list[0]

    def __len__(self):
        return self.len_unbatchedLs


def get_datadf(self_obj, test_flag, valid_flag):
    if (not test_flag) and (not valid_flag):
        data_df = getattr(self_obj, 'posinter_df')
    elif valid_flag and (not test_flag):
        data_df = getattr(self_obj, 'valid_df')
    elif test_flag:
        data_df = getattr(self_obj, 'test_df')
    return data_df
