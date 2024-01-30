import os
import sys
import os.path as osp
import torch
from copy import deepcopy

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)
from SabDab.get_SabDabData import process_Abdata
from spec_At.synergyTraining.preEmbed import pre_embed
from abatInter_SCA.spec_At.data_code.Embedding import get_singledata
from SabDab.data_code.batch_utils import get_batchdata

from model.interface import DataDirPath
from training.dataset import SamplerDataFrame
from torch.utils.data import Dataset, DataLoader

ddp = DataDirPath()


class GenDataloader(object):
    def __init__(self, config, train_val_test_df, init_pos_data, embeder):
        # self.test_flag = split == 'test'
        self.ddp = DataDirPath()
        self.batch_size = config.batch_size
        self.train_val_testflg = True if ('val' in train_val_test_df.keys()) and ('test' in train_val_test_df.keys()) else False
        self.num_data_workers = config.num_data_workers
        self.trainval_negs = config.trainval_negs
        self.speed_up = config.speed_up if hasattr(config, 'speed_up') else False
        ab_preembeds_dict = {key: None for key in train_val_test_df.keys()}
        if self.speed_up:
            params = {}
            params['init_pos_data'] = init_pos_data
            params['embeder'] = embeder
            params['template_flg'] = False
            ab_preembeds_dict = pre_embed(train_val_test_df, params=params, abtype='not_mutante')

        self._init_datasets(config, train_val_test_df, ab_preembeds_dict, init_pos_data, embeder)
        self._init_samplers()
        self._init_loaders()

    def _init_datasets(self, config, train_val_test_df=None, ab_preembeds_dict=None, init_pos_data=None, embeder=None):
        if self.num_data_workers < 1:
            self.train_set = Dataset_3d(data_df=train_val_test_df['train'], preAb_dict=ab_preembeds_dict['train'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
            if not self.train_val_testflg:
                self.test_set = Dataset_3d(data_df=train_val_test_df['val'], preAb_dict=ab_preembeds_dict['val'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
            else:
                self.valid_set = Dataset_3d(data_df=train_val_test_df['val'], preAb_dict=ab_preembeds_dict['val'], init_pos_data=init_pos_data, embeder=embeder, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
                self.test_set = Dataset_3d(data_df=train_val_test_df['test'], preAb_dict=ab_preembeds_dict['test'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
        else:
            self.train_set = Dataset_Mutiworker(data_df=train_val_test_df['train'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, device=config.device, out_dir=config.out_dir)
            if not self.train_val_testflg:
                self.test_set = Dataset_Mutiworker(data_df=train_val_test_df['val'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, device=config.device, out_dir=config.out_dir)
            else:
                self.valid_set = Dataset_Mutiworker(data_df=train_val_test_df['val'], init_pos_data=init_pos_data, embeder=embeder, device=config.device, out_dir=config.out_dir)
                self.test_set = Dataset_Mutiworker(data_df=train_val_test_df['test'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, device=config.device, out_dir=config.out_dir)
        config.is_master = True

    def _init_samplers(self):
        self.train_sampler = SamplerDataFrame(self.train_set, partion='train')
        if self.train_val_testflg:  self.valid_sampler = SamplerDataFrame(self.valid_set, partion='valid')
        self.test_sampler = SamplerDataFrame(self.test_set, partion='test')

    def _init_loaders(self):
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=self.num_data_workers,
                                       collate_fn=self.train_set.collate_wrapper,
                                       pin_memory=torch.cuda.is_available(),)
        if self.train_val_testflg:
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
    def __init__(self, data_df=None, preAb_dict=None, init_pos_data=None, embeder=None, ab_preembeds_dict=None, template_flg=True, device='cpu', trainval_negs=None, ddp=None, out_dir=None, atbox_truc=15.0):
        self.ddp, self.out_dir = ddp, out_dir
        self.template_flg = template_flg
        self.device, self.atbox_truc = device, atbox_truc

        # sabdab_interdf中是包含所有数据信息的
        self.data_df = data_df
        self.ab_preembeds_dict = ab_preembeds_dict
        self.logger = open(osp.join(self.out_dir, 'data_fpath.log'), 'a', encoding='utf-8')
        self.debug_iteri = 0
        self.init_pos_data, self.embeder = init_pos_data, embeder
        self.preAb_dict = preAb_dict  # 预先预测得到的Ab_dict

    def __getitem__(self, idx):
        self.debug_iteri += 1
        data_si = self.data_df.loc[idx]
        pos_index = int(data_si['pos_dataid'])
        Ab_abatid, At_abatid = data_si['sudo_ab'], data_si['sudo_at']

        Ab_data = self.init_pos_data[Ab_abatid]['Ab_data']
        Ab_data = process_Abdata(Ab_data, db_args=self.embeder.database_args, tmp_flg=self.template_flg, mutate_range=None)
        At_data = self.init_pos_data[At_abatid]['At_data']
        # Ab_phisicEmbed Ab_phisicAttention
        At_embeddings = At_data['At_embeddings'][0]
        At_attentions = At_data['chi_attentionsLs'][0]
        At_coords_label = At_data['atbox_coords'][0]
        At_len = At_embeddings.size(0)
        At_data = {'atbox_coords': At_coords_label, 'At_embeddings': At_embeddings, 'chi_attentionsLs': At_attentions, 'At_len': At_len}

        inter_label = int(data_si['AgClass'])
        return Ab_data, At_data, inter_label, pos_index, self.embeder.embed_type, self.device

    @staticmethod
    def collate_wrapper(unbatched_list):
        return Batch_InterData(unbatched_list)


def construct_Abdata_list(data_df, init_pos_data, embeder_args, tmp_flg=None, mutate_range=None):
    excused_Ab_abatidls = []
    Ab_data_ls, dataindex2_abindex = [], []
    label_ls, abatid_names = [], []
    print('Dataset_Mutiworker init --> construct_Abdata_list   ing ...')
    for i_ in data_df.index.to_list():
        # if i_ > 100: break  # debug
        data_si = data_df.loc[i_]
        pos_index = int(data_si['pos_dataid'])
        Ab_abatid = data_si['sudo_ab']
        Ab_data = init_pos_data[Ab_abatid]['Ab_data']
        Ab_data = process_Abdata(Ab_data, db_args=embeder_args, tmp_flg=tmp_flg, mutate_range=mutate_range, map_cpu=True)  # 将antibert特征融入进来
        if Ab_abatid not in excused_Ab_abatidls:
            excused_Ab_abatidls.append(Ab_abatid)
            Ab_data_ls.append(Ab_data)
        dataindex2_abindex.append(excused_Ab_abatidls.index(Ab_abatid))
        label_ls.append(data_si['AgClass'])
        abatid_names.append(Ab_abatid)

    return Ab_data_ls, excused_Ab_abatidls, dataindex2_abindex, label_ls, abatid_names


def construct_Atdata_list(data_df, init_pos_data):
    excused_At_abatidls = []
    At_data_ls, dataindex2_atindex = [], []
    label_ls, pos_index_ls = [], []
    print('Dataset_Mutiworker init --> construct_Atdata_list   ing ...')
    for i_ in data_df.index.to_list():
        # if i_ > 100: break  # debug
        data_si = data_df.loc[i_]
        pos_index = int(data_si['pos_dataid'])
        At_abatid = data_si['sudo_at']
        At_data = init_pos_data[At_abatid]['At_data']

        At_embeddings = At_data['At_embeddings'][0]
        At_attentions = At_data['chi_attentionsLs'][0]
        At_coords_label = At_data['atbox_coords'][0]
        At_len = At_embeddings.size(0)
        At_data = {'atbox_coords': At_coords_label, 'At_embeddings': At_embeddings, 'chi_attentionsLs': At_attentions, 'At_len': At_len}

        if At_abatid not in excused_At_abatidls:
            excused_At_abatidls.append(At_abatid)
            At_data_ls.append(At_data)
        dataindex2_atindex.append(excused_At_abatidls.index(At_abatid))
        label_ls.append(data_si['AgClass'])
        pos_index_ls.append(pos_index)
    return At_data_ls, excused_At_abatidls, dataindex2_atindex, label_ls, pos_index_ls


class Dataset_Mutiworker(Dataset):
    def __init__(self, data_df=None, init_pos_data=None, embeder=None, template_flg=True, device='cpu', out_dir=None, atbox_truc=15.0):
        self.template_flg = template_flg
        self.device, self.atbox_truc = device, atbox_truc

        # sabdab_interdf中是包含所有数据信息的
        data_df = data_df.reset_index(drop=True)  # sampler中的index是由data_df的index产生的, index是一个range(0, l)形式的索引
        self.Abdata_ls, excused_Ab_abatidls, self.dataindex2_abindex, ab_label_ls, self.Ab_abatid_names = construct_Abdata_list(data_df, init_pos_data, embeder.database_args, tmp_flg=self.template_flg, mutate_range=None)
        self.Atdata_ls, excused_At_abatidls, self.dataindex2_atindex, at_label_ls, pos_index_ls = construct_Atdata_list(data_df, init_pos_data)
        self.df = data_df

        assert ab_label_ls == at_label_ls
        self.label_ls = ab_label_ls
        self.pos_index_ls = pos_index_ls
        self.embed_type_ls = [embeder.embed_type] * len(ab_label_ls)
        self.device_info_ls = [device] * len(ab_label_ls)
        # self.logger = open(osp.join(self.out_dir, 'data_fpath.log'), 'a', encoding='utf-8')

    def __getitem__(self, idx):
        # idx = 50  # debug用
        Abdata_index = self.dataindex2_abindex[idx]
        Atdata_index = self.dataindex2_atindex[idx]

        Ab_data = self.Abdata_ls[Abdata_index]
        At_data = self.Atdata_ls[Atdata_index]

        inter_label = self.label_ls[idx]
        pos_index = self.pos_index_ls[idx]
        embed_type = self.embed_type_ls[idx]
        Ab_abatid = self.Ab_abatid_names[idx]

        # return Ab_data, At_data, inter_label, pos_index, embed_type, device_info
        return Ab_data, At_data, inter_label, pos_index, embed_type, Ab_abatid

    @staticmethod
    def collate_wrapper(unbatched_list):
        return Batch_InterData(unbatched_list)


class Dataset_preCoords(Dataset):
    def __init__(self, data_df=None, preAb_dict=None, init_pos_data=None, embeder=None, ab_preembeds_dict=None, template_flg=True, device='cpu', trainval_negs=None, ddp=None, out_dir=None, atbox_truc=15.0):
        self.ddp, self.out_dir = ddp, out_dir
        self.template_flg = template_flg
        self.device, self.atbox_truc = device, atbox_truc

        # sabdab_interdf中是包含所有数据信息的
        self.data_df = data_df
        self.ab_preembeds_dict = ab_preembeds_dict
        self.logger = open(osp.join(self.out_dir, 'data_fpath.log'), 'a', encoding='utf-8')
        self.debug_iteri = 0
        self.init_pos_data, self.embeder = init_pos_data, embeder
        self.preAb_dict = preAb_dict  # 预先预测得到的Ab_dict

    def __getitem__(self, idx):
        self.debug_iteri += 1
        data_si = self.data_df.loc[idx]
        pos_index = int(data_si['pos_dataid'])
        Ab_abatid, At_abatid = data_si['sudo_ab'], data_si['sudo_at']  # 在获取负样本数据的时候错误, sudo_at的id并不是真实的id

        Ab_data = self.init_pos_data[Ab_abatid]['Ab_data']
        # assert compare_h_l_values(data_si, Ab_data['seq_dict'])
        # Ab_data = process_Abdata(Ab_data, db_args=self.embeder.database_args, tmp_flg=self.template_flg, mutate_range=None)
        assert 'pred_coords' in Ab_data.keys()

        At_data = self.init_pos_data[At_abatid]['At_data']
        assert At_data['atseq'] == data_si['Atseq']
        # Ab_phisicEmbed Ab_phisicAttention
        At_embeddings = At_data['At_embeddings'][0]
        At_attentions = At_data['chi_attentionsLs'][0]
        At_coords_label = At_data['atbox_coords'][0]
        At_len = At_embeddings.size(0)
        At_data = {'atbox_coords': At_coords_label, 'At_embeddings': At_embeddings, 'chi_attentionsLs': At_attentions, 'At_len': At_len}

        inter_label = int(data_si['AgClass'])
        return Ab_data, At_data, inter_label, pos_index, self.embeder.embed_type, self.device


class Batch_InterData(object):
    def __init__(self, unbatched_list):
        self.len_unbatchedLs = len(unbatched_list)
        if self.len_unbatchedLs == 1 and len(unbatched_list[0]) == 5:
            Ab_data, At_data, inter_label, embed_type, self.device = unbatched_list[0]
            self.model_in = get_singledata(Ab_data, At_data, inter_label, embed_type=embed_type, device_in=self.device)
        else:
            Ab_data, At_data, inter_label, pos_index, embed_type, Ab_abatid = list(zip(*unbatched_list))
            # print('abatid ls is:', Ab_abatid)
            self.model_in = get_batchdata(Ab_data, At_data, inter_label, pos_index=pos_index, embed_type=embed_type, batch_size=self.len_unbatchedLs, Ab_abatid=Ab_abatid)

    def __len__(self):
        return self.len_unbatchedLs


def load_batch_2device(batch, device):
    batch_attrs = [attr for attr in dir(batch) if not attr.startswith('_')]
    for attr in batch_attrs:
        attr_value = getattr(batch, attr)
        if isinstance(attr_value, torch.Tensor):
            setattr(batch, attr, attr_value.to(device))
    return batch


# if __name__ == "__main__":
#     # test_dir = check_test_pdb()  # download test pdbs
