import os
import sys
import os.path as osp
import torch

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)
from spec_At.synergyTraining.preEmbed import pre_embed

from model.interface import DataDirPath
from training.dataset import SamplerDataFrame
from torch.utils.data import DataLoader
from .dataset_utils import Dataset_3d, Dataset_Mutiworker, Dataset_preCoords


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
        self.predcoords_exist = config.predcoords_exist if hasattr(config, 'predcoords_exist') else False
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
            Data_Class = Dataset_preCoords if self.predcoords_exist else Dataset_3d
            self.train_set = Data_Class(data_df=train_val_test_df['train'], preAb_dict=ab_preembeds_dict['train'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
            if not self.train_val_testflg:
                self.test_set = Data_Class(data_df=train_val_test_df['val'], preAb_dict=ab_preembeds_dict['val'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
            else:
                self.valid_set = Data_Class(data_df=train_val_test_df['val'], preAb_dict=ab_preembeds_dict['val'], init_pos_data=init_pos_data, embeder=embeder, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
                self.test_set = Data_Class(data_df=train_val_test_df['test'], preAb_dict=ab_preembeds_dict['test'], init_pos_data=init_pos_data, embeder=embeder, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
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


def load_batch_2device(batch, device):
    batch_attrs = [attr for attr in dir(batch) if not attr.startswith('_')]
    for attr in batch_attrs:
        attr_value = getattr(batch, attr)
        if isinstance(attr_value, torch.Tensor):
            setattr(batch, attr, attr_value.to(device))
    return batch
