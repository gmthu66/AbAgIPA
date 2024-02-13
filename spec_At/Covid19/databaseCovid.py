import os
import sys
import os.path as osp
import pandas as pd
import torch
from sklearn.model_selection import KFold

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)
from spec_At.Covid19.get_Covid_data import process_Abdata
from spec_At.synergyTraining.preEmbed import pre_embed
from abatInter_SCA.spec_At.data_code.Embedding import get_singledata, get_batchdata
from spec_At.utils_df import string_to_dict
from spec_At.data_code.get_data_utils import CoordsPreder
from model.interface import DataDirPath
from training.dataset import SamplerDataFrame
from torch.utils.data import Dataset, DataLoader
from spec_At.utils_df import remove_unnamed_columns
from SabDab.databaseSabDab import Batch_InterData as Batch_InterDataWithPredCoords
from SabDab.data_code.dataset_utils import Batch_InterData as Batch_PrecoordData
from tqdm import tqdm

ddp = DataDirPath()
# preAb_dir = osp.join(osp.dirname(__file__), 'Covid19database/Abs')
preAb_dir = osp.join('/home/user/protein/IgFold_data', 'Covid19database/Abs')


def get_nsplit_df(df, nfold):
    # 获取DataFrame的总行数和列数
    num_rows, num_cols = df.shape
    # 计算每个子DataFrame的行数
    rows_per_split = num_rows // nfold
    remaining_rows = num_rows % nfold
    # 初始化字典用于存储切分后的子DataFrame
    split_dict = {}
    # 切分DataFrame并存储到字典中
    start_row = 0
    for i in range(nfold):
        end_row = start_row + rows_per_split + (1 if i < remaining_rows else 0)
        split_dict[i] = df.iloc[start_row:end_row]
        start_row = end_row
    return split_dict


def kfold_split(df, num_folds=5, random_state=42):
    """
    将DataFrame划分为五折交叉验证的训练集和验证集, 并返回一个字典。
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    fold_dict = {}

    for fold, (train_index, val_index) in enumerate(kf.split(df)):
        train_set = df.iloc[train_index]
        val_set = df.iloc[val_index]
        fold_dict[fold] = {'train': train_set, 'val': val_set}

    return fold_dict


def stack_dataframes(dict_df1, dict_df2):

    stacked_dict = {}
    # 遍历两个字典的键值对
    for key in dict_df1.keys():
        # 使用 concat 函数进行堆叠
        stacked_df = pd.concat([dict_df1[key], dict_df2[key]], axis=0)
        stacked_dict[key] = stacked_df
    return stacked_dict


def update_nfold_dict(data_dict, sub_dfi):
    for i_ in data_dict.keys():
        for k_ in data_dict[i_]:
            if data_dict[i_][k_] is None:
                data_dict[i_][k_] = sub_dfi[i_][k_]
            else:
                data_dict[i_][k_] = pd.concat([data_dict[i_][k_], sub_dfi[i_][k_]], axis=0)
    print('update_nfold_dict')


def get_nfold_datadf(pos_df, neg_df, nfold=5):
    at_namels = sorted(list(set(pos_df['At_name'].to_list())))
    pos_df = remove_unnamed_columns(pos_df)
    neg_df = remove_unnamed_columns(neg_df)
    pos_df = pos_df.sample(frac=1, random_state=22).reset_index(drop=True)
    neg_df = neg_df.sample(frac=1, random_state=22).reset_index(drop=True)
    data_dict, nfold_df_ls = {}, []
    for i_ in range(nfold):
        data_dict[i_] = {'train': None, 'val': None}
    for ati in at_namels:
        pos_df_ati = pos_df[pos_df['At_name'] == ati]
        neg_df_ati = neg_df[neg_df['At_name'] == ati]
        print('{} pos sample is: {}'.format(ati, pos_df_ati.shape[0]))
        print('{} neg sample is: {}'.format(ati, neg_df_ati.shape[0]))
        nfold_pos_df_ati = kfold_split(pos_df_ati, num_folds=nfold)
        update_nfold_dict(data_dict, nfold_pos_df_ati)
        nfold_neg_df_ati = kfold_split(neg_df_ati, num_folds=nfold)
        update_nfold_dict(data_dict, nfold_neg_df_ati)
    return data_dict


def save_nfold_df2csv(train_val_nfold_df, csv_dir, atname=None):
    os.makedirs(csv_dir, exist_ok=True)
    for foldi in train_val_nfold_df.keys():
        train_set = train_val_nfold_df[foldi]['train']
        val_set = train_val_nfold_df[foldi]['val']
        # 保存训练集和验证集为CSV文件
        if atname is None:
            train_set.to_csv(f'{csv_dir}/fold_{foldi + 1}_train.csv', index=False)
            val_set.to_csv(f'{csv_dir}/fold_{foldi + 1}_val.csv', index=False)
        else:
            train_set.to_csv(f'{csv_dir}/{atname}_fold_{foldi + 1}_train.csv', index=False)
            val_set.to_csv(f'{csv_dir}/{atname}_fold_{foldi + 1}_val.csv', index=False)


def try_read_train_val(csv_dir, foldi, atname=None):

    train_set = f'{csv_dir}/fold_{foldi + 1}_train.csv' if atname is None else f'{csv_dir}/{atname}_fold_{foldi + 1}_train.csv'
    val_set = f'{csv_dir}/fold_{foldi + 1}_val.csv' if atname is None else f'{csv_dir}/{atname}_fold_{foldi + 1}_val.csv'
    if osp.exists(train_set) and osp.exists(val_set):
        train_set = pd.read_csv(train_set, index_col=False)
        val_set = pd.read_csv(val_set, index_col=False)
        data_dict = {'train': train_set, 'val': val_set}
    else:
        data_dict = None
    return data_dict


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


class GenDataloader(object):
    def __init__(self, config, train_val_test_df, init_pos_data, embeder):
        # self.test_flag = split == 'test'
        self.ddp = DataDirPath()
        self.batch_size = config.batch_size
        self.train_val_testflg = True if ('val' in train_val_test_df.keys()) and ('test' in train_val_test_df.keys()) else False

        self.loadInMem = getattr(config, 'loadInMem') if hasattr(config, 'loadInMem') else False
        self.num_data_workers = config.num_data_workers
        self.trainval_negs = config.trainval_negs
        self.predcoords_exist = config.predcoords_exist if hasattr(config, 'predcoords_exist') else False
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
            Data_Class = Dataset_preCoords if self.predcoords_exist else Dataset_3d
            self.train_set = Data_Class(data_df=train_val_test_df['train'], preAb_dict=ab_preembeds_dict['train'], init_pos_data=init_pos_data, embeder=embeder, test_flag=False, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
            if not self.train_val_testflg:
                self.test_set = Data_Class(data_df=train_val_test_df['val'], preAb_dict=ab_preembeds_dict['val'], init_pos_data=init_pos_data, embeder=embeder, test_flag=True, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
            else:
                self.valid_set = Data_Class(data_df=train_val_test_df['val'], preAb_dict=ab_preembeds_dict['val'], init_pos_data=init_pos_data, embeder=embeder, test_flag=False, trainval_negs=self.trainval_negs, valid_flag=True, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
                self.test_set = Data_Class(data_df=train_val_test_df['test'], preAb_dict=ab_preembeds_dict['test'], init_pos_data=init_pos_data, embeder=embeder, test_flag=False, template_flg=False, trainval_negs=self.trainval_negs, device=config.device, ddp=self.ddp, out_dir=config.out_dir)
        else:
            self.train_set = Dataset_Mutiworker(data_df=train_val_test_df['train'], init_pos_data=init_pos_data, loadInMem=self.loadInMem, embeder=embeder, template_flg=False, device=config.device)
            if not self.train_val_testflg:
                self.test_set = Dataset_Mutiworker(data_df=train_val_test_df['val'], init_pos_data=init_pos_data, embeder=embeder, loadInMem=self.loadInMem, template_flg=False, device=config.device)
            else:
                self.valid_set = Dataset_Mutiworker(data_df=train_val_test_df['val'], init_pos_data=init_pos_data, embeder=embeder, loadInMem=self.loadInMem, device=config.device)
                self.test_set = Dataset_Mutiworker(data_df=train_val_test_df['test'], init_pos_data=init_pos_data, embeder=embeder, loadInMem=self.loadInMem, template_flg=False, device=config.device)
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


class Dataset_preCoords(Dataset):
    def __init__(self, data_df=None, preAb_dict=None, init_pos_data=None, embeder=None, ab_preembeds_dict=None, template_flg=True, device='cpu', trainval_negs=None, test_flag=False, ddp=None, out_dir=None, atbox_truc=15.0):
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
        Ab_abatid = data_si['ab_dataid']
        At_abatid = data_si['At_name']  # 在获取负样本数据的时候错误, sudo_at的id并不是真实的id

        Ab_data = torch.load(osp.join(preAb_dir, f'preAb_{Ab_abatid}.pt'))
        Ab_data['seq_dict'] = get_aaseqDict_BySeries(data_si)
        # assert compare_h_l_values(data_si, Ab_data['seq_dict'])
        # Ab_data = process_Abdata(Ab_data, db_args=self.embeder.database_args, tmp_flg=self.template_flg, mutate_range=None)
        assert 'pred_coords' in Ab_data.keys()

        At_data = self.init_pos_data['At_data'][pos_index]
        if 'Atseq' in data_si.index:  assert At_data['atseq'] == data_si['Atseq']
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
        return Batch_PrecoordData(unbatched_list)


def construct_Abdata_list(data_df, init_pos_data, embeder_args, tmp_flg=None, mutate_range=None, preAb_dir=None, loadInMem=False):

    Ab_data_ls, dataindex2_abindex = [], []
    label_ls, pos_index_ls = [], []
    excused_Ab_dataids, abatid_names = [], []
    dataindex2_abindex = [-1 for i_ in range(data_df.shape[0])]
    aaseq_dict_ls = []
    print('Dataset_Mutiworker init --> construct_Abdata_list   ing ...')
    coords_predictor = CoordsPreder()
    for i_ in tqdm(data_df.index.to_list()):
        data_si = data_df.loc[i_]
        ab_dataid = data_si['ab_dataid']
        pos_index = int(data_si['pos_dataid'])
        aaseq_dict = get_aaseqDict_BySeries(data_si)
        mutate_range = eval(data_si['MutRange']) if 'MutRange' in data_si.index else None

        if ab_dataid not in excused_Ab_dataids:
            Ab_data, preAb_path = None, None
            if preAb_dir is not None:
                preAb_path = osp.join(preAb_dir, f'preAb_{ab_dataid}.pt')
                if osp.exists(preAb_path):
                    if not loadInMem:
                        # Ab_data = torch.load(preAb_path, map_location='cpu')
                        # if Ab_data['pred_coords'].is_leaf:
                        #     Ab_data['pred_coords'] = Ab_data['pred_coords'].detach()
                        #     torch.save(Ab_data, preAb_path)
                        Ab_data = preAb_path
                    else:
                        Ab_data = torch.load(preAb_path, map_location='cpu')
                    Ab_data_ls.append(Ab_data)
            if Ab_data is None:
                cdr_fr_info = string_to_dict(data_si['cdr_fr_info'])
                Ab_data = process_Abdata(aaseq_dict, wild_Abdata_dict=init_pos_data['Ab_data'], pos_index=pos_index, rec_cdrfr=cdr_fr_info, db_args=embeder_args, tmp_flg=tmp_flg, mutate_range=mutate_range, map_cpu=True)
                if 'pred_coords' not in Ab_data.keys():
                    Ab_data = coords_predictor.addPredCoords(Ab_data)
                if preAb_path is not None:
                    torch.save(Ab_data, preAb_path)
                    Ab_data_ls.append(preAb_path)
                else:
                    Ab_data_ls.append(Ab_data)
            if ab_dataid not in excused_Ab_dataids:  excused_Ab_dataids.append(ab_dataid)
            aaseq_dict_ls.append(aaseq_dict)

        dataindex2_abindex[i_] = excused_Ab_dataids.index(ab_dataid)
        inter_label = int(data_si['AgClass'])
        label_ls.append(inter_label)
        pos_index_ls.append(pos_index)
        abatid_names.append(ab_dataid)

    return Ab_data_ls, dataindex2_abindex, label_ls, pos_index_ls, abatid_names, aaseq_dict_ls


def construct_Atdata_list(data_df, init_pos_data):
    # At_data_ls = []
    label_ls, pos_index_ls = [], []
    excused_Atpos_index = []
    print('Dataset_Mutiworker init --> construct_Atdata_list   ing ...')
    At_data_ls = [None for i_ in range(int(data_df['pos_dataid'].max()) + 1)]
    dataindex2_atindex = [-1 for i_ in range(data_df.shape[0])]
    for i_ in data_df.index.to_list():
        data_si = data_df.loc[i_]
        At_pos_index = int(data_si['pos_dataid'])
        inter_label = int(data_si['AgClass'])
        At_data = init_pos_data['At_data'][At_pos_index]
        if At_pos_index not in excused_Atpos_index:
            At_data_ls[At_pos_index] = At_data
            excused_Atpos_index.append(At_pos_index)

        pos_index_ls.append(At_pos_index)
        label_ls.append(inter_label)
        dataindex2_atindex[i_] = At_pos_index
    assert -1 not in dataindex2_atindex
    return At_data_ls, dataindex2_atindex, label_ls, pos_index_ls


def unfold_datals(Atdata_ls):
    for i_ in range(len(Atdata_ls)):
        data = Atdata_ls[i_]
        if data is not None:
            for k_ in data.keys():
                if isinstance(data[k_], list):  data[k_] = data[k_][0]
                if isinstance(data[k_], torch.Tensor):  # detach pred_coords
                    if data[k_].is_leaf:  data[k_] = data[k_].detach()
        Atdata_ls[i_] = data
    return Atdata_ls


class Dataset_Mutiworker(Dataset):
    def __init__(self, data_df=None, init_pos_data=None, embeder=None, template_flg=True, loadInMem=False, device='cpu', atbox_truc=15.0):
        self.template_flg = template_flg
        self.device, self.atbox_truc = device, atbox_truc
        os.makedirs(preAb_dir, exist_ok=True)

        # sabdab_interdf中是包含所有数据信息的
        data_df = data_df.reset_index(drop=True)  # sampler中的index是由data_df的index产生的, index是一个range(0, l)形式的索引
        self.Abdata_ls, self.dataindex2_abindex, ab_label_ls, Ab_pos_indexls, Ab_idls, self.aaseq_dict_ls = construct_Abdata_list(data_df, init_pos_data, embeder.database_args, tmp_flg=self.template_flg, mutate_range=None, preAb_dir=preAb_dir, loadInMem=loadInMem)
        Atdata_ls, self.dataindex2_atindex, at_label_ls, At_pos_indexls = construct_Atdata_list(data_df, init_pos_data)
        self.Atdata_ls = unfold_datals(Atdata_ls)
        self.data_df = data_df

        assert ab_label_ls == at_label_ls
        assert Ab_pos_indexls == At_pos_indexls
        self.label_ls = ab_label_ls
        self.pos_index_ls = Ab_pos_indexls
        self.embed_type_ls = [embeder.embed_type] * len(ab_label_ls)
        self.device_info_ls = [device] * len(ab_label_ls)

    def __getitem__(self, idx):
        Abdata_index = self.dataindex2_abindex[idx]
        Atdata_index = self.dataindex2_atindex[idx]

        Ab_data = self.Abdata_ls[Abdata_index]
        Ab_data = torch.load(Ab_data, map_location='cpu') if isinstance(Ab_data, str) else Ab_data
        Ab_data['seq_dict'] = self.aaseq_dict_ls[idx]
        Ab_data['pred_coords'] = Ab_data['pred_coords'].detach()
        At_data = self.Atdata_ls[Atdata_index]
        At_data = torch.load(At_data, map_location='cpu') if isinstance(At_data, str) else At_data

        inter_label = self.label_ls[idx]
        pos_index = self.pos_index_ls[idx]
        embed_type = self.embed_type_ls[idx]

        # return Ab_data, At_data, inter_label, pos_index, embed_type, device_info
        return Ab_data, At_data, inter_label, pos_index, embed_type, Abdata_index

    @staticmethod
    def collate_wrapper(unbatched_list):
        if 'pred_coords' in unbatched_list[0][0].keys():
            return Batch_InterDataWithPredCoords(unbatched_list)
        else:
            return Batch_Mutiwoker(unbatched_list)


class Batch_Mutiwoker(object):
    def __init__(self, unbatched_list):
        self.len_unbatchedLs = len(unbatched_list)
        if self.len_unbatchedLs == 1:
            Ab_data, At_data, inter_label, pos_index, embed_type = unbatched_list[0]
            self.model_in = get_singledata(Ab_data, At_data, inter_label, embed_type=embed_type)
        else:
            Ab_data, At_data, inter_label, pos_index, embed_type, Abdata_index = list(zip(*unbatched_list))
            # print('abatid ls is:', Ab_abatid)
            self.model_in = get_batchdata(Ab_data, At_data, inter_label, pos_index=pos_index, embed_type=embed_type, batch_size=self.len_unbatchedLs)

    def __len__(self):
        return self.len_unbatchedLs


class Dataset_3d(Dataset):
    def __init__(self, data_df=None, preAb_dict=None, init_pos_data=None, embeder=None, ab_preembeds_dict=None, template_flg=False, test_flag=False, valid_flag=False, trainval_negs=5, device='cpu', ddp=None, out_dir=None, atbox_truc=15.0):
        self.ddp, self.out_dir, self.trainval_negs = ddp, out_dir, trainval_negs
        self.fpaths, self.idx_dict, self.no_pdbls, self.template_flg = [], {}, [], template_flg
        self.test_flag, self.valid_flag, self.device, self.atbox_truc = test_flag, valid_flag, device, atbox_truc

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
        if self.preAb_dict is not None:
            data_id = data_si['data_id']
            Ab_data = self.preAb_dict[data_id]
        else:
            aaseq_dict = get_aaseqDict_BySeries(data_si)
            mutate_range = eval(data_si['MutRange']) if 'MutRange' in data_si.index else None
            cdr_fr_info = string_to_dict(data_si['cdr_fr_info'])
            Ab_data = process_Abdata(aaseq_dict, wild_Abdata_dict=self.init_pos_data['Ab_data'], pos_index=pos_index, rec_cdrfr=cdr_fr_info, db_args=self.embeder.database_args, tmp_flg=self.template_flg, mutate_range=mutate_range)
        inter_label, Ab_data['label_coords'] = int(data_si['AgClass']), None
        # singledata = get_singledata(Ab_data, self.init_pos_data['At_data'], inter_label, pos_index=pos_index, embed_type=self.embeder.embed_type, device_in=self.device)
        # return singledata
        return Ab_data, self.init_pos_data['At_data'][pos_index], inter_label, pos_index, self.embeder.embed_type, self.device

    @staticmethod
    def collate_wrapper(unbatched_list):
        return Batch_InterData(unbatched_list)

    def __len__(self):
        return len(self.fpaths)


def get_aaseqDict_BySeries(dfi, hseq_name='Hseq', lseq_name='Lseq'):
    aaseq_dict = {}
    if hseq_name in dfi:
        if isinstance(dfi[hseq_name], str):  aaseq_dict['H'] = dfi[hseq_name]
    if lseq_name in dfi:
        if isinstance(dfi[lseq_name], str):  aaseq_dict['L'] = dfi[lseq_name]
    return aaseq_dict


class Batch_InterData(object):
    def __init__(self, unbatched_list):
        self.len_unbatchedLs = len(unbatched_list)
        if self.len_unbatchedLs == 1:
            Ab_data, At_data, inter_label, pos_index, embed_type, self.device = unbatched_list[0]
            self.model_in = get_singledata(Ab_data, At_data, inter_label, embed_type=embed_type, device_in=self.device)
        else:
            Ab_data, At_data, inter_label, pos_index, embed_type, self.device = list(zip(*unbatched_list))
            self.model_in = get_batchdata(Ab_data, At_data, inter_label, pos_index=pos_index, embed_type=embed_type, device_in=self.device, batch_size=self.len_unbatchedLs)

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
    # test_dir = check_test_pdb()  # download test pdbs
    # gen_testdb()  # 创建数据集
