import os
import os.path as osp
from database.preprocess.collect_utils import movepdb_from_sabdab
from model.interface import DataDirPath

import torch
project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
from utils.general import exists
from training.record import Namespace
from model.interface import DataDirPath
from glob import glob
from copy import deepcopy
from spec_At.data_code.Embedding import Embeder
from abatInter_SCA.spec_At.data_code.load_coords_model import init_from_ckpt, merge_model
from SabDab.get_SabDabData import process_Abdata
from SabDab.data_code.dataset_utils import Batch_InterData
from database.numbering import get_cdrfr_info
ddp = DataDirPath()


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


def appen_datals(datals, datai):
    if isinstance(datai, list):
        datals.extend(datai)
    else:
        datals.append(datai)


class CoordsPreder(object):
    def __init__(self, embeder=None):
        config = Namespace
        config = init_config(config)
        # os.makedirs(config.par_out_dir, exist_ok=True)
        model_ckpt = osp.join(ddp.model_cofig_dir, 'config_ckpt.pt')
        if not exists(model_ckpt):
            project_path = project_dir
            ckpt_path = os.path.join(project_path, "trained_models/abatInter_SCA/*.ckpt",)
            model_ckpts = sorted(list(glob(ckpt_path)))
        if isinstance(model_ckpt, list):  model_ckpt = model_ckpts[0]
        embeder = Embeder(config) if embeder is None else embeder
        coords_config = deepcopy(config)
        coords_config.node_dim = 64
        model = init_from_ckpt(model_ckpt, user_config=coords_config)  # 这里是进行Model的初始化
        self.device = torch.device('cuda:0')
        model = merge_model(model, static_model='IgFold', merge_number=1)
        self.model = model.to(self.device)
        self.embeder = embeder
        self.model.eval()

    def _remove_dictkeys(self, dict_in, rm_ls):
        for key_to_remove in rm_ls:
            if key_to_remove in dict_in:
                del dict_in[key_to_remove]
        return dict_in

    def addPredCoords(self, Ab_data):
        batch_Ab_data = process_Abdata(Ab_data, db_args=self.embeder.database_args, tmp_flg=False, mutate_range=None)
        At_data, inter_label = None, 1
        unbatched_list = [(batch_Ab_data, At_data, inter_label, self.embeder.embed_type, self.device)]
        batch = Batch_InterData(unbatched_list)
        model_out = self.model(batch.model_in)
        Ab_data['pred_coords'] = model_out.coords.detach()
        Ab_data['Ab_phisicEmbed'] = batch_Ab_data['Ab_phisicEmbed']
        Ab_data['Ab_phisicAttention'] = batch_Ab_data['Ab_phisicAttention']
        Ab_data = self._remove_dictkeys(Ab_data, ['Ab_embeddings', 'Ab_attentions'])
        # predAb_path = osp.join(pre_embed_dir, f'{Ab_abatid}/Abs/pre_embed', fname)
        Ab_data = map_datadictItem2Cpu(Ab_data)
        return Ab_data


def predcoordsByAbdata(model_ckpt=None, embeder=None,):
    config = Namespace
    config = init_config(config)
    os.makedirs(config.par_out_dir, exist_ok=True)

    model_ckpt = osp.join(ddp.model_cofig_dir, 'config_ckpt.pt')
    if not exists(model_ckpt):
        project_path = project_dir
        ckpt_path = os.path.join(project_path, "trained_models/abatInter_SCA/*.ckpt",)
        model_ckpts = sorted(list(glob(ckpt_path)))
    if isinstance(model_ckpt, list):  model_ckpt = model_ckpts[0]
    embeder = Embeder(config) if embeder is None else embeder
    model = init_from_ckpt(model_ckpt, user_config=config)  # 这里是进行Model的初始化
    model = merge_model(model, static_model='IgFold', merge_number=1)
    device = torch.device('cuda:0')
    model = model.to(device)
    with torch.no_grad():
        batch_Ab_data = process_Abdata(Ab_data, db_args=embeder.database_args, tmp_flg=False, mutate_range=None)
        At_data, inter_label = None, 1
        unbatched_list = [(batch_Ab_data, At_data, inter_label, embeder.embed_type, device)]
        batch = Batch_InterData(unbatched_list)
        model.out = model(batch.model_in)
        Ab_data['pred_coords'] = model.out.coords
        Ab_data['Ab_phisicEmbed'] = batch_Ab_data['Ab_phisicEmbed']
        Ab_data['Ab_phisicAttention'] = batch_Ab_data['Ab_phisicAttention']
        # predAb_path = osp.join(pre_embed_dir, f'{Ab_abatid}/Abs/pre_embed', fname)
        Ab_data = map_datadictItem2Cpu(Ab_data)
        # torch.save(Ab_data, predAb_path)
        # print(f'{Ab_abatid} end')


def modify_Abdata(Ab_data):
    seq_dict = {}
    seq_dict['H'] = Ab_data['seq_dict']['H']
    seq_dict['L'] = Ab_data['seq_dict']['L']
    Ab_data['seq_dict'] = seq_dict
    cdr_fr_info = get_cdrfr_info(ab_aaseq=seq_dict['H'] + seq_dict['L'], h_len=len(seq_dict['H']))
    Ab_data['cdr_fr_dict'] = cdr_fr_info
    return Ab_data


def rmsd_info(flat_rmsd, cdr_fr_dict):
    pass


def init_config(config):
    config.batch_size = 8
    config.num_data_workers = 0
    config.only_cdr = True
    config.device = torch.device('cuda:0')
    # config.device = torch.device('cpu')
    config.test_freq = 1
    config.warmup_epochs = 0  # 注意,这里的warmup epoch设置为了0
    config.nfold = 5  # 几折验证
    config.optimizer = 'RAdam'

    config.epochs = 50
    config.lr = 5e-4
    config.fineturnlr = 1e-6
    config.weight_decay = 0
    config.early_stop = 10
    config.lr_scheduler = 'CosineAnnealingLR'
    config.trainval_negs = 10
    # config.lossweight = [0.9, 0.1]
    config.monitor_metric = 'AUC-ROC'
    config.embed_type = ['Bert', 'Physic'][1]
    config.is_master = True
    config.auto_resume = True

    config.synergy = True
    # config.synergy_finetunelayers = ['template_ipa']
    config.synergy_finetunelayers = ['']
    config.run_name = 'train_test'
    config.verbose = True
    config.inter_attn_depth = 1
    return config


def map_datadictItem2Cpu(data_dict):
    for key_ in data_dict.keys():
        if isinstance(data_dict[key_], list):
            if isinstance(data_dict[key_], torch.Tensor):
                for i, v in enumerate(data_dict[key_]):
                    data_dict[key_][i] = data_dict[key_][i].to(torch.device('cpu'))
        elif isinstance(data_dict[key_], torch.Tensor):
            data_dict[key_] = data_dict[key_].to(torch.device('cpu'))
        else:
            pass
    return data_dict
