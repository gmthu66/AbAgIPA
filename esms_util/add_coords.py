import os.path as osp
import os
import torch
import argparse
import sys
project_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(project_dir)
import pandas as pd
from utils.general import exists
from tqdm import tqdm
from training.record import Namespace
from model.interface import DataDirPath
from glob import glob
from esms_util.esm_get_SabDabData import get_data, process_Abdata
from spec_At.data_code.Embedding import Embeder
from spec_At.Base_train_nfold import average_dictValues, overloop_items_dict
from spec_At.model.FAbAtInter_Glbmean_Physic import FAbAtInterPhysic
from spec_At.utils_df import compare_h_l_values
from SabDab.data_code.dataset_utils import Batch_InterData
from training.record import clean_config
from database.inter_pdb import get_residues_Abchains
from database.inter_data import get_Abcoords_label
from spec_At.parse_seqs import aaseqDict2fasta
from utils.pdb import pdb2fasta, get_pdb_chain_seq
from os.path import basename
spect_AtDir = osp.join(project_dir, 'SabDab')
pre_embed_dir = osp.join(project_dir, 'esms_util/Abs/pre_embed')
ddp = DataDirPath()


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody and Nanobody Structure predict by IPA')
    parser.add_argument('--id_fold', default=0, type=int)
    parser.add_argument('--embed_type', default='Physic', type=str)
    args = parser.parse_args()
    return args


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
    config.par_out_dir = osp.join(spect_AtDir, f'trained_models/IPACdr_{config.embed_type}')
    config.run_name = 'train_test'
    config.verbose = True
    config.inter_attn_depth = 1
    return config


def init_from_ckpt(model_ckpt, user_config=None):
    class_weight = torch.tensor(data=user_config.lossweight, device=user_config.device) if 'lossweight' in dir(user_config) else None
    inter_criterion = torch.nn.CrossEntropyLoss(weight=class_weight) if 'lossweight' in dir(user_config) else torch.nn.CrossEntropyLoss()
    Model = FAbAtInterPhysic
    if 'config' not in basename(model_ckpt):
        if exists(model_ckpt):
            # num_models = len(model_ckpt)
            print(f"Loading {model_ckpt}...")
            torch.load(model_ckpt)
            try:
                model = Model.load_from_checkpoint(model_ckpt)
                print(f"Successfully loaded IgFold model params.")
            except:
                print(f"Failed loaded IgFold model params. Construct model by cofigs in ckpt...")
                config_ckpt = torch.load(model_ckpt)['hyper_parameters']['config']
                config_ckpt = clean_config(config_ckpt, config=user_config)
                torch.save(config_ckpt, osp.join(user_config.out_dir, 'config_ckpt.pt'))
                model = Model(config=config_ckpt, inter_criterion=inter_criterion)
    else:
        config_ckpt = model_ckpt
        config_ckpt = torch.load(config_ckpt)
        config_ckpt = clean_config(config_ckpt, user_config=user_config)
        model = Model(config=config_ckpt, inter_criterion=inter_criterion)
    return model


def merge_model(model, static_model='IgFold', merge_number=1):
    if static_model == 'IgFold':
        static_model_paths = sorted(ddp.igfold_models)

    static_state_ls = []
    for i_ in range(merge_number):
        source_model = static_model_paths[i_]
        static_state = torch.load(source_model)['state_dict']
        static_state_ls.append(static_state)
    static_state = average_dictValues(static_state_ls)
    tgt_state = model.state_dict()  # 完成对结构预测模型的载入

    overloop_params_state, static_notOverloop_state = overloop_items_dict(tgt_state, static_state)
    print('over loop layers is:\n{}'.format('\n'.join(list(overloop_params_state.keys()))))
    if len(list(static_notOverloop_state.keys())) > 0:
        print('Missing layers is:\n{}'.format('\n'.join(list(static_notOverloop_state.keys()))))
    else:
        print('No Missing layers while merging')
    model.load_state_dict(overloop_params_state, strict=False)
    print(f'merge {merge_number} {static_model} in My model end')
    overloop_params_state = overloop_params_state
    return model


def add_coords(model_ckpt=None, try_gpu=True, id_fold=None):
    config = Namespace
    config = init_config(config)
    os.makedirs(config.par_out_dir, exist_ok=True)

    model_ckpt = osp.join(ddp.model_cofig_dir, 'config_ckpt.pt')
    if not exists(model_ckpt):
        project_path = project_dir
        ckpt_path = os.path.join(project_path, "trained_models/abatInter_SCA/*.ckpt",)
        model_ckpts = sorted(list(glob(ckpt_path)))
    if isinstance(model_ckpt, list):  model_ckpt = model_ckpts[0]

    embeder = Embeder(config)
    # trainval_df_dict, AbAtPair_dict = get_data(embeder)
    # init_pos_data = {'Ab_data': Ab_datals, 'At_data': At_datals}
    foldi = id_fold
    # train_val_df = trainval_df_dict[foldi]

    # ---------------- 进行训练与测试 ----------------------
    print(f"Training... model in Single fold_id:{foldi}.")
    config.out_dir = osp.join(config.par_out_dir, str(foldi))
    os.makedirs(config.out_dir, exist_ok=True)

    model = init_from_ckpt(model_ckpt, user_config=config)  # 这里是进行Model的初始化

    model = merge_model(model, static_model='IgFold', merge_number=1)
    # data = GenDataloader(config, train_val_df, AbAtPair_dict, embeder=embeder)
    # trainer = Trainer(config, data, model)
    # trainer.train()
    # data_df = pd.concat([trainval_df_dict[foldi]['train'], trainval_df_dict[foldi]['val']])
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    addembeddings_predir = osp.join(osp.dirname(pre_embed_dir), 'addattn_pre_embed')
    os.makedirs(addembeddings_predir, exist_ok=True)
    with torch.no_grad():
        Ab_excusedls = []
        Abdataf_ls = [f for f in os.listdir(pre_embed_dir) if f.endswith('.pt')]
        for Abdata_f in Abdataf_ls:
            Ab_data = torch.load(osp.join(pre_embed_dir, Abdata_f), map_location='cpu')
            if len(''.join([seq for seq in Ab_data['seq_dict'].values()])) > 350:
                print('Ab HL aaseq too long:', Abdata_f)
                Ab_data = modify_Abdata(Ab_data, Abdata_f)
            # elif 'pred_coords' in Ab_data.keys():
            #     continue
            batch_Ab_data = process_Abdata(Ab_data, db_args=embeder.database_args, tmp_flg=False, mutate_range=None)
            # Ab_abatid = None
            # At_data = AbAtPair_dict[Ab_abatid]['At_data']
            # inter_label = 1
            # unbatched_list = [(batch_Ab_data, At_data, inter_label, embeder.embed_type, device)]
            # batch = Batch_InterData(unbatched_list)
            # model.out = model(batch.model_in)
            Ab_data['pred_coords'] = Ab_data['label_coords']
            Ab_data['Ab_phisicEmbed'] = batch_Ab_data['Ab_phisicEmbed']
            Ab_data['Ab_phisicAttention'] = batch_Ab_data['Ab_phisicAttention']
            # fname = [fpath for fpath in os.listdir(osp.join(pre_embed_dir, Ab_abatid, 'Abs/pre_embed')) if fpath.endswith('.pt')][0]
            predAb_path = osp.join(addembeddings_predir, Abdata_f)
            Ab_data = map_datadictItem2Cpu(Ab_data)
            torch.save(Ab_data, predAb_path)
            Ab_excusedls.append(Abdata_f)
            print(f'{Abdata_f} end')
    print(f"Train model Successfully, fold_id:{foldi}.")


def modify_Abdata(Ab_data, preEmbed_dir):
    merge_fpath = [osp.join(preEmbed_dir, fpath) for fpath in os.listdir(preEmbed_dir) if fpath.endswith('.pdb')][0]
    HL_chs = ['H', 'L']
    seq_dict = {}
    for ch in HL_chs:
        seq_dict[ch] = get_pdb_chain_seq(merge_fpath, ch)
    # ab_fasta_file = pdb2fasta(merge_fpath)
    ab_fasta_file = osp.join(preEmbed_dir, 'ab.fasta')
    aaseqDict2fasta(seq_dict, ab_fasta_file)
    ab_residue_ls, _ = get_residues_Abchains(merge_fpath, HL_chs)
    label_coords, r_rangels, cdr_fr_dict, ab_rei_distAt = get_Abcoords_label(ab_residue_ls, ab_fasta_file, HL_chs)

    Ab_data['seq_dict'] = seq_dict
    Ab_data['label_coords'] = label_coords
    Ab_data['ab_rei_distAt'] = ab_rei_distAt
    Ab_data['cdr_fr_dict'] = cdr_fr_dict
    Ab_data['ab_fasta_file'] = ab_fasta_file
    return Ab_data


def check_AbsDir_indpath(dpath):
    if osp.isdir(dpath):
        return 'Abs' in os.listdir(dpath)
    return False


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


if __name__ == "__main__":
    add_coords(id_fold=0)
