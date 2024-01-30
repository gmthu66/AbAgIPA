"""预测的结构作为骨架输入给相互作用预测模型"""
import os
import sys
import os.path as osp
import torch
import random
import numpy as np
import argparse
from glob import glob

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)
import numpy as np

from model.interface import DataDirPath
from utils.general import exists
from training.record import CSVWriter, Namespace
from spec_At.data_code.Embedding import Embeder
from SabDab.compare.metric_rf import eval_svm
from spec_At.data_code.sequence_encoding import returnCKSAAPcode, OneHot_residuetype
from SabDab.get_SabDabData import get_data
from sklearn.ensemble import RandomForestClassifier

random.seed(22)
np.random.seed(22)
torch.manual_seed(22)
torch.cuda.manual_seed_all(22)
spect_AtDir = osp.dirname(osp.dirname(__file__))
ddp = DataDirPath()


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody and Nanobody Structure predict by IPA')
    parser.add_argument('--id_fold', default=None, type=str)
    parser.add_argument('--embed_type', default='Physic', type=str)
    args = parser.parse_args()
    return args

 
def init_config(config):
    config.batch_size = 16
    config.num_data_workers = 0
    config.epochs = 50
    config.device = torch.device('cuda:0')
    # config.device = torch.device('cpu')
    config.test_freq = 1
    config.warmup_epochs = 0  # 注意,这里的warmup epoch设置为了0
    config.nfold = 5  # 几折验证
    config.optimizer = 'RAdam'
    config.lr = 5e-4
    config.weight_decay = 0
    config.lr_scheduler = 'CosineAnnealingLR'
    config.is_master = True
    config.trainval_negs = 10
    config.embed_type = ['Bert', 'Physic'][1]
    # config.lr_scheduler = 'CosineAnnealingLRWithWarmup'

    config.auto_resume = True
    # config.par_out_dir = osp.join(spect_AtDir, 'trained_models/FAbAtInter_GlbMean')
    config.par_out_dir = osp.join(spect_AtDir, f'trained_compare_models/RF_CKSAAP')
    config.run_name = 'train_test'
    config.verbose = True
    config.inter_attn_depth = 2
    return config


def shuffle_dataframe(input_df, random_seed=0):
    shuffled_df = input_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return shuffled_df


def contruct_SeqDb(data_df, embed_type='CKSAAP'):
    inter_labells, featls = [], []
    data_df = shuffle_dataframe(data_df)
    for i_ in data_df.index:
        data_si = data_df.loc[i_]
        aaseq_dict = get_aaseqDict_BySeries(data_si)
        inter_label = int(data_si['AgClass'])
        ab_seq = ''.join(seq for seq in aaseq_dict.values())
        if embed_type is 'CKSAAP':
            ab_data = np.array(returnCKSAAPcode(ab_seq, 3)).reshape((1, -1))
            at_data = np.array(returnCKSAAPcode(data_si['Atseq'], 3)).reshape((1, -1))
            # ab_data = torch.tensor(data=ab_data, dtype=torch.float32).view(-1, 20, 20)
            # at_data = torch.tensor(data=at_data, dtype=torch.float32).view(-1, 20, 20)
        elif embed_type is 'OneHot':
            ab_data = OneHot_residuetype(ab_seq).reshape((1, -1))
            at_data = OneHot_residuetype(at_data).reshape((1, -1))
        feat = np.concatenate((ab_data, at_data), axis=1)
        inter_labells.append(inter_label)
        featls.append(feat)
    featls = np.concatenate(featls, axis=0)
    inter_labells = np.array(inter_labells)
    return featls, inter_labells


def train(model_ckpt=None, try_gpu=True, id_fold=None):
    config = Namespace
    config.id_fold = id_fold
    config = init_config(config)
    os.makedirs(config.par_out_dir, exist_ok=True)

    model_ckpt = osp.join(ddp.model_cofig_dir, 'config_ckpt.pt')
    if not exists(model_ckpt):
        ckpt_path = os.path.join(project_dir, "trained_models/abatInter_SCA/*.ckpt",)
        model_ckpts = sorted(list(glob(ckpt_path)))
    if isinstance(model_ckpt, list):  model_ckpt = model_ckpts[0]

    embeder = Embeder(config)
    trainval_df_dict, AbAtPair_dict = get_data(embeder)

    # init_pos_data = {'Ab_data': Ab_datals, 'At_data': At_datals}
    foldi = id_fold
    train_val_df = trainval_df_dict[foldi]
    train_val_df = {k_ : shuffle_dataframe(v) for k_, v in train_val_df.items()}

    # ---------------- 进行训练与测试 ----------------------
    print(f"Training... IgFold model in Single fold_id:{foldi}.")
    config.out_dir = osp.join(config.par_out_dir, str(foldi))
    os.makedirs(config.out_dir, exist_ok=True)

    train_db, train_label = contruct_SeqDb(train_val_df['train'])
    val_db, val_label = contruct_SeqDb(train_val_df['val'])
    model = RandomForestClassifier(random_state=2023)
    clf_proba = model.fit(train_db, train_label)  # 返回每个样本对应的类别概率

    val_acc, val_precision, val_recall, val_tnr, val_auc, valid_auc_pr, val_f1, fpr, tpr, pr_precisions, pr_recalls = eval_svm(val_db, val_label, clf_proba)

    # columns = ['Partition', 'Epoch', 'Loss', 'Prmsd_loss', 'Coords_loss', 'Bondlen_loss', 'Inter_loss']
    out_dir = osp.join(config.par_out_dir, f'{foldi}')
    os.makedirs(out_dir, exist_ok=True)
    metrics_cols = ['AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'TNR', 'F1_score']
    csv_writer = CSVWriter(osp.join(out_dir, 'metric.csv'), columns= metrics_cols, overwrite=False)
    csv_writer.add_scalar('AUC-ROC', val_auc)
    csv_writer.add_scalar('AUC-PR', valid_auc_pr)
    csv_writer.add_scalar('Precision', val_precision)
    csv_writer.add_scalar('Recall', val_recall)
    csv_writer.add_scalar('TNR', val_tnr)
    csv_writer.add_scalar('F1_score', val_f1)
    csv_writer.write()
    print(f"Train RF model Successfully, fold_id:{foldi}.")


def get_aaseqDict_BySeries(dfi, hseq_name='Hseq', lseq_name='Lseq'):
    aaseq_dict = {}
    if hseq_name in dfi:
        if isinstance(dfi[hseq_name], str):  aaseq_dict['H'] = dfi[hseq_name]
    if lseq_name in dfi:
        if isinstance(dfi[lseq_name], str):  aaseq_dict['L'] = dfi[lseq_name]
    return aaseq_dict


if __name__ == "__main__":
    for i_ in list(range(5)):
        args = parse_args()
        args.id_fold = i_
        # train(id_fold=args.id_fold)
        train(id_fold=args.id_fold)
