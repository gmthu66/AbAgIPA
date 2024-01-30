"""预测的结构作为骨架输入给相互作用预测模型"""
import os
import sys
import os.path as osp
import torch
import contextlib
import random
import numpy as np
import logging
import argparse
from glob import glob

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)

from training.record import ListAverageMeter
from tqdm import tqdm
from model.interface import DataDirPath
from training.evaluate import calculate_metrics
from utils.general import exists
from training.record import CSVWriter, save_dict, Namespace
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from spec_At.data_code.Embedding import Embeder
from spec_At.data_code.seqData import GenDataloader
from spec_At.model.CNN import SiameseNetwork
from spec_At.synergyTraining.SeqBase_train import TrainerBase
from SabDab.get_SabDabData import get_data
from training.train_utils import inter_class_loss

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
    config.par_out_dir = osp.join(spect_AtDir, f'trained_compare_models/CNN_CKSAAP')
    config.run_name = 'train_test'
    config.verbose = True
    config.inter_attn_depth = 2
    return config


class Trainer(TrainerBase):

    def __init__(self, config, data, model, verbose=True):
        config.try_gpu, self.verbose = True, verbose  # 测试
        super().__init__(config, data, model)
        self._init_model()  # 对模型参数初始化
        if self.verbose:
            self.tb_writer = SummaryWriter(log_dir=self.out_dir, filename_suffix=f'.{self.run_name}')
            columns = ['Partition', 'Epoch', 'Loss', 'Prmsd_loss', 'Coords_loss', 'Bondlen_loss', 'Inter_loss']
            metrics_cols = ['AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'TNR', 'F1_score']
            self.csv_writer = CSVWriter(osp.join(self.out_dir, 'metric.csv'), columns=columns + metrics_cols, overwrite=False)

    def _check_batch(self, model_in, logger=None):
        if (model_in.coords_label is None and self.partition == 'train'):
            if model_in.embeddings is not None:
                if (sum([max(e.shape[1], 0) for e in model_in.embeddings]) != model_in.coords_label.size(1)):
                    print(f'{self.partition} Check_batch Error at train_epoch iter\t{model_in.fpaths[0]}', file=logger, flush=True)
                    return False
            else:
                print(f'{self.partition} Check_batch Error at train_epoch iter\t{model_in.fpaths[0]}', file=logger, flush=True)
                return False
        return True

    def _train_epoch(self, epoch, data_loader, data_sampler, partition, inter_criterion=None):
        # reshuffle data across GPU workers
        inter_losses = ListAverageMeter('Inter_losses')
        self.partition = partition
        if isinstance(data_sampler, DistributedSampler):
            data_sampler.set_epoch(epoch)

        if partition == 'train':
            self.model.train()
        else:
            self.model.eval()
        i_ = 0
        context = contextlib.nullcontext() if partition == 'train' else torch.no_grad()
        if inter_criterion is None:  inter_criterion = torch.nn.CrossEntropyLoss()
        with context:
            y_true, y_prob = [], []
            for batch in tqdm(data_loader):
                i_ += 1
                if batch is None:  continue
                # batch_data, batch_size, seq_lens, Ab_VHLlens, max_abseqlen = clean_input(batch.model_in, device_in=self.device)
                inter_lossls = []
                ab_data, at_data, inter_label = batch
                inter_logic = self.model(ab_data.to(self.device), at_data.to(self.device))  # grad_fn表明该变量是怎么来的，用于指导反向传播
                inter_loss = inter_class_loss(inter_logic, inter_label.to(self.device), inter_criterion)
                if partition == 'train':
                    inter_loss.backward()
                y_true.extend(inter_label.cpu().numpy())
                y_prob.extend(inter_logic.detach().cpu().numpy())

                inter_lossls.append(inter_loss.detach().cpu().item())

                if partition == 'train':
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                inter_losses.update(inter_lossls, len(batch))
                # if i_ % 1000 == 0:  print(f'{partition}...: {i_}/{data_loader.sampler.num_samples}')  # debug
                # if (i_ >= 30 and (partition == 'train')) or (i_ >= 20 and (partition == 'valid')):  break  # debug

            # -------------------------- 对metric进行记录 ---------------------------
            inter_loss_avg = self.all_reduce(inter_losses.avg)
            metric_dict = calculate_metrics(y_true=np.array(y_true), y_probs=np.array(y_prob))
            if self.verbose:
                current_lr = self.optimizer.param_groups[0]['lr']
                lr = f'{current_lr:.8f}' if partition == 'train' else '--'
                logging.info(f'Epoch {epoch} {partition.upper()}, Inter_loss: {inter_loss_avg:.3f}, ')
                self.csv_writer.add_scalar('Partition', partition)
                self.csv_writer.add_scalar('Epoch', epoch)
                self.csv_writer.add_scalar('Inter_loss', inter_loss_avg)

                self.csv_writer.add_scalar('AUC-ROC', metric_dict['AUC-ROC'])
                self.csv_writer.add_scalar('AUC-PR', metric_dict['AUC-PR'])
                self.csv_writer.add_scalar('Precision', metric_dict['Precision'])
                self.csv_writer.add_scalar('Recall', metric_dict['Recall'])
                self.csv_writer.add_scalar('TNR', metric_dict['TNR'])
                self.csv_writer.add_scalar('F1_score', metric_dict['F1 Score'])

                self.csv_writer.write()
                if self.tb_writer is not None:
                    if partition == "test":
                        epoch = epoch // self.test_freq
                    self.tb_writer.add_scalar(f'Inter_loss/{partition}', inter_loss_avg, epoch)

                    self.tb_writer.add_scalar(f'AUC-ROC/{partition}', metric_dict['AUC-ROC'], epoch)
                    self.tb_writer.add_scalar(f'AUC-PR/{partition}', metric_dict['AUC-PR'], epoch)
                    self.tb_writer.add_scalar(f'Precision/{partition}', metric_dict['Precision'], epoch)
                    self.tb_writer.add_scalar(f'Recall/{partition}', metric_dict['Recall'], epoch)
                    self.tb_writer.add_scalar(f'TNR/{partition}', metric_dict['TNR'], epoch)
                    self.tb_writer.add_scalar(f'F1_score/{partition}', metric_dict['F1 Score'], epoch)
                    if partition == "train":
                        self.tb_writer.add_scalar('LR', current_lr, epoch)
            return inter_loss_avg, metric_dict


def shuffle_dataframe(input_df, random_seed=0):
    shuffled_df = input_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return shuffled_df


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

    # embeder = Embeder(config)
    # Covid_df, Ab_datals, At_datals = get_data(embeder, atseq=True)
    # pos_df = Covid_df[Covid_df['AgClass'] == 1]
    # neg_df = Covid_df[Covid_df['AgClass'] == 0]
    # init_pos_data = {'Ab_data': Ab_datals, 'At_data': At_datals}
    # foldi = id_fold
    # data_csv_dir = osp.join(project_dir, 'spec_At/Covid19/Covid19database')
    # train_val_df = try_read_train_val(data_csv_dir, foldi)
    # if train_val_df is None:
    #     train_val_nfold_df = get_nfold_datadf(pos_df, neg_df, nfold=5)
    #     save_nfold_df2csv(train_val_nfold_df, data_csv_dir)
    #     train_val_df = train_val_nfold_df[foldi]


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

    model = SiameseNetwork()

    data = GenDataloader(config, train_val_df, init_pos_data=None, input_seqtype='AbAtSeq', embed_type='CKSAAP')
    trainer = Trainer(config, data, model)
    trainer.train()
    print(f"Train IgFold model Successfully, fold_id:{foldi}.")


if __name__ == "__main__":
    for i_ in list(range(5)):
        args = parse_args()
        args.id_fold = i_
        # train(id_fold=args.id_fold)
        train(id_fold=args.id_fold)
