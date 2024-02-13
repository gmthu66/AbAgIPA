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
from os.path import basename
from glob import glob

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)

from training.record import AverageMeter, ListAverageMeter
from tqdm import tqdm
# from spec_At.model.FAbAtInter_SoftAtten import FAbAtInter
from spec_At.model.FAbAtInter_IPAGlbmean_Physic import FAbAtInterIPA
from model.interface import DataDirPath
from training.evaluate import calculate_metrics
from utils.general import exists
from training.record import CSVWriter, clean_config, save_dict, load_dict, Namespace
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from spec_At.data_code.Embedding import Embeder
from spec_At.synergyTraining.merge_params import merge_model
from spec_At.Covid19.get_Covid_data import get_data
from spec_At.Base_train_nfold import TrainerBase
from spec_At.Covid19.databaseCovid import GenDataloader, get_nfold_datadf, save_nfold_df2csv, try_read_train_val
random.seed(22)
np.random.seed(22)
torch.manual_seed(22)
torch.cuda.manual_seed_all(22)
spect_AtDir = osp.dirname(__file__)
ddp = DataDirPath()


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody and Nanobody Structure predict by IPA')
    parser.add_argument('--id_fold', default=0, type=int)
    parser.add_argument('--embed_type', default='Physic', type=str)
    args = parser.parse_args()
    return args


def init_config(config):
    config.batch_size = 1
    config.num_data_workers = 0
    config.epochs = 50
    config.device = torch.device('cuda:0')
    config.warmup_epochs = 5  # 注意,这里的warmup epoch设置为了0
    config.test_freq = 1
    config.nfold = 5  # 几折验证
    config.optimizer = 'RAdam'
    config.lr = 5e-4
    config.weight_decay = 0
    config.lr_scheduler = 'CosineAnnealingLR'
    config.is_master = True
    config.trainval_negs = 10
    config.embed_type = ['Bert', 'Physic'][1]
    # config.lr_scheduler = 'CosineAnnealingLRWithWarmup'
    config.lossweight = [0.9, 0.1]

    config.auto_resume = True
    config.synergy = True
    config.synergy_finetunelayers = ['template_ipa']
    config.par_out_dir = osp.join(spect_AtDir, f'trained_models/FAbAtInter_IPAGlbMean_{config.embed_type}')
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

    def _train_epoch(self, epoch, data_loader, data_sampler, partition):
        # reshuffle data across GPU workers
        losses = ListAverageMeter('Loss')
        prmsd_losses = ListAverageMeter('Prmsd_loss')
        coords_losses = ListAverageMeter('Coords_loss')
        bondlen_losses = ListAverageMeter('Bondlen_loss')
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
        with context:
            y_true, y_prob = [], []
            for batch in tqdm(data_loader):
                i_ += 1
                if batch.model_in is None:  continue
                # batch_data, batch_size, seq_lens, Ab_VHLlens, max_abseqlen = clean_input(batch.model_in, device_in=self.device)
                loss_ls, prmsd_lossls, coords_lossls, bondlen_lossls, inter_lossls = [], [], [], [], []
                # for i_ind in range(batch_size):
                    # data_in = onesample_data(batch_data, i_ind, max_abseqlen)
                model_out = self.model(batch.model_in)  # grad_fn表明该变量是怎么来的，用于指导反向传播
                if partition == 'train':
                    model_out.loss.backward()
                y_true.extend(batch.model_in.inter_label.cpu().numpy())
                y_prob.extend(model_out.inter_prob.cpu().numpy())

                loss_ls.append(model_out.loss.detach().cpu().item())
                if batch.model_in.Ab_coords_label is not None:
                    prmsd_lossls.append(model_out.prmsd_loss.detach().cpu().item())  # 基于结构模型进行微调应该是可行的?
                    coords_lossls.append(model_out.coords_loss.detach().cpu().item())
                    bondlen_lossls.append(model_out.bondlen_loss.detach().cpu().item())
                inter_lossls.append(model_out.inter_loss.detach().cpu().item())

                if partition == 'train':
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # try:
                    # if not self._check_batch(model_in, data_loader.dataset.logger): continue
                # except:
                #     print(f'{self.partition} Error at train_epoch iter\t{model_in.fpaths[0]}', file=data_loader.dataset.logger, flush=True)
                #     assert 1 > 2
                losses.update(loss_ls, len(batch))
                if batch.model_in.Ab_coords_label is not None:
                    prmsd_losses.update(prmsd_lossls, len(batch))
                    coords_losses.update(coords_lossls, len(batch))
                    bondlen_losses.update(bondlen_lossls, len(batch))
                inter_losses.update(inter_lossls, len(batch))
                torch.cuda.empty_cache()  # 释放显存
                # if i_ % 1000 == 0:  print(f'{partition}...: {i_}/{data_loader.sampler.num_samples}')  # debug
                # if (i_ >= 30 and (partition == 'train')) or (i_ >= 20 and (partition == 'valid')):  break  # debug

            # -------------------------- 对metric进行记录 ---------------------------
            loss_avg = self.all_reduce(losses.avg)
            # prmsd_loss_avg = self.all_reduce(prmsd_losses.avg)
            # coords_loss_avg = self.all_reduce(coords_losses.avg)
            # bondlen_loss_avg = self.all_reduce(bondlen_losses.avg)
            prmsd_loss_avg = 100
            coords_loss_avg = 100
            bondlen_loss_avg = 100
            inter_loss_avg = self.all_reduce(inter_losses.avg)
            metric_dict = calculate_metrics(y_true=np.array(y_true), y_probs=np.array(y_prob))
            if self.verbose:
                current_lr = self.optimizer.param_groups[0]['lr']
                lr = f'{current_lr:.8f}' if partition == 'train' else '--'
                logging.info(f'Epoch {epoch} {partition.upper()}, Loss: {loss_avg:.4f}, '
                            f'Prmsd_loss: {prmsd_loss_avg:.3f}, Coords_loss: {coords_loss_avg:.3f}, Bondlen_loss: {bondlen_loss_avg:.3f}, Inter_loss: {inter_loss_avg:.3f} ')
                self.csv_writer.add_scalar('Partition', partition)
                self.csv_writer.add_scalar('Epoch', epoch)
                self.csv_writer.add_scalar('Loss', loss_avg)
                self.csv_writer.add_scalar('Prmsd_loss', prmsd_loss_avg)
                self.csv_writer.add_scalar('Coords_loss', coords_loss_avg)
                self.csv_writer.add_scalar('Bondlen_loss', bondlen_loss_avg)
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
                    self.tb_writer.add_scalar(f'Loss/{partition}', loss_avg, epoch)
                    self.tb_writer.add_scalar(f'Prmsd_loss/{partition}', prmsd_loss_avg, epoch)
                    self.tb_writer.add_scalar(f'Coords_loss/{partition}', coords_loss_avg, epoch)
                    self.tb_writer.add_scalar(f'Bondlen_loss/{partition}', bondlen_loss_avg, epoch)
                    self.tb_writer.add_scalar(f'Inter_loss/{partition}', inter_loss_avg, epoch)

                    self.tb_writer.add_scalar(f'AUC-ROC/{partition}', metric_dict['AUC-ROC'], epoch)
                    self.tb_writer.add_scalar(f'AUC-PR/{partition}', metric_dict['AUC-PR'], epoch)
                    self.tb_writer.add_scalar(f'Precision/{partition}', metric_dict['Precision'], epoch)
                    self.tb_writer.add_scalar(f'Recall/{partition}', metric_dict['Recall'], epoch)
                    self.tb_writer.add_scalar(f'TNR/{partition}', metric_dict['TNR'], epoch)
                    self.tb_writer.add_scalar(f'F1_score/{partition}', metric_dict['F1 Score'], epoch)
                    if partition == "train":
                        self.tb_writer.add_scalar('LR', current_lr, epoch)
            return loss_avg, metric_dict


def config_trainset_PosAndNeg():
    train_set_ratioDict = {'A': [7894, 7237], 'B': [7894, 8237], 'C': [7894, 9237], 'D': [7894, 10237], 'E': [7894, 11237], 'F': [7894, 12237], 'G': [7894, 13237], 'H': [7894, 14237], 'I': [7894, 15237], 'J': [7894, 16237], 'K': [7894, 17237]}
    test_set_PosAndNeg = [1622, 1621]
    return train_set_ratioDict, test_set_PosAndNeg


def init_from_ckpt(model_ckpt, user_config=None):
    class_weight = torch.tensor(data=user_config.lossweight, device=user_config.device) if 'lossweight' in dir(user_config) else None
    inter_criterion = torch.nn.CrossEntropyLoss(weight=class_weight) if 'lossweight' in dir(user_config) else torch.nn.CrossEntropyLoss()
    Model = FAbAtInterIPA if user_config.embed_type is 'Bert' else FAbAtInterIPA
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


def valid_dict2json(valid_dict, odir):
    save_dict(valid_dict, osp.join(odir, 'valid_dict.json'))


def train(model_ckpt=None, try_gpu=True, id_fold=None):
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
    Covid_df, Ab_datals, At_datals = get_data(embeder)
    pos_df = Covid_df[Covid_df['AgClass'] == 1]
    neg_df = Covid_df[Covid_df['AgClass'] == 0]
    init_pos_data = {'Ab_data': Ab_datals, 'At_data': At_datals}

    foldi = id_fold
    data_csv_dir = osp.join(osp.dirname(__file__), 'Covid19database')
    train_val_df = try_read_train_val(data_csv_dir, foldi)
    if train_val_df is None:
        train_val_nfold_df = get_nfold_datadf(pos_df, neg_df, nfold=5)
        save_nfold_df2csv(train_val_nfold_df, data_csv_dir)
        train_val_df = train_val_nfold_df[foldi]

    # ---------------- 进行训练与测试 ----------------------
    print(f"Training... IgFold model in Single fold_id:{foldi}.")
    config.out_dir = osp.join(config.par_out_dir, str(foldi))
    os.makedirs(config.out_dir, exist_ok=True)

    model = init_from_ckpt(model_ckpt, user_config=config)  # 这里是进行Model的初始化
    # data = GenDataloader(config, valid_dict[foldi], foldi)
    # if config.synergy:
    #     merge_model(model, static_model='IgFold')
    data = GenDataloader(config, train_val_df, init_pos_data, embeder=embeder)
    trainer = Trainer(config, data, model)
    trainer.train()

    print(f"Train IgFold model Successfully, fold_id:{foldi}.")


if __name__ == "__main__":
    args = parse_args()
    # args.id_fold = 'A'
    train(id_fold=args.id_fold)
