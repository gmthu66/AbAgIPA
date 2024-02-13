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

project_dir = osp.dirname(osp.dirname(__file__))
sys.path.append(project_dir)

from training.record import AverageMeter, ListAverageMeter
from tqdm import tqdm
# from model.FAbAtInter1219 import FAbAtInter
from model.FAbAtInter import FAbAtInter
from model.batch_utils import clean_input, onesample_data
from model.interface import DataDirPath
from training.Base_train import TrainerBase
from training.evaluate import calculate_metrics
from utils.general import exists
from training.data import GenDataloader, get_valids
from training.record import CSVWriter, clean_config, save_dict, load_dict, Namespace
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
random.seed(22)
np.random.seed(22)
torch.manual_seed(22)
torch.cuda.manual_seed_all(22)


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody and Nanobody Structure predict by IPA')
    parser.add_argument('--id_fold', default=None, type=int)
    args = parser.parse_args()
    return args


def init_config(config):
    config.batch_size = 1
    config.num_data_workers = 0
    config.epochs = 50
    config.device = torch.device('cuda:1')
    # config.device = torch.device('cpu')
    config.test_freq = 1
    config.warmup_epochs = 0
    config.nfold = 5  # 几折验证
    config.optimizer = 'RAdam'
    config.lr = 5e-4
    config.weight_decay = 0
    config.lr_scheduler = 'CosineAnnealingLR'
    config.is_master = True
    config.trainval_negs = 10
    # config.lr_scheduler = 'CosineAnnealingLRWithWarmup'

    config.auto_resume = True
    # config.par_out_dir = osp.join(project_dir, 'trained_models/test_inter')
    config.par_out_dir = osp.join(project_dir, 'trained_models/FAbAtInter')
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
                batch_data, batch_size, seq_lens, Ab_VHLlens, max_abseqlen = clean_input(batch.model_in, device_in=self.device)
                loss_ls, prmsd_lossls, coords_lossls, bondlen_lossls, inter_lossls = [], [], [], [], []
                for i_ind in range(batch_size):
                    data_in = onesample_data(batch_data, i_ind, max_abseqlen)
                    model_out = self.model(data_in)  # grad_fn表明该变量是怎么来的，用于指导反向传播
                    if partition == 'train':
                        model_out.loss.backward()
                    y_true.extend(data_in.inter_label.cpu().numpy())
                    y_prob.extend(model_out.inter_prob.cpu().numpy())

                    loss_ls.append(model_out.loss.detach().cpu().item())
                    prmsd_lossls.append(model_out.prmsd_loss.detach().cpu().item())
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
                prmsd_losses.update(prmsd_lossls, len(batch))
                coords_losses.update(coords_lossls, len(batch))
                bondlen_losses.update(bondlen_lossls, len(batch))
                inter_losses.update(inter_lossls, len(batch))
                # if i_ % 1000 == 0:  print(f'{partition}...: {i_}/{data_loader.sampler.num_samples}')  # debug
                # if (i_ >= 30 and (partition == 'train')) or (i_ >= 20 and (partition == 'valid')):  break

            # -------------------------- 对metric进行记录 ---------------------------
            loss_avg = self.all_reduce(losses.avg)
            prmsd_loss_avg = self.all_reduce(prmsd_losses.avg)
            coords_loss_avg = self.all_reduce(coords_losses.avg)
            bondlen_loss_avg = self.all_reduce(bondlen_losses.avg)
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


def init_from_ckpt(model_ckpt, user_config=None):
    inter_criterion = torch.nn.CrossEntropyLoss()
    if 'config' not in basename(model_ckpt):
        if exists(model_ckpt):
            # num_models = len(model_ckpt)
            print(f"Loading {model_ckpt}...")
            torch.load(model_ckpt)
            try:
                model = FAbAtInter.load_from_checkpoint(model_ckpt)
                print(f"Successfully loaded IgFold model params.")
            except:
                print(f"Failed loaded IgFold model params. Construct model by cofigs in ckpt...")
                config_ckpt = torch.load(model_ckpt)['hyper_parameters']['config']
                config_ckpt = clean_config(config_ckpt, config=user_config)
                torch.save(config_ckpt, osp.join(user_config.out_dir, 'config_ckpt.pt'))
                model = FAbAtInter(config=config_ckpt, inter_criterion=inter_criterion)
    else:
        config_ckpt = model_ckpt
        config_ckpt = torch.load(config_ckpt)
        config_ckpt = clean_config(config_ckpt, user_config=user_config)
        model = FAbAtInter(config=config_ckpt, inter_criterion=inter_criterion)
    return model


def valid_dict2json(valid_dict, odir):
    save_dict(valid_dict, osp.join(odir, 'valid_dict.json'))


def train(model_ckpt=None, try_gpu=True, id_fold=None):
    ddp = DataDirPath()
    config = Namespace
    config = init_config(config)
    os.makedirs(config.par_out_dir, exist_ok=True)
    if osp.exists(osp.join(config.par_out_dir, 'valid_dict.json')):
        valid_dict = load_dict(osp.join(config.par_out_dir, 'valid_dict.json'))
        keyls = list(valid_dict.keys())
        for k_ in keyls:
            valid_dict.update({int(k_): valid_dict.pop(k_)})
    else:
        valid_dict = get_valids(nfold=config.nfold, n_splits=10)  # 获得valid id字典
        valid_dict2json(valid_dict, config.par_out_dir)

    model_ckpt = osp.join(ddp.model_cofig_dir, 'config_ckpt.pt')
    if not exists(model_ckpt):
        project_path = project_dir
        ckpt_path = os.path.join(project_path, "trained_models/abatInter_SCA/*.ckpt",)
        model_ckpts = sorted(list(glob(ckpt_path)))
    if isinstance(model_ckpt, list):  model_ckpt = model_ckpts[0]

    # ---------------- 进行训练与测试 ----------------------
    if id_fold is None:
        for foldi in valid_dict.keys():
            config.out_dir = osp.join(config.par_out_dir, str(foldi))
            os.makedirs(config.out_dir, exist_ok=True)

            model = init_from_ckpt(model_ckpt, user_config=config)
            data = GenDataloader(config, valid_dict[foldi], foldi)
            trainer = Trainer(config, data, model)
            trainer.train()

            print(f"Train IgFold model Successfully, fold_id:{foldi}.")
    else:
        foldi = id_fold
        print(f"Training... IgFold model in Single fold_id:{foldi}.")
        config.out_dir = osp.join(config.par_out_dir, str(foldi))
        os.makedirs(config.out_dir, exist_ok=True)

        model = init_from_ckpt(model_ckpt, user_config=config)
        data = GenDataloader(config, valid_dict[foldi], foldi)
        trainer = Trainer(config, data, model)
        trainer.train()

        print(f"Train IgFold model Successfully, fold_id:{foldi}.")


if __name__ == "__main__":
    args = parse_args()
    train(id_fold=args.id_fold)
