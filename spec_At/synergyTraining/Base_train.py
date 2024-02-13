import torch
import time
import os
import numpy as np
import logging
import time, shutil
import os.path as osp
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from training.lr_scheduler import get_lr_scheduler, WarmupCosineLR
from training.opt import RAdam
from training.record import save_curves_array
from model.interface import DataDirPath
ddp = DataDirPath()


class TrainerBase(ABC):

    def __init__(self, config, data, model):
        super().__init__()
        device = getattr(config, 'device')
        # device = torch.device("cuda:0" if torch.cuda.is_available() and config.try_gpu else "cpu")
        print(f"Using device: {device}")
        time.sleep(3)
        self.model = model
        self.synergy_flg = config.synergy
        if self.synergy_flg:
            self.merge_model(static_model='IgFold', merge_number=1)

        self.device = config.device
        self.epochs = config.epochs
        self.monitor_metric = getattr(config, 'monitor_metric') if hasattr(config, 'monitor_metric') else 'loss'
        self.earlt_stop_trunc = getattr(config, 'early_stop') if hasattr(config, 'early_stop') else 10
        self.predcoords_exist = config.predcoords_exist if hasattr(config, 'predcoords_exist') else False
        self.test_freq = config.test_freq
        self.auto_resume = config.auto_resume
        self.out_dir = config.out_dir
        self.best_perf = float('inf') if self.monitor_metric in ['loss'] else float('-1.0')
        self.is_master = config.is_master
        self.train_loader = data.train_loader
        self.train_sampler = data.train_sampler
        self.valid_loader = data.valid_loader
        self.test_loader = data.test_loader
        self.start_epoch = 1
        self.run_name = config.run_name

        if 'cuda' in self.device.type:
            self.model.to(self.device)

        # optimizer
        if config.optimizer not in ['Adam', 'AdamW', 'RAdam']:
            raise NotImplementedError
        if not self.synergy_flg:
            optim = getattr(torch.optim, config.optimizer) if config.optimizer in ['Adam', 'AdamW'] else RAdam
            self.optimizer = optim(self.model.parameters(),
                                lr=config.lr,
                                betas=(0.9, 0.999), 
                                eps=1e-08,
                                weight_decay=config.weight_decay)
        else:
            self.optimizer = self._synergy_optimizer(config)
        # LR scheduler
        # self.scheduler = get_lr_scheduler(scheduler=config.lr_scheduler,
        #                                 optimizer=self.optimizer,
        #                                 warmup_epochs=config.warmup_epochs, 
        #                                 total_epochs=config.epochs)
        self.scheduler = WarmupCosineLR(self.optimizer, config.warmup_epochs, config.epochs)
        self.tb_writer = None
        print("Loaded AntiBERTy model.")

    def train(self):
        try:
            self._auto_resume()  # 这里是获得上一次的训练结果
        except:
            if self.is_master:
                logging.info(f'Failed to load checkpoint from {self.out_dir}, start training from scratch..')
        train_t0 = time.time()
        epoch_times, epoch, early_stop_indecator = [], 0, 0
        with tqdm(range(self.start_epoch, self.epochs+1)) as tq:
            for epoch in tq:
                #  remove_sbatch_logs()
                tq.set_description(f'Epoch {epoch}')
                epoch_t0 = time.time()
                train_loss, train_metric_dict = self._train_epoch(epoch=epoch,
                                                           data_loader=self.train_loader,
                                                           data_sampler=self.train_sampler,
                                                           partition='train')
                valid_loss, valid_metric_dict = self._train_epoch(epoch=epoch,
                                                           data_loader=self.valid_loader,
                                                           data_sampler=None,
                                                           partition='valid')  # 进行验证实验

                self.scheduler.step()
                tq.set_postfix(train_loss=train_loss, valid_loss=valid_loss)
                # tq.set_postfix(train_loss=train_loss, valid_loss=valid_loss, train_prmsd=abs(train_prmsd), valid_prmsd=abs(valid_prmsd))

                epoch_times.append(time.time() - epoch_t0)

                # save checkpoint
                if self.monitor_metric is 'loss':
                    monitor_metric = valid_loss
                    is_best = (monitor_metric < self.best_perf)  # 保存当前的最好结果
                    self.best_perf = min(valid_loss, self.best_perf)
                else:
                    monitor_metric = valid_metric_dict[self.monitor_metric]
                    is_best = (monitor_metric > self.best_perf)
                    self.best_perf = max(monitor_metric, self.best_perf)
                if self.is_master:
                    self._save_checkpoint(epoch=epoch, is_best=is_best, best_perf=self.best_perf)

                # predict on test set using the latest model
                if epoch % self.test_freq == 0:
                    if self.is_master:
                        logging.info('Evaluating the latest model on test set')
                    test_loss, test_metric_dict = self._train_epoch(epoch=epoch, data_loader=self.test_loader, 
                                                              data_sampler=None,
                                                              partition='test')
                    if is_best:  # 保存一个npz文件, 它是一个
                        save_curves_array(test_metric_dict, osp.join(osp.dirname(self.csv_writer.csv_fpath), 'test_curves.npz'))
                if is_best:  # 保存一个npz文件, 它是一个
                    early_stop_indecator = 0
                else:
                    early_stop_indecator += 1
                if early_stop_indecator >= self.earlt_stop_trunc:
                    break

        # evaluate best model on test set
        if self.is_master:
            log_msg = [f'Total training time: {time.time() - train_t0:.1f} sec,',
                    f'total number of epochs: {epoch:d},',
                    f'average epoch time: {np.mean(epoch_times):.1f} sec']
            logging.info(' '.join(log_msg))
            self.tb_writer = None  # do not write to tensorboard
            logging.info('---------Evaluate Best Model on Test Set---------------')
        with open(os.path.join(self.out_dir, 'model_best.pt'), 'rb') as fin:
            best_model = torch.load(fin, map_location='cpu')['model']
        self.model.load_state_dict(best_model)
        test_loss, test_metric_dict = self._train_epoch(epoch=-1,
                        data_loader=self.test_loader,
                        data_sampler=None,
                        partition='test')
        save_curves_array(test_metric_dict, osp.join(osp.dirname(self.csv_writer.csv_fpath), 'test_curves.npz'))


    def _save_checkpoint(self, epoch, is_best, best_perf):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_perf': best_perf,
        }
        filename = os.path.join(self.out_dir, 'model_last.pt')
        torch.save(state_dict, filename)
        if is_best:
            logging.info(f'Saving current model as the best')
            shutil.copyfile(filename, os.path.join(self.out_dir, 'model_best.pt')) 

    def merge_model(self, static_model='IgFold', merge_number=1):
        if static_model == 'IgFold':
            static_model_paths = sorted(ddp.igfold_models)

        static_state_ls = []
        for i_ in range(merge_number):
            source_model = static_model_paths[i_]
            static_state = torch.load(source_model)['state_dict']
            static_state_ls.append(static_state)
        static_state = average_dictValues(static_state_ls)
        tgt_state = self.model.state_dict()  # 完成对结构预测模型的载入

        overloop_params_state, static_notOverloop_state = overloop_items_dict(tgt_state, static_state)
        print('over loop layers is:\n{}'.format('\n'.join(list(overloop_params_state.keys()))))
        if len(list(static_notOverloop_state.keys())) > 0:
            print('Missing layers is:\n{}'.format('\n'.join(list(static_notOverloop_state.keys()))))
        else:
            print('No Missing layers while merging')
        self.model.load_state_dict(overloop_params_state, strict=False)
        print(f'merge {merge_number} {static_model} in My model end')
        self.overloop_params_state = overloop_params_state

    def _init_model(self, verbose=True):  # 对model参数进行初始化
        if verbose:
            for name in self.model.state_dict():
                print(name)  # 打印出model中每一层的名称
        # for name, p in self.model.named_parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_normal_(p)
        #         # nn.init.kaiming_normal_(p)
        #     elif p.dim() == 1 and p.size(0) > 0:
        #         if 'bias' in name:
        #             nn.init.zeros_(p)
        #         elif 'weight' in name:
        #             nn.init.ones_(p)
        # print('init model end')

    def _auto_resume(self):
        """查看之前是否有过训练"""
        assert self.auto_resume
        # load from local output directory
        with open(os.path.join(self.out_dir, 'model_last.pt'), 'rb') as fin:
            checkpoint = torch.load(fin, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_perf = checkpoint['best_perf']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.is_master:
            logging.info(f'Loaded checkpoint from {self.out_dir}, resume training at epoch {self.start_epoch}..')

    def _synergy_optimizer(self, config):
        optim = getattr(torch.optim, config.optimizer) if config.optimizer in ['Adam', 'AdamW'] else RAdam
        fix_p, finetune_p, inter_p, fix_pnames, finetune_pnames, inter_pnames = [], [], [], [], [], []
        for name, p in self.model.named_parameters():
            if name in self.overloop_params_state.keys():
                if substrlsInString(name, config.synergy_finetunelayers):
                    finetune_p += [p]
                    finetune_pnames += [name]
                else:
                    p.requires_grad = False
                    fix_p += [p]
                    fix_pnames += [name]
            else:
                inter_p += [p]
                inter_pnames += [name]
        fix_dict = {'params': fix_p, 'lr': 0.0}
        finetune_dict = optimparam_dict(finetune_p, config, lr=config.fineturnlr)
        inter_dict = optimparam_dict(inter_p, config)
        optimizer = optim([fix_dict, finetune_dict, inter_dict])
        print('get synergy_optimizer end')
        return optimizer

    @abstractmethod
    def _train_epoch(self):
        pass

    @staticmethod
    def all_reduce(val):
        if torch.cuda.device_count() < 2:
            return val

        if not isinstance(val, torch.Tensor):
            val = torch.Tensor([val])
        # avg_tensor = hvd.allreduce(val)
        avg_tensor = torch.mean(val)
        return avg_tensor.item()

def optimparam_dict(paramls, config, lr=None, betas=(0.9, 0.999), eps=1e-08, weight_decay=None):
    if lr is None:  lr = config.lr
    if weight_decay is None:  weight_decay = config.weight_decay
    return {'params': paramls, 'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}


class DataLoaderBase(ABC):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_data_workers = config.num_data_workers

        self.train_set = None
        self.valid_set = None
        self.test_set = None

        self.train_sampler = None
        self.valid_sampler = None
        self.test_sampler = None

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    @abstractmethod
    def _init_datasets(self):
        pass

    def _init_samplers(self):
        self.train_sampler = RandomSampler(self.train_set)
        self.valid_sampler = RandomSampler(self.valid_set)
        self.test_sampler = RandomSampler(self.test_set)

    def _init_loaders(self):
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=self.num_data_workers,
                                       collate_fn=self.train_set.collate_wrapper,
                                       pin_memory=torch.cuda.is_available())
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


def overloop_items_dict(dictA, dictB):
    common_dict = {}
    diff_dict = {}
    for key in dictB.keys():
        if key in dictA:
            common_dict[key] = dictB[key]
        else:
            diff_dict[key] = dictB[key]
    return common_dict, diff_dict


def average_dictValues(list_of_dicts):
    dict1 = list_of_dicts[0]
    average_dict = {}
    for key in dict1.keys():
        # 使用 torch.stack 将张量叠加在一起，然后计算平均值
        average_value = torch.stack([d[key] for d in list_of_dicts]).mean(dim=0)
        average_dict[key] = average_value
    return average_dict


def substrlsInString(string, substrLs):
    for substr in substrLs:
        if substr in string:
            return True
    return False
