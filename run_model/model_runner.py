import os
import os.path as osp
import torch
import sys
import numpy as np
from time import time
from glob import glob
from os.path import basename

sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
import abatInter_SCA
from abatInter_SCA.model import *
from abatInter_SCA.utils.general import exists
from abatInter_SCA.run_model.intering import infer_inters
from abatInter_SCA.model.FAbAtInter_Glbmean import FAbAtInter as Inter_Glbmean
from abatInter_SCA.model.FAbAtInter_SoftAtten import FAbAtInter as Inter_SoftAtten
from abatInter_SCA.model.interface import PROJECT_DIR, DataDirPath
from abatInter_SCA.training.record import clean_config
Model_Dict = {'FAbAtInter_SoftAtten': Inter_SoftAtten, 'FAbAtInter_GlbMean': Inter_Glbmean}


class ModelRunner():
    def __init__(self, model_name=None, num_models=4, model_pts=None, device=None, try_gpu=True, out_dir=None) -> None:

        model_conifg, config_pt = init_model_cofig(out_dir)
        model_conifg.out_dir = out_dir
        os.makedirs(model_conifg.out_dir, exist_ok=True)
        Model = Model_Dict[model_name]

        if exists(model_pts):
            num_models = len(model_pts)
        else:
            if num_models < 1 or num_models > 4:
                raise ValueError("num_models must be between 1 and 4.")

            if not exists(model_pts):
                project_path = os.path.dirname(
                    os.path.realpath(abatInter_SCA.__file__)
                )
                pt_path = os.path.join(
                    project_path,
                    f"trained_models/{model_name}/*.pt",)
                model_pts = list(glob(pt_path))

        model_pts = list(sorted(model_pts))[:num_models]
        print(f"Loading {num_models} IgFold models...")

        device = torch.device("cuda:0" if torch.cuda.is_available() and try_gpu else "cpu") if device is None else device
        print(f"Using device: {device}")

        self.models = []
        for pt_file in model_pts:
            checkpoint = torch.load(pt_file, map_location='cpu')
            print(f"Loading {pt_file}...")
            model_now = init_from_configpt(Model, config_pt, user_config=model_conifg)
            model_now.load_state_dict(checkpoint['model'])
            self.models.append(model_now.eval().to(device))
        print(f"Successfully loaded {num_models} IgFold models.")

    def infer_inters(self, data_in):
        ":param sequences: Dictionary of sequences."
        y_true, y_prob = [], []
        start_time = time()
        for model in self.models:
            model_out = model(data_in)  # grad_fn表明该变量是怎么来的，用于指导反向传播
            y_prob.extend(probability_classify(model_out.inter_prob.cpu().numpy()))
        y_true.extend(data_in.inter_label.cpu().numpy())
        print(f"Completed folding in {time() - start_time:.2f} seconds.")
        return y_prob[0], y_true[0]


def probability_classify(y_probs):
    exp_x = np.exp(y_probs)
    y_probs = (exp_x / np.sum(exp_x, axis=1, keepdims=True)) if (y_probs.sum() != y_probs.shape[0]) else y_probs
    return y_probs

def init_from_configpt(Model, config_pt, user_config=None):
    inter_criterion = torch.nn.CrossEntropyLoss()
    if 'config' not in basename(config_pt):
        if exists(config_pt):
            # num_models = len(model_ckpt)
            print(f"Loading {config_pt}...")
            torch.load(config_pt)
            try:
                model = Model(config=user_config, inter_criterion=inter_criterion)
                model = Model.load_state_dict(config_pt)
                print(f"Successfully loaded Inter model params.")
            except:
                print(f"Failed loaded Inter model params. Construct model by cofigs in ckpt...")
                config_pt = torch.load(config_pt)['hyper_parameters']['config']
                config_pt = clean_config(config_pt, config=user_config)
                torch.save(config_pt, osp.join(user_config.out_dir, 'config_ckpt.pt'))
                model = Model(config=config_pt, inter_criterion=inter_criterion)
    else:
        config_pt = config_pt
        config_pt = torch.load(config_pt)
        config_pt = clean_config(config_pt, user_config=user_config)
        model = Model(config=config_pt, inter_criterion=inter_criterion)
    return model


def init_model_cofig(out_dir):
    config = Namespace
    config = init_config(config)
    ddp = DataDirPath()
    config_pt = osp.join(ddp.model_cofig_dir, 'config_ckpt.pt')
    if not exists(config_pt):
        project_path = PROJECT_DIR
        ckpt_path = os.path.join(project_path, "trained_models/abatInter_SCA/*.ckpt",)
        config_pt = sorted(list(glob(ckpt_path)))
    if isinstance(config_pt, list):  config_pt = config_pt[0]
    return config, config_pt


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
    config.par_out_dir = osp.join(PROJECT_DIR, 'trained_models/FAbAtInter_GlbMean')
    config.run_name = 'train_test'
    config.verbose = True
    config.inter_attn_depth = 2
    return config


class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
