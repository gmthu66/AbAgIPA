import torch
import os.path as osp
from os.path import basename
from model.interface import DataDirPath
from utils.general import  exists
from spec_At.Base_train_nfold import average_dictValues, overloop_items_dict
from training.record import clean_config
from spec_At.data_code.coords_model import FAbAtInterPhysic
ddp = DataDirPath()


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
