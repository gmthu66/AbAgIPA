import torch
from model.interface import DataDirPath
ddp = DataDirPath()


def merge_model(tgtmodel, static_model, merge_number=1):
    if static_model == 'IgFold':
        static_model_paths = sorted(ddp.igfold_models)
    for i_ in range(merge_number):
        source_model = static_model_paths[i_]
        static_state = torch.load(source_model)
    print(f'merge {merge_number} {static_model} in My model end')
