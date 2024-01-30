import torch
import os
import os.path as osp
import pickle
from os.path import basename
from dataclasses import dataclass
from typing import List, Optional, Union, AnyStr
from collections import defaultdict
project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
PROJECT_DIR = project_dir


@dataclass
class IgFoldInput():
    """
    Input type of for IgFold model.
    """

    Ab_embeddings: List[List[torch.FloatTensor]]
    Ab_attentions: List[List[torch.FloatTensor]]
    Ab_seqlens: List[int]
    Ab_VHLlens: List[List[int]]
    Ab_template_coords: Optional[torch.FloatTensor] = None
    Ab_template_mask: Optional[torch.BoolTensor] = None
    Ab_batch_mask: Optional[torch.BoolTensor] = None
    Ab_align_mask: Optional[torch.BoolTensor] = None
    Ab_coords_label: Optional[torch.FloatTensor] = None
    return_embeddings: Optional[bool] = False
    fpaths : Optional[List[AnyStr]] = None

    At_embeddings: List[List[torch.FloatTensor]] = None
    At_attentions: List[List[torch.FloatTensor]] = None
    At_coords_label: Optional[torch.FloatTensor] = None
    Ab_cdr_indexs: Optional[List[int]] = None
    At_batch_mask: Optional[torch.BoolTensor] = None
    At_len : int = None
    inter_label : Optional[torch.IntTensor] = None


@dataclass
class SingleDataSamp():
    """
    Input type of for IgFold model.
    """
    Ab_embedding: torch.FloatTensor
    Ab_attention: torch.FloatTensor
    Ab_seqlen: List[int]
    Ab_VHLlen: List[int]
    Ab_maxlen: Optional[int] = None
    Ab_temp_coords: Optional[torch.FloatTensor] = None
    Ab_temp_mask: Optional[torch.BoolTensor] = None
    Ab_batch_mask: Optional[torch.BoolTensor] = None
    Ab_align_mask: Optional[torch.BoolTensor] = None
    Ab_coords_label: Optional[torch.FloatTensor] = None
    Ab_cdr_indexs: Optional[List[int]] = None
    Ab_mut_range: Optional[List[int]] = None
    return_embeddings: Optional[bool] = False
    fpaths : Optional[List[AnyStr]] = None

    At_embeddings: List[List[torch.FloatTensor]] = None
    At_attentions: List[List[torch.FloatTensor]] = None
    At_coords_label: Optional[torch.FloatTensor] = None
    At_batch_mask: Optional[torch.BoolTensor] = None
    At_len : int = None
    inter_label : Optional[torch.IntTensor] = None


@dataclass
class PhysicDataSamp():
    """
    Input type of for IgFold model.
    """
    Ab_phisicEmbed: torch.FloatTensor
    Ab_phisicAttention: torch.FloatTensor
    Ab_seqlen: List[int]
    Ab_VHLlen: List[int]
    Ab_maxlen: Optional[int] = None
    Ab_embedding: Optional[torch.FloatTensor] = None
    Ab_attention: Optional[torch.FloatTensor] = None
    Ab_temp_coords: Optional[torch.FloatTensor] = None
    Ab_temp_mask: Optional[torch.BoolTensor] = None
    Ab_batch_mask: Optional[torch.BoolTensor] = None
    Ab_align_mask: Optional[torch.BoolTensor] = None
    Ab_coords_label: Optional[torch.FloatTensor] = None
    Ab_pred_coords: Optional[torch.FloatTensor] = None
    Ab_cdr_indexs: Optional[List[int]] = None
    Ab_mut_range: Optional[List[int]] = None
    return_embeddings: Optional[bool] = False
    fpaths : Optional[List[AnyStr]] = None

    At_embeddings: List[List[torch.FloatTensor]] = None
    At_attentions: List[List[torch.FloatTensor]] = None
    At_coords_label: Optional[torch.FloatTensor] = None
    At_batch_mask: Optional[torch.BoolTensor] = None
    At_len : Optional[int] = None
    inter_label : Optional[torch.IntTensor] = None
    Ab_abatid : Optional[AnyStr] = None


@dataclass
class InterFoldOutput():
    """
    Output type of for IgFold model.
    """

    coords: Optional[torch.FloatTensor] = None
    prmsd: Optional[torch.FloatTensor] = None
    translations: Optional[torch.FloatTensor] = None
    rotations: Optional[torch.FloatTensor] = None
    coords_loss: Optional[torch.FloatTensor] = None
    torsion_loss: Optional[torch.FloatTensor] = None
    bondlen_loss: Optional[torch.FloatTensor] = None
    prmsd_loss: Optional[torch.FloatTensor] = None
    inter_prob: Optional[torch.FloatTensor] = None
    inter_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    bert_embs: Optional[torch.FloatTensor] = None
    bert_attn: Optional[torch.FloatTensor] = None
    gt_embs: Optional[torch.FloatTensor] = None
    structure_embs: Optional[torch.FloatTensor] = None

    Atres_attn: Optional[torch.FloatTensor] = None
    Abres_attn: Optional[torch.FloatTensor] = None
    Atpoint_attn: Optional[torch.FloatTensor] = None
    Abpoint_attn: Optional[torch.FloatTensor] = None


@dataclass
class IgFoldOutput():
    """
    Output type of for IgFold model.
    """

    coords: torch.FloatTensor
    prmsd: torch.FloatTensor
    translations: torch.FloatTensor
    rotations: torch.FloatTensor
    coords_loss: Optional[torch.FloatTensor] = None
    torsion_loss: Optional[torch.FloatTensor] = None
    bondlen_loss: Optional[torch.FloatTensor] = None
    prmsd_loss: Optional[torch.FloatTensor] = None
    inter_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    bert_embs: Optional[torch.FloatTensor] = None
    bert_attn: Optional[torch.FloatTensor] = None
    gt_embs: Optional[torch.FloatTensor] = None
    structure_embs: Optional[torch.FloatTensor] = None


class DataDirPath():
    def __init__(self):
        self.sabdab_summtsv = '/data/gm_data/AbAtInteraction/AbAtIPA/init_data/sabdab_summary_all.tsv'
        self.only_struct_InferDir = '/data/gm_data/AbAtInteraction/AbAtIPA_data'

        self.sabdab_dir = '/data/gm_data/databaseMSA/sabdab/all_structures/chothia'
        self.af_oas_paired = '/home/user/protein/IgFold_data/af_oas_paired'
        self.af_oas_unpaired = '/home/user/protein/IgFold_data/af_oas_unpaired'
        self.sabdab_only_Ab = '/home/user/protein/IgFold_data/sabdab'
        self.test_dirs = '/home/user/protein/IgFold_data/test'
        # self.af_oas_paired = '/data/gm_data/IgFold_data/af_oas_paired'
        # self.af_oas_unpaired = '/data/gm_data/IgFold_data/af_oas_unpaired'
        # self.sabdab_only_Ab = '/data/gm_data/IgFold_data/sabdab'
        # self.test_dirs = '/data/gm_data/IgFold_data/test'

        only_struct_dir = '/data/gm_data/IgFold_data'
        self.error_dict_path, self.right_dict_path = osp.join(only_struct_dir, 'pdb_error_dict.pkl'), osp.join(only_struct_dir, 'pdb_right_dict.pkl')
        self.inter_noninter_dict_info = {'inter': osp.join(project_dir, 'init_data', 'inter_df.tsv'), 'noninter': osp.join(project_dir, 'init_data', 'noninter_df.tsv')}
        self.model_cofig_dir = osp.join(project_dir, 'abatInter_SCA/config')

        for d in [self.af_oas_paired, self.af_oas_unpaired, self.sabdab_only_Ab, self.test_dirs]:
            os.makedirs(osp.join(d, 'pre_embed'), exist_ok=True)

        self.init_pdbdir = '/data/gm_data/AbAtInteraction/AbAtIPA/init_data/pdbs'
        self.dock_dir = '/home/user/protein/IgFold_data'
        # self.dock_dir = '/data/gm_data'
        os.makedirs(self.init_pdbdir, exist_ok=True)

    @property
    def igfold_models(self):
        trained_igfold_models_dir = '/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/spec_At/share_parmModel'
        return [osp.join(trained_igfold_models_dir, fpath) for fpath in os.listdir(trained_igfold_models_dir) if fpath.endswith('.ckpt')]

    @property
    def train_dirs(self):
        return [self.af_oas_paired, self.af_oas_unpaired, self.sabdab_only_Ab]

    def pred_embed_file(self, fpath):
        for dir_ in [self.af_oas_paired, self.af_oas_unpaired, self.sabdab, self.test_dirs]:
            if dir_ in fpath:
                return osp.join(dir_, 'pre_embed/{}.npz'.format(basename(fpath).split('.')[0]))


class APP_envs(object):
    def __init__(self):
        self.hdock = '/data/gm_data/AbAtInteraction/rely_app/HDOCKlite/hdock'
        self.CD_HIT_Dir = '/usr/lib/cd-hit'


def save_dict(dic, f_path):
    with open(f_path, 'wb') as f:
        pickle.dump(dic, f)


def read_dict(f_path):
    with open(f_path, 'rb') as f:
        dic = pickle.load(f)
    return dic


class PRO_Constant():
    def __init__(self):
        pass

    @property
    def ser2aa_dict(self):
        dd = defaultdict(str)
        DICT = {
            'ALA': 'A',
            'ARG': 'R',
            'ASN': 'N',
            'ASP': 'D',
            'CYS': 'C',
            'GLN': 'Q',
            'GLU': 'E',
            'GLY': 'G',
            'HIS': 'H',
            'ILE': 'I',
            'LEU': 'L',
            'LYS': 'K',
            'MET': 'M',
            'PHE': 'F',
            'PRO': 'P',
            'SER': 'S',
            'THR': 'T',
            'TRP': 'W',
            'TYR': 'Y',
            'VAL': 'V',
            'ASX': 'N',
            'GLX': 'Q',
            'UNK': 'G',
            'HSD': 'H',
            'PCA': '',
            'GLC': '',
            'DG': '',
            'DC': '',
        }
        dd.update(DICT)
        return dd
