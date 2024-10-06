import os.path as osp
from Bio import PDB
from typing import List, Optional, SupportsFloat
from Bio.PDB.Residue import Residue
now_dir = osp.dirname(__file__)


class ResidueAb(object):
    def __init__(self, residue: Optional[Residue] = None, dist_with_Ab: Optional[SupportsFloat] = None):
        self.residue = residue
        self.dist_with_Ab = dist_with_Ab


def esm_pdb2resls(pdb_file_path, antibody_chains, true_pdb_resls=None):
    right_flg = True
    parser = PDB.PDBParser(QUIET=True)
    # 解析 PDB 文件
    structure = parser.get_structure("structure", pdb_file_path)
    structure, atinterface = structure[0], True
    antibody_coords, antibody_residuels = [], []
    for chain in structure:
        if chain.id in antibody_chains:
            for residue in chain:
                residue_coords = residue_backbone_coords(residue)
                if len(residue_coords) > 0:
                    antibody_coords.extend(residue_coords)
                    antibody_residuels.append(residue)
    if len(antibody_residuels) < 1:  return None, None, None, None, None
    if len(true_pdb_resls) != len(antibody_residuels):
        right_flg = False
        antibody_residuels = [o.residue for o in true_pdb_resls]
        with open(osp.join(now_dir, 'notequal_length.txt'), 'a') as w:
            w.write('>{} true_pdb_resls len:{} esm_ab_resls len:{}\n'.format(pdb_file_path, len(true_pdb_resls), len(antibody_residuels)))
    # assert len(true_pdb_resls) == len(antibody_residuels)  # 当长度不一样时需要记录
    assert len(true_pdb_resls) < 350
    residueAb_ls = []
    for i_, res in enumerate(antibody_residuels):
        abres_coords = residue_backbone_coords(res)
        if true_pdb_resls is None:
            disti = 4.0
        else:
            disti = true_pdb_resls[i_].dist_with_Ab
        residueAb = ResidueAb(residue=res, dist_with_Ab=disti)
        residueAb_ls.append(residueAb)
    return residueAb_ls, right_flg


def residue_backbone_coords(residue, keyatoms=["N", "CA", "C"]):
    residue_coords = []
    if PDB.is_aa(residue):
        for atom_name in keyatoms:
            try:
                atom = residue[atom_name]
                residue_coords.append(atom.get_coord())
            except KeyError:  # 如果某个原子不存在，则跳过
                pass
    return residue_coords
