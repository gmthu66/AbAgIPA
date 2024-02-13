import os
import sys
import os.path as osp
import pymol2 as pm2
from os.path import basename
from Bio import PDB

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
sys.path.append(project_dir)

from database.inter_pdb import PDBparser_modify_chids, residue_backbone_coords, save_residuels2pdb
from database.preprocess.handle_interpdb import save_chain_to_file, modify_chain_id
from database.parse_utils import PDB_pre, has_case_insensitive_keys_values
from database.inter_pdb import save_residuels2pdb
from utils.abnumber_ import renumber_pdb, IsTwoAbSeq
from spec_At.parse_seqs import proteins_contain_each_other, get_amino_acid_sequence


def get_residues_Bychains(pdb_file_path, ch_ls=['H', 'L'], idmap=None):
    # 创建 PDB 解析器
    parser = PDB.PDBParser(QUIET=True)
    # 解析 PDB 文件
    structure = parser.get_structure("structure", pdb_file_path)
    structure = structure[0]
    chids = [ch.id for ch in structure]
    chain_list = [ch for ch in structure]
    # if idmap is None:
    #     if 'A' in chids or 'B' in chids:
    #         idmap = {'A': 'H', 'B': 'L', 'H': 'H', 'L': 'L'}
    chain_list = PDBparser_modify_chids(chain_list, idmap)
    # 获取抗体的坐标
    antibody_coords, antibody_residuels = [], []
    for chain in chain_list:
        if chain.id in ch_ls:
            for residue in chain:
                residue_coords = residue_backbone_coords(residue)
                if len(residue_coords) > 0:
                    antibody_coords.extend(residue_coords)
                    antibody_residuels.append(residue)
    return antibody_residuels, antibody_coords


def get_residues_BychNotAlterChainId(pdb_file_path, ch_ls=['H', 'L']):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file_path)
    structure = structure[0]
    chids = [ch.id for ch in structure]
    chain_list = [ch for ch in structure]
    res_ls = []
    for chain in chain_list:
        if chain.id in ch_ls:
            for residue in chain:
                residue_coords = residue_backbone_coords(residue)
                if len(residue_coords) > 0:
                    res_ls.append(residue)
    return res_ls


def get_chls(merge_path):
    # 创建 PDB 解析器
    parser = PDB.PDBParser(QUIET=True)
    # 解析 PDB 文件
    structure = parser.get_structure("structure", merge_path)
    structure = structure[0]
    chids = [ch.id for ch in structure]
    return chids


def save_and_modify(affine_dict, inter_f, ch, ch2, temp_dir):
    obj_ = basename(inter_f).split('.')[0]
    original_ch_path = osp.join(temp_dir, obj_ + ch + ".pdb")
    save_chain_to_file(inter_f, ch, original_ch_path)
    out_pdb = osp.join(temp_dir, obj_ + '_' + ch2 + ".pdb")
    modify_chain_id(original_ch_path, original_chain_id=ch, new_chain_id=ch2, output_file=out_pdb)
    os.remove(original_ch_path)
    affine_dict[ch] = ch2
    return out_pdb


def ch_affine(HL_chs, At_chs, merge_path, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    char_list = sorted(list(set([chr(ord('C') + i) for i in range(24)]) - set(['H', 'L'])))
    total_ls = get_chls(merge_path)
    char_list = sorted(list(set(char_list) - set(total_ls)))
    if len(char_list) < 1:
        char_list = [chr(ord('a') + i) for i in range(26)]
        char_list = sorted(list(set(char_list) - set(total_ls)))
    at_affine_dict, ab_affine_dict, Abch_pdbs, Atch_pdbs = {}, {}, [], []

    ab_j, at_j = 0, 0
    for i_, ch in enumerate(total_ls):
        if ch in HL_chs:
            ab_j = HL_chs.index(ch)
            out_pdb = save_and_modify(ab_affine_dict, merge_path, ch, ['H', 'L'][ab_j], temp_dir)
            # ab_j += 1
            Abch_pdbs.append(out_pdb)
        elif ch in At_chs:
            out_pdb = save_and_modify(at_affine_dict, merge_path, ch, char_list[at_j], temp_dir)
            at_j += 1
            Atch_pdbs.append(out_pdb)
    return ab_affine_dict, at_affine_dict, Abch_pdbs, Atch_pdbs


def merge_chs2pdb(input_files, affine_dict, temp_dir, output_file=None, nm=None):
    if isinstance(affine_dict, dict) and output_file is None:
        pdb_obj = basename(input_files[0]).split('_')[0]
        ch_ls = affine_dict.values()
        output_file = osp.join(temp_dir, "{}_{}.pdb".format(pdb_obj, '_'.join(ch_ls)))
    if nm == 'Ab' and has_case_insensitive_keys_values(affine_dict):
        if proteins_contain_each_other(input_files[0], input_files[1]):
            Hseq, Lseq = resplit_HLpdbs(input_files[0], input_files[1])
        merge_pdbs(input_files, output_file)
        pdbfile_HETATM2ATOM(output_file)
    else:
        merge_pdbs(input_files, output_file)
    print(f'merge_pdbs to {output_file}')
    if nm == 'Ab':
        pdb_preer = PDB_pre()
        pdb_preer.trunc_pdb(output_file)
        try:
            renumber_pdb(output_file)
            print('truncate and renumber Ab/Nb pdbfile by chothia rule:\n\t', output_file)
        except:
            print('truncate Ab/Nb pdbfile by chothia rule, but renumber Fail:\n\t', output_file)
    return output_file


def merge_pdbs(input_files, output_file):
    # 创建一个 PDBParser
    parser = PDB.PDBParser(QUIET=True)
    # 创建一个 PDBIO 对象用于写入合并后的 PDB 文件
    io = PDB.PDBIO()
    # 创建一个空的结构对象，用于存储合并后的结构
    merged_structure = PDB.Structure.Structure("merged_structure")
    merged_resls = []

    # 遍历所有输入文件，解析并合并结构
    for input_file in input_files:
        # 解析单个 PDB 文件
        structure = parser.get_structure("temp", input_file)
        # 遍历结构中的模型、链和原子，并将它们添加到合并后的结构中
        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue):  merged_resls.append(residue)
    # 将合并后的结构写入输出文件
    save_residuels2pdb(merged_resls, output_file)


def clean_pdbfile(pdb_path):
    pm = pm2.PyMOL()
    pm.start()
    pm.cmd.load(pdb_path)
    pm.cmd.remove('solvent')
    pm.cmd.remove('HETATM')  # 删除HETATM开头的数据行(删除小分子)
    pm.cmd.save(pdb_path, selection='all')
    pm.stop()


def modify_residue_and_parents(pdb_file, residue_number, new_chain_id):
    # Create a structure object from the PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    # Helper function to recursively modify parent IDs
    def modify_parents(entity):
        # Check if the entity has a parent
        if hasattr(entity, 'parent'):
            # Modify the ID of the current entity
            entity.id = (new_chain_id, entity.id[1], entity.id[2])
            # Recursively modify the parent
            modify_parents(entity.parent)


def create_chain_with_new_id(residue_list, new_chain_id):
    # Create a structure object from the PDB file
    parser = PDB.PDBParser(QUIET=True)
    new_chain = PDB.Chain.Chain(new_chain_id)  # 指定创建的是一个氨基酸链
    # Add residues to the new chain
    for residue in residue_list:
        new_chain.add(residue)
    return new_chain


def pdbfile_HETATM2ATOM(pdbfile):
    with open(pdbfile, 'r') as infile:
        lines = infile.readlines()
    modified_lines = []
    for line in lines:
        if line.startswith('HETATM'):
            # 将'HETATM'替换为'ATOM'
            line = line.replace('HETATM', 'ATOM  ', 1)
            # 如果'HETATM'后的字符数不足，用空格填充
            line = line.ljust(len(lines[0]))
        modified_lines.append(line)
    # Save the modified content to a new file
    with open(pdbfile, 'w') as outfile:
        outfile.writelines(modified_lines)


def split_byReslsIndex(residues, ee, pdb1, pdb2, ch1, ch2):        
    residues_1 = residues[:ee]
    residues_2, st = residues[ee:], 0
    for residue in residues_2:
        st += 1
        residue.id = (ch2, st, residue.id[2])
        residue.full_id = (residue.full_id[0], residue.full_id[1], ch2, residue.id)
    new_chain2 = create_chain_with_new_id(residues_2, ch2)
    residues_2 = [res for res in new_chain2]
    save_residuels2pdb(residues_1, pdb1)
    save_residuels2pdb(residues_2, pdb2)


def resplit_HLpdbs(Hpdb, Lpdb):
    Hch, Lch = Hpdb.split('.')[0][-1], Lpdb.split('.')[0][-1]
    init_Hseq = get_amino_acid_sequence(Hpdb, ch_ls=[Hch])
    Hee, Hinfo1, Hinfo2 = IsTwoAbSeq(init_Hseq)
    if Hee is not None:
        H_residues = get_residues_BychNotAlterChainId(Hpdb, [Hch])
        split_byReslsIndex(H_residues, Hee, Hpdb, Lpdb, Hch, Lch)
        Hseq, Lseq = init_Hseq[:Hee], init_Hseq[Hee:]
        print('resplit_HLpdbs end')
    else:
        init_Lseq = get_amino_acid_sequence(Lpdb, ch_ls=[Lch])
        Lee, Linfo1, Linfo2 = IsTwoAbSeq(init_Lseq)
        if Lee is not None:
            L_residues = get_residues_BychNotAlterChainId(Lpdb, [Lch])
            split_byReslsIndex(L_residues, Lee, Lpdb, Hpdb, Lch, Hch)
            Lseq, Hseq = init_Hseq[:Lee], init_Hseq[Lee:]
            print('resplit_HLpdbs end')
    return Lseq, Hseq
