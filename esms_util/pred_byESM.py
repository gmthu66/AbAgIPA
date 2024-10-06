"""environment: ESMFold OpenFold"""
import os
import os.path as osp
import torch
import esm
import glob
import biotite.structure.io as bsio
from os.path import basename
from Bio.PDB import PDBParser, PDBIO


def modify_pdb_chains(input_pdb, seqch_ls, modified_output_pdb=None):
    modified_output_pdb = input_pdb if modified_output_pdb is None else modified_output_pdb

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)
    
    # Variables for tracking current chain and residue count
    current_chain_id = None
    residue_count = 0
    
    chain_map = {}
    chain_order = 0

    for model in structure:
        for chain in model:
            # Rename the chain to 'H' or 'L' based on sequence
            if current_chain_id is None:
                current_chain_id = seqch_ls[0]
            elif chain.id != current_chain_id:
                chain_order += 1
                current_chain_id = seqch_ls[chain_order]
                residue_count = 0  # Reset residue count for each new chain
            
            # Assign new chain ID
            chain.id = current_chain_id
            residue_count = 1  # Start residue numbering from 1

            for residue in chain:
                residue.id = (residue.id[0], residue_count, residue.id[2])
                residue_count += 1

    # Saving the modified PDB structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(modified_output_pdb)

    print(f"Modified PDB file saved as {modified_output_pdb}")
    return modified_output_pdb


def predict_antibody_structure(fasta_file, esm_Abpdbs_dir, model):
    if osp.isdir(esm_Abpdbs_dir):
        obj_name = basename(fasta_file)
        output_pdb_file = osp.join(esm_Abpdbs_dir, obj_name.replace(".fasta", ".pdb"))
    else:
        output_pdb_file = fasta_file.replace(".fasta", ".pdb")
    # 读取 Fasta 文件
    with open(fasta_file, 'r') as f:
        lines = f.readlines()

    # 提取序列
    sequences = []
    chainls = []
    for line in lines:
        if line.startswith('>'):
            ch = line.strip('>').strip('\n')
            if ch not in chainls:
                chainls.append(ch)
            else:
                break
        if not line.startswith('>'):
            sequences.append(line.strip())

    # 将序列连接为单链，使用 ':' 分隔重链和轻链
    sequence = ':'.join(sequences)

    # 进行结构预测
    # if not osp.exists(output_pdb_file):
    with torch.no_grad():
        output = model.infer_pdb(sequence)
    # 将结果写入 PDB 文件
    with open(output_pdb_file, 'w') as f:
        f.write(output)
    modify_pdb_chains(output_pdb_file, chainls, output_pdb_file)
    # 读取并打印 pLDDT
    struct = bsio.load_structure(output_pdb_file, extra_fields=["b_factor"])
    print(f'Mean pLDDT: {struct.b_factor.mean()}')
    return output_pdb_file


def predict_structures_in_directory(fasta_dir, esm_Abpdbs_dir):
    # 获取所有 .fasta 文件路径
    fasta_files = glob.glob(os.path.join(fasta_dir, '*.fasta'))
    fasta_files = sorted(fasta_files)
    # 加载 ESMFold 模型
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    error_fastafs = []
    # 遍历每个 .fasta 文件并进行结构预测
    for fasta_file in fasta_files:
        # try:
        print(f'Processing {fasta_file}...')
        output_pdb_file = predict_antibody_structure(fasta_file, esm_Abpdbs_dir, model)
        print(f'Structure predicted and saved to {output_pdb_file}')
        # except:
        #     error_fastafs.append(error_fastafs)
    print('error preds fastas:', error_fastafs)


if __name__ == "__main__":
    # fasta_directory = '/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/SabDab/pdbs/temp'
    fasta_directory = '/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/esms_util/Abs'
    # esm_Abpdbs_dir = osp.join(osp.dirname(__file__), 'esm_Abpdbs')  # 替换为PDB 输出目录
    # predict_structures_in_directory(fasta_directory, esm_Abpdbs_dir)

    pre_embedAb_dir = '/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/esms_util/Abs/pre_embed'
    fasta_files = glob.glob(os.path.join(fasta_directory, '*.fasta'))
    for fastai in fasta_files:
        fname = basename(fastai).strip('.fasta')
        pdbobj = fname.split('_')[0]
        chls = fname.split('_')[1:]
        findstring = f'{pdbobj}_{"".join(chls)}'
        pt_files = []
        # 遍历目录并查找符合条件的文件
        for root, dirs, files in os.walk(pre_embedAb_dir):
            for f in files:
                if f.endswith('.pt') and findstring in f:
                    pt_files.append(os.path.join(root, f))
        if len(pt_files) > 0:
            for pti in pt_files:
                os.remove(pti)
    print('delete wrong preembed pts by ptfiles')
