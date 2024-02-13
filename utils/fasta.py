import os.path as osp
from Bio import SeqIO, pairwise2, PDB
from Bio.PDB import PDBParser


def get_fasta_chain_seq(
    fasta_file,
    chain_id,
    check_ch=False,
):
    if not osp.exists(fasta_file):
        return None, []
    obj_ = SeqIO.parse(fasta_file, 'fasta')
    ch_ls = [r_.id for r_ in obj_]
    if check_ch:
        return None, ch_ls

    ch_dict = {"A": "H", "B": "L", "H": "H", "L": "L"} if (chain_id not in ch_ls) else None  # 如果chain_id不在pdb文件中，用ch_dict对pdb_id进行一次映射
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        if ch_dict is None:
            if ":{}".format(chain_id) in chain.id or chain.id in ":{}".format(chain_id):
                return str(chain.seq), ch_ls
        else:
            if ":{}".format(chain_id) in ch_dict[chain.id] or ch_dict[chain.id] in ":{}".format(chain_id):
                return str(chain.seq), ch_ls
    print(f'Error get_fasta_chain_seq:\t{fasta_file}\n fasta chain_id:{chain_id} not in pdb file, affined chain_id to ch_dict but return aaseq is None',)
    return None, ch_ls


def get_fasta_chain_dict(fasta_file):
    seq_dict = {}
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        seq_dict[chain.id] = str(chain.seq)
    return seq_dict


def pairwise_align(
    seq1,
    seq2,
):
    ###
    # Aligns two sequences using the Needleman-Wunsch algorithm
    # Returns alignment of seq2 into seq1
    ###
    ali = pairwise2.align.globalxx(
        seq1,
        seq2,
    )[0]
    ali_list = []
    seq1_i, seq2_i = 0, 0
    for ali_seq in ali.seqB.split("-"):
        if len(ali_seq) == 0:
            seq1_i += 1
        else:
            l = len(ali_seq)
            ali_list.append((seq1_i, seq1_i + l, seq2_i, seq2_i + l))
            seq1_i += l

    return ali_list


def get_chainseq_FromPDB(pdbf, tgt_chain):
    # 创建PDBParser对象
    parser = PDBParser(QUIET=True)

    try:
        # 解析PDB文件
        structure = parser.get_structure('protein', pdbf)
    except FileNotFoundError:
        print(f"Error: File {pdbf} not found.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None
    # 存储氨基酸序列的列表
    amino_acid_sequence = []
    model = structure[0]
    for chain in model:
        # 检查chain id是否匹配
        if chain.id == tgt_chain:
            # 遍历每个残基
            for residue in chain:
                # 检查残基是否为氨基酸
                if PDB.is_aa(residue):
                # if residue.id[0] == ' ' and residue.id[2] in {' ', 'A', 'C', 'G', 'T', 'U'}:
                    # 获取氨基酸的三字母代码
                    aa_code = residue.get_resname()
                    amino_acid_sequence.append(aa_code)
    aaseq = ''.join(amino_acid_sequence)
    return aaseq
