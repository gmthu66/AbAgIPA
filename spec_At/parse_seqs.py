import pandas as pd
# import matplotlib.pyplot as plt
import os.path as osp
import Bio.PDB as PDB
import os
from utils.abnumber_ import chain_info
from utils.constants import ResnDICT


def plot_amino_acid_distribution(amino_acid_sequences, fig_path):
    # 将氨基酸序列列表转换为DataFrame
    amino_acid_sequences = [list(aaseq) for aaseq in amino_acid_sequences]
    df = pd.DataFrame(amino_acid_sequences)

    # 计算每个位置上氨基酸的频率分布
    amino_acid_frequencies = df.apply(lambda x: x.value_counts(normalize=True))
    amino_acid_frequencies = amino_acid_frequencies.fillna(0.0)
    most_frequent_letters = amino_acid_frequencies.idxmax().to_list()
    background_seq = ''.join(most_frequent_letters)
    # 绘制氨基酸频率分布图
    # plt.figure(figsize=(10, 6))
    # for amino_acid in amino_acid_frequencies.index:
    #     sizes = amino_acid_frequencies.loc[amino_acid] * 100  # 转换为百分比
    #     plt.scatter(amino_acid_frequencies.columns, [amino_acid] * len(amino_acid_frequencies.columns),
    #                 s=sizes, alpha=0.7, label=amino_acid)

    # plt.xlabel('氨基酸位置索引')
    # plt.ylabel('氨基酸')
    # plt.title('氨基酸频率分布图')
    # plt.legend()
    # plt.savefig()
    # plt.show()
    return background_seq


def get_chain_info(aaseq):
    return chain_info(aaseq)


def get_overloop_range(static_seq, tgtseq):
    for i_ in range(len(tgtseq), 0, -1):
        if static_seq in tgtseq[i_:]:
            return [i_, i_ + len(static_seq)]
    return [None, None]


def aaseqDict2fasta(seq_dict, fasta_path, exist_ok=False):
    if ((not osp.exists(fasta_path)) and exist_ok) or (not exist_ok):
        with open(fasta_path, "w") as f:
            for chain, seq in seq_dict.items():
                f.write(">{}\n{}\n".format(chain, seq,))


def add_aaseqDict2fasta(seq_dict, fasta_path, exist_ok=False):
    with open(fasta_path, "a") as f:
        for chain, seq in seq_dict.items():
            f.write(">{}\n{}\n".format(chain, seq,))


def get_amino_acid_sequence(pdb_file, ch_ls=None):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    amino_acids = []
    # for model in structure:
    model = structure[0]
    ch_ls = [ch.id for ch in model] if ch_ls is None else ch_ls
    for chain in model:
        if chain.id in ch_ls:
            for residue in chain:
                if PDB.is_aa(residue):
                    amino_acids.append(residue.resname)

    sequence = "".join([ResnDICT[resn] for resn in amino_acids])
    return sequence


def proteins_contain_each_other(pdb_file1, pdb_file2):
    sequence1 = get_amino_acid_sequence(pdb_file1)
    sequence2 = get_amino_acid_sequence(pdb_file2)

    contains_each_other = sequence1 in sequence2 or sequence2 in sequence1
    if contains_each_other is False:
        contains_each_other = longest_common_sentence(sequence1, sequence2, trunc=100)
    return contains_each_other


def longest_common_sentence(s1, s2, trunc=100):
    m, n = len(s1), len(s2)    
    # 创建一个二维数组用于存储最长公共子串的长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0  # 记录最长的连续相同句子字符串长度
    # 填充二维数组
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0
    return max_length > trunc


def VHVL_At_chainFromSeries(info_i):
    """输入Series数据或者对应的dict数据, 返回VHVL_At信息"""
    if not isinstance(info_i, dict):
        info_i = info_i.to_dict()
    ab_chs = ''.join(eval(info_i['sudo_abchains'])).replace('chain ', '')
    at_chs = ''.join(eval(info_i['sudo_atchains'])).replace('chain ', '')
    pdb_obj = info_i['parent_id']
    rec_i = f'{pdb_obj}_{ab_chs}_{at_chs}'
    return rec_i
