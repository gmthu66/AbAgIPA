import numpy as np
import copy
import torch
import torch.nn.functional as F
from itertools import chain, product


AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
DP = list(product(AA, AA))
DP_list = []
for i in DP:
    DP_list.append(str(i[0]) + str(i[1]))


def returnCKSAAPcode(query_seq, k):
    if 'O' in query_seq:
        query_seq = query_seq.replace('O', 'K')
    code_final = []
    for turns in range(k + 1):
        DP_dic = {}
        code = []
        code_order = []
        for i in DP_list:
            DP_dic[i] = 0
        for i in range(len(query_seq) - turns - 1):
            tmp_dp_1 = query_seq[i]
            tmp_dp_2 = query_seq[i + turns + 1]
            tmp_dp = tmp_dp_1 + tmp_dp_2
            if tmp_dp in DP_dic.keys():
                DP_dic[tmp_dp] += 1
            else:
                DP_dic[tmp_dp] = 1
        for i, j in DP_dic.items():
            code.append(j / (len(query_seq) - turns - 1))
        # for i in AAindex_list:
        #     code_order.append(code[DP_list.index(i)])
        # code_final+=code_order
        code_final += code
    return code_final


def CKSAAP_encoding(seq, k):
    # 将氨基酸序列转换为数字序列
    aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'O': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    num_seq = [aa_dict[aa] for aa in seq]

    # 统计K-spaced氨基酸对出现的频率
    freq_dict = {}
    for i in range(len(seq)-k):
        pair = tuple(sorted([num_seq[i], num_seq[i+k]]))
        if pair not in freq_dict:
            freq_dict[pair] = 0
        freq_dict[pair] += 1

    # 计算每个K-spaced氨基酸对的频率向量
    vector = []
    for i in range(20):
        for j in range(i, 20):
            pair = tuple(sorted([i, j]))
            if pair in freq_dict:
                vector.append(freq_dict[pair])
            else:
                vector.append(0)

    # 将所有K-spaced氨基酸对的频率向量连接起来，得到CKSAAP特征向量
    return vector


def OneHot_residuetype(aa_seq, nbin=20):
    # alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV"), dtype='|S1').view(np.uint8)
    # aa_seq_ini = copy.deepcopy(aa_seq)
    aa_seq_ini_int = np.array(list(aa_seq), dtype='|S1').view(np.uint8)
    aa_seq = copy.deepcopy(aa_seq_ini_int)
    for i_ in range(alphabet.shape[0]):
        # chr_str = chr(alphabet[i_])
        aa_seq[aa_seq == alphabet[i_]] = i_
    # aa_seq[aa_seq > 20] = 20
    aa_seq = torch.tensor(data=aa_seq, dtype=torch.int64)
    aa_seq_onehot = F.one_hot(aa_seq, num_classes=nbin).to(torch.float32)
    return aa_seq_onehot
