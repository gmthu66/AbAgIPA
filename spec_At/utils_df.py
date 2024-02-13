import numpy as np
import re
import pandas as pd


def string_to_dict(mat_dict_str):
    """string of ndarray 2 ndarray dict"""
    if isinstance(mat_dict_str, list):
        if isinstance(mat_dict_str[0], str):
            mat_dict_str = ''.join(mat_dict_str)
    string_ls = mat_dict_str.split('])')
    # 创建一个空字典
    mat_dict = {}
    for linei in string_ls:
        matches = re.findall(r"'(.*?)'", linei)
        steds = [int(num) for num in re.findall(r'\b\d+\b', linei)]
        if len(matches) == 1:
            k_ = matches[0]
            mat_dict[k_] = np.array(steds).reshape((-1, 2))
    return mat_dict


def remove_unnamed_columns(df):
    # 使用 filter 函数选择不含有 "Unnamed" 的列
    filtered_columns = df.filter(regex='^(?!Unnamed).*$')    
    # 创建一个新的 DataFrame，包含不含有 "Unnamed" 的列
    new_df = df[filtered_columns.columns]
    return new_df


def ratio_minidf(data_df, PosNegLabel=None, ratio=None):
    pos_df = data_df[data_df[PosNegLabel] == 1]
    pos_df = pos_df.sample(frac=ratio, random_state=22)
    neg_df = data_df[data_df[PosNegLabel] == 0]
    neg_df = neg_df.sample(frac=ratio, random_state=22)
    data_df = pd.concat([pos_df, neg_df], ignore_index=True)
    data_df = data_df.sample(frac=1.0, random_state=22)
    return data_df


def add_concatenated_length_column(df, key1, key2, trunc=260):
    # 使用 apply 函数，对每一行的"Hseq"和"Lseq"执行拼接并计算长度
    df['concat_length'] = df.apply(lambda row: len(row[key1] + row[key2]), axis=1)
    df_filtered = df[df['concat_length'] <= trunc]
    return df_filtered


def compare_h_l_values(series, h_l_dict):
    """
    """
    dict_Hseq = h_l_dict['H'] if 'H' in h_l_dict.keys() else None
    dict_Lseq = h_l_dict['L'] if 'L' in h_l_dict.keys() else None
    hseq_value = series.get("Hseq", None)
    hseq_value = None if not isinstance(hseq_value, str) else hseq_value
    lseq_value = series.get("Lseq", None)
    lseq_value = None if not isinstance(lseq_value, str) else lseq_value
    flg = False
    if (dict_Hseq is not None) or (hseq_value is not None):
        flg = (dict_Hseq == hseq_value)
        if flg is False:  return False
    if (dict_Lseq is not None) or (lseq_value is not None):
        flg = (dict_Lseq == lseq_value)
        if flg is False:  return False
    return flg


def modify_posdataid(pos_stdDf, pos_dataid_byAt=True, at_name='sudo_at'):
    pos_abatids = sorted(list(set(pos_stdDf[at_name].to_list())))  # left做at
    for i_, abatid in enumerate(pos_abatids):
        pos_stdDf.loc[pos_stdDf[at_name] == abatid, 'nonredun_dataid'] = i_
    return pos_stdDf


def partition_into_groups(nums, num_groups):
    # 计算每个组的目标元素和
    target_sum = sum(nums) // num_groups
    # 初始化组和当前组元素和
    groups = [[] for _ in range(num_groups)]
    current_group_sum = [0] * num_groups

    # 将元素按降序排列
    sorted_indices = sorted(range(len(nums)), key=lambda i: nums[i], reverse=True)
    # 贪心地将元素放入组中
    for index in sorted_indices:
        # 选择当前组中元素和最小的组
        current_group = min(range(num_groups), key=lambda i: current_group_sum[i])
        # 将元素放入当前组
        groups[current_group].append(index)
        # 更新当前组元素和
        current_group_sum[current_group] += nums[index]
    return groups


def abseq_FromDict(Pair_dict):
    abseq_dict = Pair_dict['Ab_data']['seq_dict']
    Hseq = abseq_dict['H'] if 'H' in abseq_dict.keys() else None
    Lseq = abseq_dict['L'] if 'L' in abseq_dict.keys() else None
    total_abseq = ''.join([seq for seq in [Hseq, Lseq] if seq is not None])
    return Hseq, Lseq, total_abseq


def atseq_FromDict(Pair_dict):
    return Pair_dict['At_data']['atseq']


if __name__ == "__main__":
    mat_dict_str = "{'V0_fr': array([[  0,  25],\n       [ 32,  51],\n       [ 57,  98],\n       [109, 120]]), 'V0_cdr': array([[ 25,  32],\n       [ 51,  57],\n       [ 98, 109]]), 'V1_fr': array([[120, 143],\n       [154, 169],\n       [176, 208],\n       [218, 228]]), 'V1_cdr': array([[143, 154],\n       [169, 176],\n       [208, 218]])}"
    mat_dict = string_to_dict(mat_dict_str)
    # 打印恢复后的字典
    print(mat_dict)
