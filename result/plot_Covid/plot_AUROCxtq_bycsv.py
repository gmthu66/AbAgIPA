import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os.path as osp
from os.path import basename


def plot_violinplot_from_csv(csv_path, save_path=None, col_order=None):
    # 从 CSV 文件读取数据
    df = pd.read_csv(csv_path, index_col=False)
    if col_order is not None:
        df = df[col_order]

    # 转换数据为绘图所需的格式
    data_for_plot = []
    labels = []

    for column in df.columns:
        data_for_plot.extend(df[column].values)
        labels.extend([column] * len(df))

    # 使用 seaborn 设置图形样式
    sns.set(style='whitegrid', font_scale=1.2)

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 绘制小提琴图
    violin_plot = sns.violinplot(x=labels, y=data_for_plot, inner='quartile', palette='pastel', width=0.8)

    # 显示数据点分布
    swarm_plot = sns.swarmplot(x=labels, y=data_for_plot, color='black', alpha=0.5)

    # 添加均值线
    mean_values = [df[column].mean() for column in df.columns]
    mean_lines = plt.plot(range(len(mean_values)), mean_values, marker='o', linestyle='--', color='red', markersize=8, label='Mean')

    # 设置标题和标签
    plt.ylabel('AUROC Score', fontsize=14)

    # 设置坐标轴刻度标签
    plt.xticks(fontsize=12, rotation=45, ha="right")  # 旋转刻度标签，使其更易读
    plt.yticks(fontsize=12)

    # 显示图例，并将位置调整为右上角
    plt.legend(loc='upper right')

    # 设置y轴范围
    plt.ylim(0.4, 1.0)

    # 去掉上方和右方的边框
    sns.despine()

    # 显示图形或保存图形
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # 使用 bbox_inches='tight' 保存时避免裁切内容
    else:
        plt.show()


# 指定 CSV 文件路径
# 画SabDab的AUC-ROC箱式图
csv_path = '/data/gm_data/AbAtInteraction/AbAtIPA/abatInter_SCA/SabDab/trained_models/res_dir/AUC-ROC.csv'
# 画AUC-PR的箱式图
csv_dir = osp.dirname(csv_path)
fig_path = osp.join(csv_dir, basename(csv_path).split('.')[0] + '_violin.jpg')

# 调用函数画箱式图
plot_violinplot_from_csv(csv_path, fig_path)
