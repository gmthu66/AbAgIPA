from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc, confusion_matrix
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def calculate_metrics(y_true, y_probs, plot_roc_pr_curves=False, need_softmax=True):
    """y_true是一个大小为[l]的array或list, y_probs为大小为[l, 2]的array"""
    # 计算 AUC-ROC
    if need_softmax:
        exp_x = np.exp(y_probs)
        y_probs = (exp_x / np.sum(exp_x, axis=1, keepdims=True)) if (y_probs.sum() != y_probs.shape[0]) else y_probs
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    auc_roc = auc(fpr, tpr)
    # auc_roc = roc_auc_score(y_true, y_probs)
    # sorted_indices = np.argsort(y_probs[:, 1])
    # y_probs_sorted = y_probs[sorted_indices]
    # y_true_sorted = y_true[sorted_indices]
    # 计算 AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs[:, 1])
    auc_pr = auc(recall_curve, precision_curve)  # 原本这里写反了
    # 计算混淆矩阵
    predy_ls = y_probs.argmax(axis=1)
    tn, fp, fn, tp = confusion_matrix(y_true, predy_ls).ravel()
    # train_acc = (tn + tp) / (tn + tp + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)  # 也是真阳性率
    tnr = tn / (tn + fp)  # 返回真阴性率 就是特异度 specificity, 在无病人群中, 检测出阴性的几率
    f1 = f1_score(precision, recall)

    # 绘制 ROC 和 PR 曲线
    # if plot_roc_pr_curves:
    #     fpr, tpr, _ = roc_curve(y_true, y_probs)
    #     precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)

    #     plt.figure(figsize=(12, 4))
    #     # ROC 曲线
    #     plt.subplot(1, 2, 1)
    #     plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.2f}')
    #     plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('ROC Curve')
    #     plt.legend()

    #     # PR 曲线
    #     plt.subplot(1, 2, 2)
    #     plt.plot(recall_curve, precision_curve, label=f'AUC-PR = {auc_pr:.2f}')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Precision-Recall Curve')
    #     plt.legend()
    #     plt.show()

    return {
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Precision': precision,
        'Recall': recall,
        'TNR': tnr,
        'F1 Score': f1,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
    }


def f1_score(precision_i, recall_i, eps=1e-8):
    f1 = 2 * precision_i * recall_i / (precision_i + recall_i + eps)
    return f1


def save_attention_heatmap(attention_matrix, save_path, title='Attention Heatmap'):
    # 设置图形大小
    attention_matrix = attention_matrix.detach().cpu()
    attention_matrix = scale_tensor(attention_matrix, scale_min=-5, sacle_max=5)
    fig, ax = plt.subplots(figsize=(8, 8), gridspec_kw={'height_ratios': [1], 'width_ratios': [1]})
    # cmap = 'cividis'  # 选择颜色映射（从紫色到红色）
    cmap = 'RdBu_r'  # 选择颜色映射（从紫色到红色）
    # 使用Seaborn库中的heatmap函数画热图
    # sns.heatmap(attention_matrix, cmap='viridis', annot=True, fmt='.4f', cbar_kws={'label': 'Attention Score'})
    # sns.heatmap(attention_matrix, cmap=cmap, annot=True, fmt='.4f', cbar_kws={'label': 'Attention Score'})
    sns.heatmap(attention_matrix, cmap=cmap, center=0, vmin=-10, vmax=8, annot=False, cbar_kws={'label': 'Attention Score'})

    # 设置标题和标签
    plt.title(title)
    plt.tight_layout()
    # plt.xlabel('Residue Index')
    # plt.ylabel('Residue Index')

    # 保存图像到指定路径
    plt.savefig(save_path, dpi=300)


def scale_tensor(tensor, scale_min, sacle_max):
    min_value = tensor.min()
    max_value = tensor.max()

    scaled_tensor = scale_min + (sacle_max - scale_min) * (tensor - min_value) / (max_value - min_value)    
    return scaled_tensor
