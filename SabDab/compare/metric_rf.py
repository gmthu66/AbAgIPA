from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve, auc


def eval_svm(data_x, y_label, model):
    # fpr, tpr, thresholds = roc_curve(y_label, model.decision_function(data_x), pos_label=1)
    predy_score = model.predict_proba(data_x)
    fpr, tpr, thresholds = roc_curve(y_label, predy_score[:, 1], pos_label=1)  # 标签为1的为阳性, 其余为阴性
    predy_ls = predy_score.argmax(axis=1)
    auc_roc = roc_auc_score(y_label, predy_score[:, 1])

    pr_precisions, pr_recalls, _ = precision_recall_curve(y_label, predy_score[:, 1])
    auc_pr = auc(pr_recalls, pr_precisions)
    tn, fp, fn, tp = confusion_matrix(y_label, predy_ls).ravel()
    acc = (tn + tp) / (tn + tp + fn + fp)
    tnr = tn / (tn + fp)  # 返回真阴性率 就是特异度 specificity, 在无病人群中, 检测出阴性的几率
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = f1_score(precision, recall)
    return acc, precision, recall, tnr, auc_roc, auc_pr, f1, fpr, tpr, pr_precisions, pr_recalls


def f1_score(precision_i, recall_i, eps=1e-8):
    f1 = 2 * precision_i * recall_i / (precision_i + recall_i + eps)
    return f1
