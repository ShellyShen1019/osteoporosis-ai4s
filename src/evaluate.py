"""评估与可解释性：AUC / PR-AUC 指标、ROC/PR 曲线绘制、SHAP 解释。

对应申报书 3.1.5 与 4.2 技术创新（可解释性 SHAP）。
"""

from typing import Tuple
import numpy as np


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, y_score))


def calibration(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_score, bins) - 1
    frac_pos, conf = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        frac_pos.append(y_true[m].mean())
        conf.append(y_score[m].mean())
    return np.array(conf), np.array(frac_pos)


def plot_curves(y_true, y_score, out_path="runs/roc_pr.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(fpr, tpr, label=f"AUC={roc_auc(y_true, y_score):.3f}")
    ax[0].set_title("ROC"); ax[0].legend()
    ax[1].plot(rec, prec, label=f"PR-AUC={pr_auc(y_true, y_score):.3f}")
    ax[1].set_title("Precision-Recall"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def shap_explain(model, static_sample, dynamic_sample, feature_names=None, top_k: int = 10):
    """基于 SHAP 的全局/局部解释（可选依赖 shap）。

    说明：对交叉注意力融合模型，SHAP 作用于融合后表征；此处给出占位接口，
    真实使用请安装 shap 并对 head 输入计算 Shapley 值。
    """
    try:
        import shap
        # 演示：对静态分支解释
        explainer = shap.Explainer(model.head)
        sv = explainer(static_sample)
        return sv.values
    except Exception as e:
        print(f"[warn] SHAP 未启用：{e}")
        return None
