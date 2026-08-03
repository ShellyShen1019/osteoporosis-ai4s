"""基线模型对比：LightGBM / XGBoost / SVM / AdaBoost / LR。

对应申报书 3.2 技术路线图中的对比基线。树模型使用静态 + 动态池化统计量（表格化），
与多模态深度学习模型在统一测试集上比较，验证本案例模型的增益。
"""

from typing import Dict
import numpy as np


def _flatten_for_tabular(Xs: np.ndarray, Xd: np.ndarray) -> np.ndarray:
    """将动态序列池化为统计量，供树/线性模型使用（演示：均值+标准差）。"""
    dyn_mean = Xd.mean(axis=1)
    dyn_std = Xd.std(axis=1)
    return np.concatenate([Xs, dyn_mean, dyn_std], axis=1)


def run_baselines(Xs_tr, Xd_tr, y_tr, Xs_te, Xd_te, y_te) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.metrics import roc_auc_score

    Xtr = _flatten_for_tabular(Xs_tr, Xd_tr)
    Xte = _flatten_for_tabular(Xs_te, Xd_te)

    models = {
        "LR": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "AdaBoost": AdaBoostClassifier(),
    }
    # 可选：LightGBM / XGBoost（若已安装）
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=200, verbose=-1)
    except Exception:
        pass
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(n_estimators=200, eval_metric="logloss")
    except Exception:
        pass

    results = {}
    for name, m in models.items():
        m.fit(Xtr, y_tr)
        if hasattr(m, "predict_proba"):
            prob = m.predict_proba(Xte)[:, 1]
        else:
            prob = m.decision_function(Xte)
        results[name] = float(roc_auc_score(y_te, prob))
    return results


if __name__ == "__main__":
    # 快速自测（合成数据）
    rng = np.random.default_rng(0)
    n, d_s, d_d, T = 600, 30, 22, 4
    Xs = rng.normal(0, 1, (n, d_s)).astype(np.float32)
    Xd = rng.normal(0, 1, (n, T, d_d)).astype(np.float32)
    y = (rng.normal(0, 1, n) > 0).astype(np.float32)
    split = n // 2
    res = run_baselines(Xs[:split], Xd[:split], y[:split],
                        Xs[split:], Xd[split:], y[split:])
    for k, v in res.items():
        print(f"{k}: AUC={v:.3f}")
