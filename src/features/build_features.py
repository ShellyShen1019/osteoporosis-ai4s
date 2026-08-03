"""特征工程：静态/动态特征构造与跨模态筛选。

对应申报书 3.1.5：基线-动态交互项、短期/长期时间窗统计量、LGBFS 动态自适应阈值筛选。
"""

from typing import Tuple
import numpy as np
import pandas as pd


def add_interaction(static: pd.DataFrame, dynamic_means: pd.DataFrame,
                    static_cols: list, dyn_cols: list) -> pd.DataFrame:
    """构造静态因子 × 动态指标变化率 的交互特征（演示：静态 × 动态均值偏差）。"""
    out = pd.DataFrame(index=static.index)
    for sc in static_cols[:3]:  # 取代表性静态因子做演示
        for dc in dyn_cols[:3]:
            out[f"{sc}_x_{dc}"] = static[sc].to_numpy() * (
                dynamic_means[dc].to_numpy() - dynamic_means[dc].mean()
            )
    return out


def temporal_stats(Xd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从动态序列提取短期(末步)与长期(趋势斜率)统计量。

    Xd: (N, seq_len, F)
    返回 short (N,F) 与 trend (N,F)（对时间维做一阶线性拟合斜率）。
    """
    short = Xd[:, -1, :]
    n, seq_len, f = Xd.shape
    t = np.arange(seq_len)
    # 每特征的最小二乘斜率
    xm = t.mean()
    trend = ((t - xm) * (Xd - Xd.mean(axis=1, keepdims=True)).mean(axis=1)) / ((t - xm) ** 2).sum()
    return short, trend


def select_features_lgbfs(X: pd.DataFrame, y: np.ndarray,
                          importance_threshold: float = 0.01) -> list:
    """包裹式特征选择（LightGBM feature selection, LGBFS）。

    保留重要性分 > threshold 的特征；若 lightgbm 不可用则退回方差阈值。
    """
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X, y)
        imp = dict(zip(X.columns, model.feature_importances_ / model.feature_importances_.sum()))
        kept = [c for c, v in imp.items() if v > importance_threshold]
        return kept
    except Exception:
        # 退回：剔除近零方差特征
        vars = X.var()
        return X.columns[vars > 1e-6].tolist()


def adaptive_threshold_selection(importance: np.ndarray) -> np.ndarray:
    """动态特征自适应筛选：按重要性分布自动确定最优保留阈值（克服固定阈值跨社区泛化差）。

    采用重要性累积占比 90% 的截断作为演示策略。
    """
    imp = np.sort(importance)[::-1]
    cum = np.cumsum(imp) / imp.sum()
    k = int(np.searchsorted(cum, 0.90) + 1)
    return np.arange(len(imp))[:k]
