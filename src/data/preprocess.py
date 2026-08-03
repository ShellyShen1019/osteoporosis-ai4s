"""数据预处理与缺失值处理。

对应申报书 3.1.1：缺失值多重插补(MICE)、异常值检测、静态/动态数据拆分。
真实数据接入见 load_raw()；本模块函数均兼容合成数据（src/train.py --demo）。
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def mice_impute(df: pd.DataFrame, max_iter: int = 50) -> pd.DataFrame:
    """链式方程多重插补（MICE）。

    连续型变量用 MICE；分类变量按缺失率分别处理（<10% 众数插补，>=10% 归为'未知'）。
    需要 sklearn：from sklearn.experimental import enable_iterative_imputer; IterativeImputer
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    out = df.copy()
    if num_cols:
        imp = IterativeImputer(max_iter=max_iter, random_state=42)
        out[num_cols] = imp.fit_transform(df[num_cols])

    for c in cat_cols:
        miss_rate = df[c].isna().mean()
        if miss_rate < 0.10:
            out[c] = out[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "未知")
        else:
            out[c] = out[c].fillna("未知")
    return out


def tukey_fences(s: pd.Series, k: float = 1.5) -> pd.Series:
    """Tukey's fences 静态离群值检测，返回布尔掩码（True=离群）。"""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def standardize(df: pd.DataFrame, fit_cols: list = None) -> Tuple[pd.DataFrame, Dict]:
    """Z-score 标准化，返回标准化结果与均值/标准差（用于测试集复用）。"""
    from sklearn.preprocessing import StandardScaler
    cols = fit_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler().fit(df[cols])
    out = df.copy()
    out[cols] = scaler.transform(df[cols])
    return out, {"cols": cols, "mean": scaler.mean_, "scale": scaler.scale_}


def split_static_dynamic(X: pd.DataFrame, static_cols: list,
                         dynamic_cols: list, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """将宽表拆分为静态特征与动态序列特征。

    动态特征按 (样本 × 时间步 × 特征) 组织；此处用重复/滚动方式构造演示序列。
    """
    Xs = X[static_cols].to_numpy(dtype=np.float32)
    Xd_raw = X[dynamic_cols].to_numpy(dtype=np.float32)
    # 演示：将动态特征沿时间维复制 seq_len 步（真实场景按年度采样点堆叠）
    Xd = np.stack([Xd_raw] * seq_len, axis=1)
    return Xs, Xd


def load_raw(path: str = None) -> pd.DataFrame:
    """加载原始数据（CSV/Excel）。真实数据接入点。

    期望列包含：community, year, 静态与动态特征, label(OP阳性)。
    未提供 path 时返回 None，由调用方决定使用合成数据。
    """
    if path is None:
        return None
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)
