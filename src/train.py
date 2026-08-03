"""训练入口：端到端演示（合成数据）或接入真实数据。

用法：
    python -m src.train --demo            # 用合成数据跑通完整 pipeline
    python -m src.train --data data/raw/xxx.csv   # 接入真实数据

合成数据用于验证代码链路；真实建模请替换 load_raw 并提供 community/year 列。
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CONFIG
from src.data.nested_cv import NestedSpatioTemporalCV, Sample
from src.model.multimodal import MultiModalOPModel
from src.evaluate import roc_auc, pr_auc


def generate_synthetic(n: int = 1500, seed: int = 42):
    """生成合成多模态数据 + 社区/年份标签，用于演示。"""
    rng = np.random.default_rng(seed)
    Xs = rng.normal(0, 1, (n, CONFIG.n_static)).astype(np.float32)
    Xd = rng.normal(0, 1, (n, CONFIG.seq_len, CONFIG.n_dynamic_feat)).astype(np.float32)
    # 让标签与部分静态/动态特征弱相关
    logit = 0.6 * Xs[:, 0] + 0.4 * Xd[:, -1, 1].mean(axis=-1) + rng.normal(0, 0.8, n)
    y = (logit + rng.normal(0, 0.5, n) > 0.3).astype(np.float32)
    # 社区/年份：分层抽样（每社区×年份约 n/9）
    comms = np.array(CONFIG.communities)
    yrs = np.array(CONFIG.years)
    ci = rng.integers(0, len(comms), n)
    yi = rng.integers(0, len(yrs), n)
    community = comms[ci]
    year = yrs[yi]
    return Xs, Xd, y, community, year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="使用合成数据")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    if args.demo or args.data is None:
        Xs, Xd, y, community, year = generate_synthetic()
    else:
        raise NotImplementedError("真实数据接入：在 preprocess.load_raw 中实现后在此分支加载。")

    samples = [Sample(idx=i, community=community[i], year=int(year[i]))
               for i in range(len(y))]
    cv = NestedSpatioTemporalCV(list(CONFIG.communities), list(CONFIG.years))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_aucs, fold_prs = [], []
    for sp in cv.split(samples):
        model = MultiModalOPModel(
            n_static=CONFIG.n_static, n_dynamic=CONFIG.n_dynamic_feat,
            seq_len=CONFIG.seq_len, static_hidden=CONFIG.static_hidden,
            dynamic_hidden=CONFIG.dynamic_hidden, fusion_dim=CONFIG.fusion_dim,
            dropout=CONFIG.dropout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        tr_idx, va_idx, te_idx = sp["train"], sp["val"], sp["test"]
        # 用训练+验证拟合，测试评估（演示省略单独验证早停）
        fit_idx = tr_idx + va_idx
        ds = TensorDataset(
            torch.tensor(Xs[fit_idx]), torch.tensor(Xd[fit_idx]), torch.tensor(y[fit_idx]))
        dl = DataLoader(ds, batch_size=CONFIG.batch_size, shuffle=True)
        model.train()
        for _ in range(args.epochs):
            for s_b, d_b, y_b in dl:
                s_b, d_b, y_b = s_b.to(device), d_b.to(device), y_b.to(device)
                opt.zero_grad()
                logit, _ = model(s_b, d_b)
                loss = loss_fn(logit, y_b)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            s_t, d_t, y_t = (torch.tensor(Xs[te_idx]).to(device),
                             torch.tensor(Xd[te_idx]).to(device),
                             torch.tensor(y[te_idx]).to(device))
            logit, _ = model(s_t, d_t)
            prob = torch.sigmoid(logit).cpu().numpy()
        auc = roc_auc(y[te_idx], prob)
        pr = pr_auc(y[te_idx], prob)
        fold_aucs.append(auc)
        fold_prs.append(pr)
        print(f"  Fold {sp['fold']} (测试社区={sp['test_community']}): "
              f"AUC={auc:.3f}  PR-AUC={pr:.3f}")

    print(f"\n嵌套式时空验证平均 AUC={np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f} "
          f"| PR-AUC={np.mean(fold_prs):.3f}")


if __name__ == "__main__":
    main()
