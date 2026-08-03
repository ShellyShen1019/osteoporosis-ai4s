"""嵌套式时空交叉验证单元测试（纯 Python，无需第三方依赖）。

运行：
    python tests/test_nested_cv.py
或：
    python -m pytest tests/ -q
"""

import os
import sys

# 将仓库根目录加入 sys.path，便于直接以脚本方式运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.nested_cv import NestedSpatioTemporalCV, make_samples, Sample


def test_outer_folds_equal_num_communities():
    samples = make_samples(n_per_cell=50)
    cv = NestedSpatioTemporalCV(["A", "B", "C"], [2026, 2027, 2028])
    folds = list(cv.split(samples))
    # 3 个社区 × 2 个内层验证年份(2026,2027) = 6 个外层折
    assert len(folds) == 6, f"期望 6 折，实际 {len(folds)}"


def test_spatial_isolation():
    """训练/验证集不得包含测试社区的样本（空间隔离）。"""
    samples = make_samples(n_per_cell=50)
    cv = NestedSpatioTemporalCV(["A", "B", "C"], [2026, 2027, 2028])
    for sp in cv.split(samples):
        test_comm = sp["test_community"]
        train_comms = {s.community for s in samples if s.idx in sp["train"]}
        val_comms = {s.community for s in samples if s.idx in sp["val"]}
        assert test_comm not in train_comms
        assert test_comm not in val_comms


def test_temporal_ordering():
    """训练年份须全部早于验证年份（时序递进）。"""
    samples = make_samples(n_per_cell=50)
    cv = NestedSpatioTemporalCV(["A", "B", "C"], [2026, 2027, 2028])
    yr = {s.idx: s.year for s in samples}
    for sp in cv.split(samples):
        max_train_year = max(yr[i] for i in sp["train"])
        min_val_year = min(yr[i] for i in sp["val"])
        # 内层：训练年份 < 验证年份（时序递进）
        assert max_train_year < min_val_year, "训练年份必须早于验证年份"
        # 注：内层验证年可与外层测试年相同（如均为 2028），但分属不同社区，无数据泄漏；
        # 空间隔离已由 test_spatial_isolation 保证。


def test_index_disjoint_within_fold():
    samples = make_samples(n_per_cell=50)
    cv = NestedSpatioTemporalCV(["A", "B", "C"], [2026, 2027, 2028])
    for sp in cv.split(samples):
        s = set(sp["train"]) | set(sp["val"]) | set(sp["test"])
        assert len(s) == len(sp["train"]) + len(sp["val"]) + len(sp["test"])


if __name__ == "__main__":
    test_outer_folds_equal_num_communities()
    test_spatial_isolation()
    test_temporal_ordering()
    test_index_disjoint_within_fold()
    print("All nested_cv tests passed ✓")
    # 打印一份可视化摘要
    samples = make_samples(n_per_cell=50)
    cv = NestedSpatioTemporalCV(["A", "B", "C"], [2026, 2027, 2028])
    print(cv.summary(samples))
