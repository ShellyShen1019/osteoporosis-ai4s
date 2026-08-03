"""嵌套式时空交叉验证（Nested Spatio-Temporal Cross-Validation）。

这是本案例的方法论核心，用于严格评估模型在「未知社区 + 未来时间点」的泛化能力，
克服传统随机交叉验证对 OP 风险时空异质性的过度乐观偏差。

设计约束（对应申报书 3.1.4）：
  1. 空间隔离性：训练数据仅用于同社区预测，跨社区数据不可互用；
  2. 时序递进性：模型训练禁止使用未来时序数据（如 2027 年数据不可参与 2026 年模型）。

外层循环（空间分层）：逐次选定单个社区卫生服务中心作为独立测试集（取该社区最晚年份），
其余社区构成建模样本池。
内层循环（时间分层）：在建模样本池内按时间顺序划分训练时段与验证时段，
采用逐年滚动的滑动窗口（本例 2 轮：2026→2027、2027→2028）。

本文件仅依赖 Python 标准库，可被 tests/test_nested_cv.py 直接运行验证。
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Iterator


@dataclass
class Sample:
    idx: int
    community: str
    year: int
    # 其余字段（特征/标签）按需扩展


def _index_by(values):
    return [s.idx for s in values]


class NestedSpatioTemporalCV:
    """嵌套式时空交叉验证生成器。

    参数
    ----
    communities : 社区标识列表，如 ["A", "B", "C"]
    years       : 年份列表（升序），如 [2026, 2027, 2028]
    test_year   : 留出社区用作测试集的年份（取最晚年份）
    """

    def __init__(self, communities: List[str], years: List[int], test_year: int = None):
        self.communities = list(communities)
        self.years = sorted(years)
        self.test_year = test_year if test_year is not None else self.years[-1]

    def split(self, samples: List[Sample]) -> Iterator[Dict[str, Any]]:
        """逐个产出外层折（leave-one-community-out）及其内层时间滑动窗。

        每次产出形如：
            {
              "fold": 1,
              "test_community": "A",
              "train": [idx...],   # 其余社区、且年份 < 内层验证年份
              "val":   [idx...],   # 其余社区、年份 == 内层验证年份
              "test":  [idx...],   # 留出社区、最晚年份
            }
        """
        fold = 0
        for test_comm in self.communities:
            pool = [s for s in samples if s.community != test_comm]
            test = [s for s in samples
                    if s.community == test_comm and s.year == self.test_year]
            if not test:
                continue  # 该社区无最晚年份数据则跳过
            # 内层：逐年滚动时间窗（2026→2027、2027→2028）
            # 池内验证年份可到最晚年份（与留出社区测试集不同社区，无数据泄漏）
            for val_year in self.years:
                train = [s for s in pool if s.year < val_year]
                val = [s for s in pool if s.year == val_year]
                if not train or not val:
                    continue  # 训练年份须早于验证年份（时序递进）
                fold += 1
                yield {
                    "fold": fold,
                    "test_community": test_comm,
                    "train": _index_by(train),
                    "val": _index_by(val),
                    "test": _index_by(test),
                }

    def summary(self, samples: List[Sample]) -> str:
        lines = [f"嵌套式时空交叉验证：社区={self.communities} 年份={self.years} 测试年={self.test_year}"]
        for sp in self.split(samples):
            lines.append(
                f"  Fold {sp['fold']} | 测试社区={sp['test_community']} | "
                f"train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}"
            )
        return "\n".join(lines)


def make_samples(n_per_cell: int = 50,
                 communities=("A", "B", "C"),
                 years=(2026, 2027, 2028)) -> List[Sample]:
    """构造示例样本（每个 社区×年份 单元 n_per_cell 个），便于测试/演示。"""
    samples = []
    i = 0
    for c in communities:
        for y in years:
            for _ in range(n_per_cell):
                samples.append(Sample(idx=i, community=c, year=y))
                i += 1
    return samples


if __name__ == "__main__":
    samples = make_samples()
    cv = NestedSpatioTemporalCV(list(("A", "B", "C")), list((2026, 2027, 2028)))
    print(cv.summary(samples))
