"""全局配置：特征定义、社区/年份时空结构、模型超参。

所有可调参数集中在此，便于复现与消融实验。
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- 时空结构（嵌套式时空验证）----
    communities: tuple = ("A", "B", "C")          # 社区卫生服务中心
    years: tuple = (2026, 2027, 2028)             # 采集年份（时间窗）
    test_year: int = 2028                          # 留出社区使用其最晚年份作为测试集

    # ---- 多模态特征维度 ----
    n_static: int = 30          # 静态候选变量（基线属性 + 临床指标）
    n_dynamic_feat: int = 22    # 每时间步动态特征（血液指标/运动/健康状态）
    seq_len: int = 4            # 动态序列长度（按年度采样点）

    # ---- 模型超参 ----
    static_hidden: tuple = (64, 32)
    dynamic_hidden: int = 64
    lstm_layers: int = 1
    fusion_dim: int = 32
    dropout: float = 0.3
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    seed: int = 42

    # ---- 标签 ----
    # 1 = 骨质疏松阳性（DXA 金标准 T值≤-2.5 或 QUS 初筛 T值≤-1.0 且复核阳性）
    pos_label: int = 1


CONFIG = Config()
