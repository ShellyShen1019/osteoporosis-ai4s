# 骨质疏松动态预警模型 · AI4S 案例代码仓库

> **AI for Science (AI4S) 科研赛道参赛案例**
> 基于多模态嵌套式时空验证的骨质疏松（OP）动态预警模型与基层智能筛查系统

本项目是南昌大学第一附属医院全科医学科「教师人工智能创新应用能力竞赛 · AI4S 科研赛道」的配套代码仓库，
对应研究：**基于多模态嵌套式时空验证的骨质疏松动态预警模型构建与社区防控体系研究**。

## 科学问题
骨质疏松症（OP）是我国老龄化社会的重大公共卫生问题，50 岁以上人群患病率约 19.2%，
但基层 DXA 设备覆盖率不足 15%、漏诊率超 45%，「早筛难—干预迟」困局突出。
现有筛查工具（FRAX®、OSTA、IOF、QUS）或依赖昂贵设备，或仅用静态指标、忽视个体差异与动态演变。
本项目以 AI 模型融合**生物学衰老（表型年龄及加速度）、昼夜运动行为（DN-PAPQ 量表）、老年综合评估（CGA）**
等多模态动态数据，构建可推广至基层的 OP 动态预警模型。

## 方法亮点
1. **多模态融合架构**：静态通道（全连接网络）处理基线特征，动态通道（LSTM）提取时序模式，
   融合层采用**交叉注意力机制**生成联合表征
   `H_fusion = Softmax(Q_static · K_dynamicᵀ / √d) · V_dynamic`。
2. **嵌套式时空交叉验证（Nested Spatio-Temporal CV）**：
   - 空间维：按社区（A/B/C）隔离训练集与测试集；
   - 时间维：按年份（2026/2027/2028）禁用未来数据；
   - 外层空间留出 + 内层时间滑动窗口，模拟新社区与未来时点的真实部署，克服传统 CV 的过度乐观偏差。
3. **可解释性**：采用 SHAP 解析模型决策，服务临床可信落地。

## 目录结构
```
src/
  config.py                 # 全局配置（特征维度、社区/年份定义等）
  data/
    nested_cv.py            # 嵌套式时空交叉验证（纯 Python，无第三方依赖）
    preprocess.py           # 缺失值多重插补(MICE)、标准化、静态/动态拆分
  features/
    build_features.py       # 静态/动态特征工程与跨模态筛选
  model/
    multimodal.py           # 静态全连接 + 动态 LSTM + 交叉注意力融合模型 (PyTorch)
  train.py                  # 训练入口（含合成数据示例，可端到端运行）
  evaluate.py               # 指标计算(AUC/PR/SHAP)与可视化
  baselines.py              # LightGBM / XGBoost / SVM / AdaBoost / LR 对比
demo/
  app.py                    # Streamlit 轻量演示（输入多模态特征→输出风险评分）
tests/
  test_nested_cv.py         # 嵌套式时空验证单元测试（纯 Python 可跑）
```

## 快速开始
```bash
pip install -r requirements.txt
# 1) 验证嵌套式时空交叉验证逻辑（无需 torch）
python -m pytest tests/ -q
# 或直接运行
python tests/test_nested_cv.py

# 2) 端到端跑通（使用合成数据演示完整 pipeline）
python -m src.train --demo

# 3) 启动本地演示界面
streamlit run demo/app.py
```

## 数据说明
- 真实数据来自南昌市 6 区社区卫生服务中心分层抽样（2026.1–2028.12，计划 ≥1500 例，OP 阳性事件 ≥300）。
- 社区 QUS 初筛 → 南昌大学第一附属医院 DXA 金标准双盲复核。
- 多模态特征：生物学衰老（表型年龄及加速度，9 项血液指标）、昼夜节律运动行为（自研 DN-PAPQ 量表）、
  老年综合评估（CGA）及临床指标，共 30 项候选变量。
- 本仓库 `src/train.py --demo` 使用**合成数据**演示完整流程；真实数据接入见 `src/data/preprocess.py` 的 `load_raw()`。

## 相关成果
- Shen X, Xiong J, Wang S, Hu G, Zhang S. **Application of machine learning in osteoporosis screening: a narrative review**.
  *npj Digital Medicine* (Nature 旗下期刊, 2026, IF≈18). DOI: https://doi.org/10.1038/s41746-026-02516-6
- 研究遵循 STROBE 与 TRIPOD+AI 国际报告规范。

## 许可证
MIT
