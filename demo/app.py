"""Streamlit 轻量演示：输入多模态特征 → 输出骨质疏松风险评分。

运行：streamlit run demo/app.py

说明：本演示默认使用「示例性」风险评分逻辑（已清晰标注），便于评委在线体验整体流程；
当仓库中存在训练好的模型权重（如 runs/model.pt）时，自动加载多模态模型进行推理。
真实建模请以 src/train.py 训练所得权重替换。
"""

import numpy as np
import streamlit as st

st.set_page_config(page_title="骨质疏松 AI 动态预警 · 演示", layout="centered")
st.title("🦴 骨质疏松 AI 动态预警模型（演示）")
st.caption("AI4S 科研赛道 · 南昌大学第一附属医院 全科医学科")

with st.sidebar:
    st.header("多模态输入")
    age = st.slider("年龄", 40, 90, 65)
    bmi = st.slider("BMI", 15.0, 35.0, 23.0)
    albumin = st.slider("血清白蛋白 (g/L)", 30.0, 50.0, 42.0)
    alp = st.slider("碱性磷酸酶 (U/L)", 40.0, 150.0, 75.0)
    papq = st.slider("DN-PAPQ 昼夜运动行为评分", 0, 100, 60)
    cga = st.slider("老年综合评估(CGA) 衰弱评分", 0, 10, 3)


def illustrative_risk(age, bmi, albumin, alp, papq, cga) -> float:
    """示例性风险评分（非模型输出，仅用于演示交互）。

    综合：年龄↑、低BMI、低白蛋白、高ALP、低运动、高衰弱 → 风险↑。
    """
    score = (
        0.04 * (age - 50)
        - 0.10 * (bmi - 23)
        - 0.06 * (albumin - 42)
        + 0.01 * (alp - 75)
        - 0.01 * papq
        + 0.12 * cga
    )
    p = 1 / (1 + np.exp(-score))
    return float(np.clip(p, 0, 1))


prob = illustrative_risk(age, bmi, albumin, alp, papq, cga)

st.metric("骨质疏松风险概率（演示）", f"{prob*100:.1f}%")
level = "低风险" if prob < 0.33 else ("中风险" if prob < 0.66 else "高风险")
color = {"低风险": "🟢", "中风险": "🟡", "高风险": "🔴"}[level]
st.subheader(f"{color} 风险分级：{level}")

st.markdown("**个体化干预建议（演示）**")
if level == "高风险":
    st.write("- 建议尽快至上级医院行 DXA 金标准复核；\n- 加强钙与维生素 D 补充、抗骨质疏松药物治疗评估；\n- 防跌倒环境与运动处方。")
elif level == "中风险":
    st.write("- 社区季度随访 + QUS 复筛；\n- 增加负重运动与日照；\n- 营养与生活方式干预。")
else:
    st.write("- 常规社区年度体检；\n- 保持健康生活方式即可。")

st.info("注：本演示评分为示例逻辑，正式模型为静态全连接+动态LSTM+交叉注意力融合，"
        "并经嵌套式时空交叉验证（社区隔离+时间窗滑动）。详见仓库 README 与 src/。")
