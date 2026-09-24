"""FreshSense: a Streamlit prototype for non-destructive food sensing research."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="FreshSense | 食品无损检测", page_icon="🍃", layout="wide")

THEME = {
    "navy": "#102E29", "teal": "#0B806B", "mint": "#DDF5EE", "pale": "#F5F9F7",
    "ink": "#173B34", "muted": "#668078", "orange": "#E7873B", "red": "#D85D4D",
}

st.markdown(f"""
<style>
 .stApp {{ background: {THEME['pale']}; color: {THEME['ink']}; }}
 section[data-testid="stSidebar"] {{ background: {THEME['navy']}; }}
 section[data-testid="stSidebar"] * {{ color: #EAF7F2 !important; }}
 .block-container {{ max-width: 1450px; padding: 1.5rem 2.6rem 3rem; }}
 .hero {{ background: linear-gradient(120deg, #102E29, #126253 60%, #159A7F); border-radius: 24px;
          padding: 2.2rem 2.5rem; color: white; box-shadow: 0 16px 35px rgba(17, 82, 68, .18); }}
 .hero h1 {{ font-size: 2.45rem; margin: .25rem 0 .45rem; letter-spacing: -.045em; }}
 .eyebrow {{ color: #AFEBDD; font-size: .78rem; font-weight: 750; letter-spacing: .12em; }}
 .hero p {{ color: #DCF7EF; margin: 0; font-size: 1.02rem; max-width: 790px; line-height: 1.65; }}
 .metric-card {{ background: white; border: 1px solid #E1EDE8; border-radius: 17px; padding: 1.05rem 1.15rem;
                 min-height: 116px; box-shadow: 0 6px 18px rgba(16, 59, 50, .045); }}
 .metric-label {{ color: {THEME['muted']}; font-size: .83rem; font-weight: 650; }}
 .metric-value {{ color: {THEME['ink']}; font-size: 1.65rem; font-weight: 780; margin-top: .32rem; }}
 .metric-note {{ color: #80968F; font-size: .79rem; margin-top: .35rem; }}
 .section-title {{ color: {THEME['ink']}; font-size: 1.28rem; font-weight: 760; margin: 1.65rem 0 .72rem; }}
 .panel {{ background: white; border: 1px solid #E1EDE8; border-radius: 19px; padding: 1.25rem;
           box-shadow: 0 6px 18px rgba(16, 59, 50, .045); }}
 .status-dot {{ display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; background: #24A580; margin-right: .42rem; }}
 .stButton > button {{ background: {THEME['teal']}; color: white; border: none; border-radius: 10px; font-weight: 720; }}
 .stButton > button:hover {{ background: #075F50; color: white; }}
</style>
""", unsafe_allow_html=True)


SAMPLE_PROFILES = {
    "鸡肉": {"factor": 0.78, "base": 92, "note": "挥发性胺类相关响应仍处于新鲜参考区间"},
    "牛奶": {"factor": 0.96, "base": 77, "note": "酸化相关气味变化轻微，建议结合储存温度判断"},
    "草莓": {"factor": 0.65, "base": 94, "note": "果香酯类与水分响应稳定"},
    "花生": {"factor": 0.86, "base": 85, "note": "气味指纹与花生类别高度一致，请注意过敏原风险"},
}


def simulate_data(food: str, condition: str) -> pd.DataFrame:
    """Generate a stable 16-channel demo signal with the same shape as instrument output."""
    seed = sum(map(ord, f"{food}:{condition}"))
    rng = np.random.default_rng(seed)
    time_s = np.arange(0, 281, 2)
    rise = np.clip((time_s - 30) / 35, 0, 1)
    decay = np.exp(-np.maximum(time_s - 125, 0) / 105)
    pulse = rise * decay
    severity = {"新鲜": 0.72, "轻度变质": 1.08, "明显变质": 2.5}[condition]
    data: dict[str, np.ndarray] = {"time_s": time_s}
    for i in range(1, 17):
        if condition == "新鲜":
            weight = 1.0
        elif condition == "轻度变质":
            weight = 1.5 if i in [1,5,10,15] else 1.1
        else:
            weight = 3.0 if i in [1,5,10,15] else 1.5
        amplitude = severity*weight*(0.15+i*0.035+rng.uniform(-0.045,0.045))
        sign = -1 if i in {3,7,11,16} else 1
        noise = rng.normal(loc=0,scale=0.005,size=len(time_s)).cumsum()/4
        data[f"S{i:02d}"]=sign*amplitude*pulse+noise
    return pd.DataFrame(data)


def validate_and_prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    renamed = raw.rename(columns={"时间 (s)": "time_s", "时间(s)": "time_s", "time": "time_s"}).copy()
    sensor_cols = [col for col in renamed.columns if str(col).upper().startswith("S")]
    if "time_s" not in renamed.columns:
        return None, "CSV 缺少时间列。请使用 time_s 或 时间 (s)。"
    if len(sensor_cols) < 4:
        return None, "至少需要 4 个以 S 开头的传感器列，例如 S01、S02。"
    use_cols = ["time_s", *sensor_cols]
    data = renamed[use_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 20:
        return None, "有效数据行不足 20 行，无法完成时序分析。"
    data = data.sort_values("time_s").drop_duplicates("time_s")
    baseline = data[sensor_cols].iloc[: max(3, len(data) // 10)].mean()
    data[sensor_cols] = data[sensor_cols].subtract(baseline, axis=1)
    return data, None


def feature_table(data: pd.DataFrame) -> pd.DataFrame:
    sensors = [col for col in data.columns if col != "time_s"]
    rows = []
    for channel in sensors:
        values = data[channel].to_numpy()
        rows.append({
            "通道": channel,
            "峰值响应": round(float(np.max(np.abs(values))), 4),
            "末端响应": round(float(values[-1]), 4),
            "响应面积": round(float(np.trapz(np.abs(values), data["time_s"])), 3),
        })
    return pd.DataFrame(rows)


def assess(data: pd.DataFrame, food: str) -> dict[str, object]:
    feats = feature_table(data)
    intensity = float(feats["峰值响应"].median())
    active_ratio = float((feats["峰值响应"] > 0.04).mean())
    score = int(np.clip(100 - intensity * 36 - max(0, active_ratio - 0.82) * 20, 18, 98))
    if score >=80:
        label, risk = "新鲜", "低风险"
    elif score >=60:
        label, risk = "建议尽快食用", "中等风险"
    else:
        label, risk = "不建议食用", "高风险"
    if food == "花生":
        label = "花生类别已识别"
        if risk == "低风险":
            risk = "过敏原提示"
    confidence = round(float(np.clip(82 + active_ratio * 14 - intensity * 2.5, 75, 98.7)), 1)
    quality = "良好" if active_ratio >= 0.75 else "需复测"
    return {"score": score, "label": label, "risk": risk, "confidence": confidence,
            "active": int((feats["峰值响应"] > 0.04).sum()), "quality": quality, "features": feats}


def line_chart(data: pd.DataFrame, selected: list[str]) -> go.Figure:
    colors = ["#0B806B", "#15A48A", "#E7873B", "#5E8DD4", "#B26791", "#6D8A7D", "#D85D4D", "#7A68C2"]
    fig = go.Figure()
    for i, col in enumerate(selected):
        fig.add_trace(go.Scatter(x=data.time_s, y=data[col], mode="lines", name=col,
                                 line={"width": 3.0, "color": colors[i % len(colors)]}))
    fig.add_vrect(x0=30, x1=125, fillcolor="#A4E4D3", opacity=.25, line_width=0,
                  annotation_text="气味暴露区间", annotation_position="top left",
                  annotation_font = dict(size=14,color="#0B806B",weight="bold"),
                  annotation_showarrow=False,
                  annotation_borderpad=0)
    fig.update_layout(height=360, margin={"l": 50, "r": 20, "t": 28, "b": 30}, paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
                      legend={"orientation": "h", "y": 1.13}, xaxis={"title": "时间 (s)", "showgrid": True,
                      "gridcolor": "#D1D5DB","tickfont":{"color":"#333333","size":12},"title_font":{"color":"#111111",
                        "size":14,"weight":"bold"}},
                      yaxis={"title": "归一化响应", "gridcolor": "#D1D5DB", "zerolinecolor": "#9CA3AF",
                             "tickfont":{"color":"#333333","size":12},"title_font":{"color":"#111111","size":14,"weight":"bold"}})
    return fig
def radar_chart(features: pd.DataFrame) -> go.Figure:
    labels = features["通道"].tolist()
    values = features["峰值响应"].tolist()
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself",
                                    line={"color": "#0B806B", "width": 2}, fillcolor="rgba(11,128,107,.22)"))
    fig.update_layout(height=360, margin={"l": 25, "r": 25, "t": 20, "b": 20}, paper_bgcolor="#F8F9FA", showlegend=False,
                      polar={
                          "bgcolor":"#F8F9FA",
                          "radialaxis":{
                              "showticklabels":True,
                              "range":[0,1.0],
                              "gridcolor":"#D1D5DB",
                              "gridwidth":1.5,
                              "tickfont":{"color":"#333333","size":11},
                              "linecolor":"#9CA3AF",
                          },
                          "angularaxis":{
                              "gridcolor":"#D1D5DB",
                              "linecolor":"#9CA3AF",
                              "tickfont":{"color":"#333333","size":12,"weight":"bold"},
                          }
                      }
                      )
    return fig


def make_report(record: dict[str, object]) -> bytes:
    report = pd.DataFrame([record]).rename(columns={"timestamp": "检测时间", "batch": "批次编号", "food": "食品样品",
                                                     "label": "检测结论", "risk": "风险等级", "score": "新鲜度评分",
                                                     "confidence": "模型置信度", "quality": "数据质量"})
    return report.to_csv(index=False).encode("utf-8-sig")


if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.markdown("## 🍃 FreshSense")
    st.caption("食品无损检测研究工作台")
    st.divider()
    food = st.selectbox("食品样品", list(SAMPLE_PROFILES))
    batch = st.text_input("批次编号", "FS-2026-0919-A01")
    source = st.radio("数据来源", ["使用模拟传感器数据", "上传 CSV 数据"], label_visibility="visible")
    condition = st.select_slider("模拟样品状态", ["新鲜", "轻度变质", "明显变质"], value="新鲜",
                                  disabled=source != "使用模拟传感器数据")
    file = st.file_uploader("上传 CSV", type="csv", disabled=source != "上传 CSV 数据")
    st.divider()
    run = st.button("开始智能检测", use_container_width=True)
    st.caption("支持 16 通道，也支持至少 4 通道的预实验数据。")

if run:
    if source == "上传 CSV 数据":
        if file is None:
            st.sidebar.error("请先选择 CSV 文件。")
        else:
            prepared, error = validate_and_prepare(pd.read_csv(file))
            if error:
                st.sidebar.error(error)
            else:
                result = assess(prepared, food)
                st.session_state.result = {"data": prepared, **result, "source": "CSV 上传", "food": food, "batch": batch}
    else:
        prepared = simulate_data(food, condition)
        result = assess(prepared, food)
        st.session_state.result = {"data": prepared, **result, "source": "模拟数据", "food": food, "batch": batch}
    if st.session_state.result is not None:
        item = st.session_state.result
        record = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "batch": batch, "food": food,
                  "label": item["label"], "risk": item["risk"], "score": item["score"], "confidence": item["confidence"],
                  "quality": item["quality"]}
        st.session_state.history.insert(0, record)

st.markdown("""
<div class="hero"><div class="eyebrow">MULTIPLEXED GAS SENSING · RESEARCH PROTOTYPE</div>
<h1>食品新鲜度与气味指纹检测</h1>
<p>将多通道气体传感器时序、信号质量和模型判定放在同一工作流中。</p></div>
""", unsafe_allow_html=True)

if st.session_state.result is None:
    st.info("请在左侧设置样品与数据来源，再点击“开始智能检测”。系统会生成完整的分析结果。")
    current_data = simulate_data("鸡肉", "新鲜")
    preview = assess(current_data, "鸡肉")
    state_text = "等待检测"
else:
    current = st.session_state.result
    current_data, preview, state_text = current["data"], current, "检测完成"

now = datetime.now().strftime("%Y-%m-%d %H:%M")
metrics = [
    ("系统状态", state_text, "<span class='status-dot'></span>传感器阵列在线"),
    ("数据来源", current.get("source", "演示预览") if st.session_state.result else "演示预览", "时间序列已完成预处理"),
    ("模型置信度", f"{preview['confidence']}%", "基于信号特征的原型判定"),
    ("检测时间", now, "环境参考：25°C · 48% RH"),
]
cols = st.columns(4)
for col, (label, value, note) in zip(cols, metrics):
    col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='section-title' style='padding-left:1rem;'>检测结果</div>", unsafe_allow_html=True)


# 使用容器包裹，并在内部加上和"智能判定"一样的标题
# 合并为一个大白框
with st.container(border=True):
    # ================= 1. 食品状态区域 =================
    st.markdown("<div class='section-title' style='margin-top: 0;'>食品状态</div>", unsafe_allow_html=True)

    color = THEME["teal"] if preview["risk"] == "低风险" else THEME["orange"] if preview["risk"] == "中等风险" else THEME["red"]
    st.markdown(f"<span style='font-size:1.55rem;font-weight:780;color:{color};'>{preview['label']}</span>", unsafe_allow_html=True)

    st.progress(preview["score"] / 100, text=f"新鲜度评分: {preview['score']} / 100")
    # 根据风险等级动态生成解释文案
    if preview["risk"] == "高风险":
       note_text = "挥发性胺类相关响应显著升高，已超出新鲜参考区间。"
    elif preview["risk"] == "中等风险":
       note_text = "挥发性胺类相关响应正在上升，建议尽快食用。"
    elif preview["risk"] == "过敏原提示":
       note_text = SAMPLE_PROFILES.get(food, SAMPLE_PROFILES["鸡肉"])["note"]
    else:
       note_text = "挥发性胺类相关响应仍处于新鲜参考区间。"
    st.write(note_text)
    st.caption(f"数据质量: {preview['quality']} · 有效响应通道: {preview['active']} / {len(preview['features'])}")

    # ================= 2. 风险与建议区域（直接接在后面） =================
    # 加一条淡淡的水平分割线，视觉上分隔一下
    st.markdown("<div class='section-title' style='margin-top: 0;'>风险与建议</div>", unsafe_allow_html=True)
    if preview["risk"] == "高风险":
       bg_color, border_color, text_color = "rgba(216, 93, 77, 0.1)", "#D85D4D", "#D85D4D"
       note = "建议停止食用，并使用微生物或理化方法进行复核。"
    elif preview["risk"] == "中等风险":
        bg_color, border_color, text_color = "rgba(231, 135, 59, 0.1)", "#E7873B", "#E7873B"
        note = "建议尽快食用，或结合其他理化指标进行复核。"
    elif preview["risk"] == "过敏原提示":
       bg_color, border_color, text_color = "rgba(231, 135, 59, 0.1)", "#E7873B", "#E7873B"
       note = "该结果仅用于类别识别演示，不替代过敏原定量检测。"
    else:
       bg_color, border_color, text_color = "rgba(11, 128, 107, 0.1)", "#0B806B", "#0B806B"
       note = "结果可用于研究展示。真实应用需依据标准化采样与独立验证集校准。"

st.markdown(f"""
<div style="background: {bg_color}; border-left: 4px solid {border_color}; padding: 12px 16px; border-radius: 8px; margin-top: 8px; color: {text_color}; font-size: 0.95rem;">
    {note}
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section-title'>传感器响应与气味指纹</div>", unsafe_allow_html=True)
plot_left, plot_right = st.columns([1.5, 1])
sensor_cols = [column for column in current_data.columns if column != "time_s"]
with plot_left:
    selected = st.multiselect("显示传感器通道", sensor_cols, default=sensor_cols[:6], max_selections=8)
    st.plotly_chart(line_chart(current_data, selected or sensor_cols[:1]), use_container_width=True)
with plot_right:
    st.plotly_chart(radar_chart(preview["features"]), use_container_width=True)

st.markdown("<div class='section-title'>研究记录与数据导出</div>", unsafe_allow_html=True)
history_tab, features_tab, guide_tab = st.tabs(["检测历史", "信号特征", "接入真实模型"])
with history_tab:
    if st.session_state.history:
        history = pd.DataFrame(st.session_state.history)
        st.dataframe(history.rename(columns={"timestamp": "检测时间", "batch": "批次", "food": "食品", "label": "结论", "risk": "风险", "score": "评分", "confidence": "置信度", "quality": "数据质量"}), use_container_width=True, hide_index=True)
        st.download_button("下载本次检测报告 CSV", make_report(st.session_state.history[0]), "FreshSense_检测报告.csv", "text/csv")
    else:
        st.caption("尚无检测记录。完成一次检测后，记录将显示在这里。")
with features_tab:
    st.dataframe(preview["features"], use_container_width=True, hide_index=True)
    csv = current_data.to_csv(index=False).encode("utf-8-sig")
    st.download_button("下载预处理后的传感器数据", csv, "FreshSense_预处理信号.csv", "text/csv")
with guide_tab:
    st.markdown("""
    1. 真实仪器数据应包含 `time_s`（或 `时间 (s)`）以及至少 4 个以 `S` 开头的传感器列。
    2. 建议同时记录温湿度、流量、食品质量、储存时间和批次，便于后续做可重复性分析。
    """)
