# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# ========== 字体配置（去除中文依赖） ==========
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="质检与满意度分析看板", layout="wide")
st.title("质检-满意度分析")

# ====================== 上传多个文件 ======================
uploaded_files = st.file_uploader(
    "请上传多个质检数据文件（支持 Excel / CSV，可多选）",
    type=["xlsx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    all_dfs = []
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_type == "csv":
                df_tmp = pd.read_csv(uploaded_file)
            else:
                df_tmp = pd.read_excel(uploaded_file)
            df_tmp["source_file"] = uploaded_file.name
            all_dfs.append(df_tmp)
        except Exception as e:
            st.error(f"❌ 文件 {uploaded_file.name} 读取失败: {e}")

    df = pd.concat(all_dfs, ignore_index=True)
    st.success(f"✅ 成功加载 {len(uploaded_files)} 个文件，共 {len(df)} 条记录")

    # ====================== 数据清洗 ======================
    st.subheader("数据清洗与逻辑处理")
    required_cols = ["score", "solution", "service_attitude", "response_speed", "case_classification"]
    for c in required_cols:
        if c not in df.columns:
            st.error(f"❌ 缺少字段：{c}")
            st.stop()
    df = df.replace("-", np.nan)
    df = df[df["score"].notna()].copy()
    df["response_speed"] = df["response_speed"].fillna(1)

    def safe_to_int(x):
        if pd.isna(x):
            return 0
        s = str(x).strip().lower()
        if s in ["1", "1.0", "true", "t", "y", "yes"]:
            return 1
        elif s in ["0", "0.0", "false", "f", "n", "no"]:
            return 0
        else:
            return 0

    pass_cols = ["solution", "service_attitude", "response_speed", "case_classification"]
    for col in pass_cols:
        df[col] = df[col].apply(safe_to_int)

    df["satisfied"] = np.where(df["score"] >= 4, 1, 0)
    df["overall_pass"] = np.where((df[pass_cols].sum(axis=1) == 4), 1, 0)

    st.markdown("""
    ✅ **清洗逻辑说明：**
    - 剔除未打分记录；
    - `'-'` 自动识别为空；
    - `response_speed` 空视为通过；
    - 四项均为 1 为整体通过；
    - 打分 ≥ 4 判定为满意；
    - 支持多文件合并。
    """)

    # ====================== 汇总统计 ======================
    st.subheader("各项通过率与满意率")
    summary = df[pass_cols + ["overall_pass", "satisfied"]].mean().to_frame("rate")
    summary["rate"] = summary["rate"].apply(lambda x: round(x * 100, 2))
    st.dataframe(summary.T.style.format("{:.2f}%"))

    # ====================== 相关性分析 ======================
    st.subheader("相关性分析（Pearson）")
    fig_corr, ax_corr = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    sns.heatmap(df[pass_cols + ["satisfied"]].corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax_corr)
    ax_corr.set_title("Correlation between QC Items and Satisfaction")
    st.pyplot(fig_corr)

    # ====================== 显著性检验 ======================
    st.subheader("显著性差异分析（t 检验：通过组 vs 未通过组满意率）")
    result_list = []
    for col in pass_cols:
        passed = df[df[col] == 1]["satisfied"]
        failed = df[df[col] == 0]["satisfied"]
        if len(passed) > 2 and len(failed) > 2:
            t, p = stats.ttest_ind(passed, failed, equal_var=False)
            diff = passed.mean() - failed.mean()
            result_list.append([col, round(passed.mean(), 3), round(failed.mean(), 3), round(diff, 3), round(p, 4)])
    result_df = pd.DataFrame(result_list, columns=["指标项", "通过组满意率", "未通过组满意率", "差异", "p值"])
    st.dataframe(result_df)

    # ====================== Logistic 回归 ======================
    st.subheader("Logistic 回归分析")
    X = sm.add_constant(df[pass_cols])
    y = df["satisfied"]
    logit_model = sm.Logit(y, X).fit(disp=False)
    coef_df = pd.DataFrame({
        "指标项": logit_model.params.index[1:],
        "回归系数": logit_model.params.values[1:],
        "p值": logit_model.pvalues.values[1:]
    }).sort_values("回归系数", ascending=False)
    st.dataframe(coef_df.style.background_gradient(cmap="RdYlGn", axis=0))

    fig_bar, ax_bar = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    sns.barplot(x="回归系数", y="指标项", data=coef_df, ax=ax_bar)
    ax_bar.axvline(0, color="gray", linestyle="--")
    ax_bar.set_xlabel("Regression Coefficient")
    ax_bar.set_ylabel("QC Item")
    ax_bar.set_title("Impact of QC Items on Satisfaction (Logistic Coefficients)")
    for i, v in enumerate(coef_df["回归系数"].values):
        ax_bar.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.3f}",
                    va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    st.pyplot(fig_bar)

    # ====================== 时间趋势 ======================
st.subheader("时间趋势分析（按月）")
if "质检时间" in df.columns:
    dt = pd.to_datetime(df["质检时间"], errors="coerce")
    df["month"] = dt.dt.to_period("M").astype(str)

    trend_df = (
        df.dropna(subset=["month"])
          .groupby("month")[["satisfied", "overall_pass"]]
          .mean()
          .reset_index()
          .sort_values("month")
    )

    # 计算百分比
    trend_df["Satisfaction Rate (%)"] = (trend_df["satisfied"] * 100).round(2)
    trend_df["Pass Rate (%)"] = (trend_df["overall_pass"] * 100).round(2)

    # 绘图
    fig_trend, ax_trend = plt.subplots(figsize=(9, 4.5), dpi=150)
    x = np.arange(len(trend_df["month"]))
    y1 = trend_df["Satisfaction Rate (%)"]
    y2 = trend_df["Pass Rate (%)"]

    # 折线绘制
    ax_trend.plot(x, y1, marker="o", linewidth=2.2, label="Satisfaction Rate (%)", color="#1f77b4")
    ax_trend.plot(x, y2, marker="o", linewidth=2.2, label="Pass Rate (%)", color="#ff7f0e")

    # ========== 数据标签 ==========
    for i, (v1, v2) in enumerate(zip(y1, y2)):
        ax_trend.annotate(
            f"{v1:.1f}%", (x[i], v1),
            textcoords="offset points", xytext=(0, 6), ha="center",
            fontsize=8.5, color="#1f77b4"
        )
        ax_trend.annotate(
            f"{v2:.1f}%", (x[i], v2),
            textcoords="offset points", xytext=(0, -12), ha="center",
            fontsize=8.5, color="#ff7f0e"
        )

    # 坐标与样式
    ax_trend.set_xticks(x)
    ax_trend.set_xticklabels(trend_df["month"], rotation=30, ha="right", fontsize=9)
    ax_trend.set_ylabel("Percentage (%)", fontsize=9)
    ax_trend.set_title("Monthly Trend: Satisfaction vs Pass Rate", fontsize=11, pad=12)
    ax_trend.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    ax_trend.legend(fontsize=9, loc="best", frameon=True)
    st.pyplot(fig_trend)

    # 自动结论
    latest = trend_df.iloc[-1]
    delta_sat = latest["Satisfaction Rate (%)"] - trend_df.iloc[0]["Satisfaction Rate (%)"]
    delta_pass = latest["Pass Rate (%)"] - trend_df.iloc[0]["Pass Rate (%)"]
    msg = f"""
    **1️⃣ 当前整体满意率：** {latest["Satisfaction Rate (%)"]:.1f}%（较首月 {'↑' if delta_sat>=0 else '↓'} {abs(delta_sat):.1f}%）  
    **2️⃣ 当前整体质检通过率：** {latest["Pass Rate (%)"]:.1f}%（较首月 {'↑' if delta_pass>=0 else '↓'} {abs(delta_pass):.1f}%）  
    **3️⃣ 趋势关系：** {'同步上升 → 内部改进与客户感知一致。' if np.sign(delta_sat)==np.sign(delta_pass) else '方向不一致 → 可能存在标准与感知脱节。'}
    """
    st.markdown(msg)

 # ====================== 分业务线分析（英文显示） ======================
    if "business_line" in df.columns:
        st.subheader("分业务线分析")
    
        # 中文转英文映射
        biz_map = {
            "贸易线": "Trade Line",
            "品牌线": "Brand Line",
            "不清晰": "no clear",
        }
        df["business_line_en"] = df["business_line"].map(biz_map).fillna(df["business_line"])
    
        biz_df = (
            df.groupby(["business_line", "business_line_en"])[pass_cols + ["overall_pass", "satisfied"]]
            .mean()
            .apply(lambda x: round(x * 100, 2))
            .reset_index()
        )
    
        st.dataframe(biz_df[["business_line"] + pass_cols + ["overall_pass", "satisfied"]])
    
        fig_biz, ax_biz = plt.subplots(figsize=(8, 4.5), dpi=150)
        sns.scatterplot(
            data=biz_df,
            x="overall_pass", y="satisfied",
            hue="business_line_en", s=120, ax=ax_biz
        )
        for _, row in biz_df.iterrows():
            ax_biz.text(row["overall_pass"] + 0.3, row["satisfied"], row["business_line_en"], fontsize=9)
    
        ax_biz.set_xlabel("Overall Pass Rate (%)")
        ax_biz.set_ylabel("Satisfaction Rate (%)")
        ax_biz.set_title("Business Line: Pass Rate vs Satisfaction")
        ax_biz.legend(title="Business Line", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_biz.grid(alpha=0.2, linestyle="--")
        st.pyplot(fig_biz)


    # ====================== 分渠道分析 ======================
    if "ticket_channel" in df.columns:
        st.subheader("分渠道分析")
        ch_df = df.groupby("ticket_channel")[pass_cols + ["overall_pass", "satisfied"]].mean().apply(lambda x: round(x * 100, 2))
        ch_df_reset = ch_df.reset_index().rename(columns={"index": "ticket_channel"})
        st.dataframe(ch_df)
        fig_ch, ax_ch = plt.subplots(figsize=(8, 4.5), dpi=150)
        sns.scatterplot(data=ch_df_reset, x="overall_pass", y="satisfied", hue="ticket_channel", s=120, ax=ax_ch)
        for _, row in ch_df_reset.iterrows():
            ax_ch.text(row["overall_pass"] + 0.3, row["satisfied"], row["ticket_channel"], fontsize=9)
        ax_ch.set_xlabel("Overall Pass Rate (%)")
        ax_ch.set_ylabel("Satisfaction Rate (%)")
        ax_ch.set_title("Channel: Pass Rate vs Satisfaction")
        ax_ch.legend(title="Channel", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig_ch)

    st.success("✅ 分析完成，所有图表已优化显示。")

else:
    st.info("请上传多个质检文件后开始分析。")
