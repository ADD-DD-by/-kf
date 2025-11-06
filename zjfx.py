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
    **清洗逻辑说明：**
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

    # ====================== ✅ 两两组合交互分析 ======================
    st.subheader("两两组合对满意度的影响（交互项分析）")

    comb_results = []
    interaction_cols = []

    for i in range(len(pass_cols)):
        for j in range(i + 1, len(pass_cols)):
            c1, c2 = pass_cols[i], pass_cols[j]
            combo_name = f"{c1} × {c2}"
            df[combo_name] = df[c1] * df[c2]
            interaction_cols.append(combo_name)

            combo_group = (
                df.groupby(combo_name)["satisfied"]
                  .agg(["mean", "count"])
                  .rename(columns={"mean": "满意率", "count": "样本量"})
                  .reset_index()
            )

            if len(combo_group) == 2:
                diff = combo_group.loc[1, "满意率"] - combo_group.loc[0, "满意率"]
                t, p = stats.ttest_ind(
                    df[df[combo_name] == 1]["satisfied"],
                    df[df[combo_name] == 0]["satisfied"],
                    equal_var=False
                )
                comb_results.append({
                    "组合": combo_name,
                    "交互通过组满意率": round(combo_group.loc[1, "满意率"], 3),
                    "未交互组满意率": round(combo_group.loc[0, "满意率"], 3),
                    "差异": round(diff, 3),
                    "p值": round(p, 4)
                })

    combo_df = pd.DataFrame(comb_results)
    st.dataframe(combo_df)

    st.subheader("交互项 Logistic 回归分析")
    X_interact = sm.add_constant(df[pass_cols + interaction_cols])
    y = df["satisfied"]
    logit_interact = sm.Logit(y, X_interact).fit(disp=False)
    coef_inter_df = pd.DataFrame({
        "变量": logit_interact.params.index[1:],
        "回归系数": logit_interact.params.values[1:],
        "p值": logit_interact.pvalues.values[1:]
    }).sort_values("回归系数", ascending=False)
    st.dataframe(coef_inter_df.style.background_gradient(cmap="RdYlGn", axis=0))

    sig_inter = coef_inter_df[
        (coef_inter_df["p值"] < 0.05) &
        (coef_inter_df["变量"].isin(interaction_cols))
    ]
    if not sig_inter.empty:
        fig_int, ax_int = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        sns.barplot(x="回归系数", y="变量", data=sig_inter, ax=ax_int)
        ax_int.axvline(0, color="gray", linestyle="--")
        ax_int.set_title("显著交互项对满意度的影响")
        st.pyplot(fig_int)
    else:
        st.info("没有显著的两两交互项（p < 0.05）")

    # ====================== 自动结论生成模块 ======================
    st.subheader("📊 自动生成结论与质检标准优化建议")

    try:
        sig_items = coef_df[coef_df["p值"] < 0.05]
        if not sig_items.empty:
            key_item = sig_items.sort_values("回归系数", ascending=False).iloc[0]["指标项"]
            lowest_item = sig_items.sort_values("回归系数", ascending=True).iloc[0]["指标项"]

            st.markdown(f"""
            **1️⃣ 最显著提升满意度的质检项：** `{key_item}`  
            → 建议优先优化该项标准、强化一致性与执行深度。

            **2️⃣ 显示负向相关的质检项：** `{lowest_item}`  
            → 说明标准可能过严或定义模糊，建议复核打分逻辑。

            **3️⃣ 若发现“通过率上升但满意度下降”，需重点复盘：**
            - 可能是“标准偏离客户感知”；
            - 或“服务交付与质检判定不一致”；
            - 建议结合质检文本样本，细化二级维度定义。
            """)
        else:
            st.info("暂无显著性指标，当前数据不足以得出调整建议。")

        if not sig_inter.empty:
            inter_item = sig_inter.iloc[0]["变量"]
            st.markdown(f"""
            **4️⃣ 存在显著交互项：** `{inter_item}`  
            → 该组合（如“方案 × 响应速度”）对满意度影响显著，说明两项需联合考核。
            """)
    except Exception as e:
        st.warning(f"⚠️ 自动结论生成失败：{e}")

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

        trend_df["Satisfaction Rate (%)"] = (trend_df["satisfied"] * 100).round(2)
        trend_df["Pass Rate (%)"] = (trend_df["overall_pass"] * 100).round(2)

        fig_trend, ax_trend = plt.subplots(figsize=(9, 4.5), dpi=150)
        x = np.arange(len(trend_df["month"]))
        y1 = trend_df["Satisfaction Rate (%)"]
        y2 = trend_df["Pass Rate (%)"]

        ax_trend.plot(x, y1, marker="o", linewidth=2.2, label="Satisfaction Rate (%)", color="#1f77b4")
        ax_trend.plot(x, y2, marker="o", linewidth=2.2, label="Pass Rate (%)", color="#ff7f0e")

        for i, (v1, v2) in enumerate(zip(y1, y2)):
            ax_trend.annotate(f"{v1:.1f}%", (x[i], v1), textcoords="offset points", xytext=(0, 6),
                              ha="center", fontsize=8.5, color="#1f77b4")
            ax_trend.annotate(f"{v2:.1f}%", (x[i], v2), textcoords="offset points", xytext=(0, -12),
                              ha="center", fontsize=8.5, color="#ff7f0e")

        ax_trend.set_xticks(x)
        ax_trend.set_xticklabels(trend_df["month"], rotation=30, ha="right", fontsize=9)
        ax_trend.set_ylabel("Percentage (%)", fontsize=9)
        ax_trend.set_title("Monthly Trend: Satisfaction vs Pass Rate", fontsize=11, pad=12)
        ax_trend.grid(alpha=0.25, linestyle="--", linewidth=0.5)
        ax_trend.legend(fontsize=9, loc="best", frameon=True)
        st.pyplot(fig_trend)

    st.success("✅ 全部分析与自动结论生成完毕。")

else:
    st.info("请上传多个质检文件后开始分析。")
