# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# ========== 中文字体（mac 优化） ==========
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="质检与满意度分析看板", layout="wide")
st.title("多文件质检-满意度分析（含业务线、渠道、趋势与结论）")

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

    # 合并多个文件
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
    corr = df[pass_cols + ["overall_pass", "satisfied"]].corr()["satisfied"].sort_values(ascending=False)
    st.write(corr)

    fig_corr, ax_corr = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    sns.heatmap(df[pass_cols + ["satisfied"]].corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax_corr)
    ax_corr.set_title("质检项通过与满意度的相关性矩阵")
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
    X = df[pass_cols]
    X = sm.add_constant(X)
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
    ax_bar.set_title("各质检项对满意度的影响（Logistic 回归系数）")
    for i, v in enumerate(coef_df["回归系数"].values):
        ax_bar.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.3f}",
                    va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    st.pyplot(fig_bar)
# ====================== 各质检项与满意度的关系可视化 ======================
    st.subheader("不同质检项与满意度的关系")

    fig_scatter, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=150)
    axes = axes.flatten()

    for i, col in enumerate(pass_cols):
        sns.stripplot(
            data=df, x=col, y="score", ax=axes[i],
            jitter=0.25, alpha=0.5, palette="coolwarm"
        )
        axes[i].set_title(f"{col} vs 客户评分", fontsize=10)
        axes[i].set_xlabel("该项是否通过 (0=未通过, 1=通过)")
        axes[i].set_ylabel("客户评分")
        axes[i].grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    st.pyplot(fig_scatter)

    st.markdown("""
    **图形解读：**
    - 每个子图展示该项质检通过与否下的客户评分分布；
    - 若“通过=1”的评分明显更集中在高分段（4-5），说明该项对满意度有正向影响；
    - 若差异不明显，说明该项通过与否对客户感知影响较弱。
    """)

    # ====================== 时间趋势 ======================
    st.subheader("时间趋势分析（按月）")
    if "质检时间" in df.columns:
        dt = pd.to_datetime(df["质检时间"], format="%Y%m%d %H:%M:%S", errors="coerce")
        dt2 = pd.to_datetime(df["质检时间"], errors="coerce")
        df["质检时间_dt"] = dt.fillna(dt2)
        df["month"] = df["质检时间_dt"].dt.to_period("M").astype(str)

        trend_df = (
            df.dropna(subset=["month"])
              .groupby("month")[["satisfied", "overall_pass"]]
              .mean()
              .reset_index()
              .sort_values("month")
        )
        trend_df["满意率(%)"] = (trend_df["satisfied"] * 100).round(2)
        trend_df["整体通过率(%)"] = (trend_df["overall_pass"] * 100).round(2)

        fig_trend, ax_trend = plt.subplots(figsize=(9, 4.5), dpi=150)
        x = np.arange(len(trend_df["month"]))
        ax_trend.plot(x, trend_df["满意率(%)"], marker="o", linewidth=2, label="满意率(%)")
        ax_trend.plot(x, trend_df["整体通过率(%)"], marker="o", linewidth=2, label="整体通过率(%)")
        for i, v in enumerate(trend_df["满意率(%)"]):
            ax_trend.annotate(f"{v:.1f}%", (x[i], v), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
        for i, v in enumerate(trend_df["整体通过率(%)"]):
            ax_trend.annotate(f"{v:.1f}%", (x[i], v), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8)
        ax_trend.set_xticks(x)
        ax_trend.set_xticklabels(trend_df["month"], rotation=35, ha='right', fontsize=9)
        ax_trend.set_ylabel("比例（%）", fontsize=9)
        ax_trend.set_title("按月趋势：满意率 vs 整体质检通过率", fontsize=11)
        ax_trend.legend(fontsize=9)
        ax_trend.grid(alpha=0.2, linestyle="--")
        st.pyplot(fig_trend)

        st.subheader("自动结论解读")
        latest = trend_df.iloc[-1]
        delta_sat = latest["满意率(%)"] - trend_df.iloc[0]["满意率(%)"]
        delta_pass = latest["整体通过率(%)"] - trend_df.iloc[0]["整体通过率(%)"]
        msg = f"""
        **1️⃣ 当前整体满意率：** {latest["满意率(%)"]:.1f}%（较首月 {'↑' if delta_sat>=0 else '↓'} {abs(delta_sat):.1f}%）  
        **2️⃣ 当前整体质检通过率：** {latest["整体通过率(%)"]:.1f}%（较首月 {'↑' if delta_pass>=0 else '↓'} {abs(delta_pass):.1f}%）  
        **3️⃣ 趋势关系：** {'同步上升 → 内部改进与客户感知一致。' if np.sign(delta_sat)==np.sign(delta_pass) else '方向不一致 → 可能存在标准与感知脱节。'}  
        **4️⃣ 关键影响指标：** {coef_df.iloc[0]['指标项']}（回归系数 {coef_df.iloc[0]['回归系数']:.3f}）。
        """
        st.markdown(msg)

    # ====================== 分业务线分析 ======================
    if "business_line" in df.columns:
        st.subheader("分业务线分析")
        biz_df = (
            df.groupby("business_line")[pass_cols + ["overall_pass", "satisfied"]]
            .mean()
            .apply(lambda x: round(x * 100, 2))
        )
        st.dataframe(biz_df)
        fig_biz, ax_biz = plt.subplots(figsize=(8, 4.5), dpi=150)
        sns.scatterplot(data=biz_df, x="overall_pass", y="satisfied", hue=biz_df.index, s=120, ax=ax_biz)
        for i in biz_df.index:
            ax_biz.text(biz_df.loc[i, "overall_pass"]+0.3, biz_df.loc[i, "satisfied"], i, fontsize=9)
        ax_biz.set_xlabel("整体质检通过率（%）")
        ax_biz.set_ylabel("满意率（%）")
        ax_biz.set_title("不同业务线质检与满意度关系")
        ax_biz.grid(alpha=0.2, linestyle="--")
        st.pyplot(fig_biz)

    # ====================== 分渠道分析 ======================
    if "ticket_channel" in df.columns:
        st.subheader("分渠道分析")
        ch_df = (
            df.groupby("ticket_channel")[pass_cols + ["overall_pass", "satisfied"]]
            .mean()
            .apply(lambda x: round(x * 100, 2))
        )
        st.dataframe(ch_df)
        fig_ch, ax_ch = plt.subplots(figsize=(8, 4.5), dpi=150)
        sns.scatterplot(data=ch_df, x="overall_pass", y="satisfied", hue=ch_df.index, s=120, ax=ax_ch)
        for i in ch_df.index:
            ax_ch.text(ch_df.loc[i, "overall_pass"]+0.3, ch_df.loc[i, "satisfied"], i, fontsize=9)
        ax_ch.set_xlabel("整体质检通过率（%）")
        ax_ch.set_ylabel("满意率（%）")
        ax_ch.set_title("不同渠道质检与满意度关系")
        ax_ch.grid(alpha=0.2, linestyle="--")
        st.pyplot(fig_ch)

    st.success("✅ 分析完成。可通过业务线与渠道双维度洞察质检表现与客户感知。")

else:
    st.info("请上传多个质检文件后开始分析。")
