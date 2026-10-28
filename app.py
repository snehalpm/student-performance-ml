"""
Streamlit Web Application for Student Performance Prediction and At-Risk Analysis.
Interactive dashboard featuring dataset overview, exploratory visual analysis,
model performance benchmarks, and real-time student grade/risk prediction.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils import load_data, get_default_student
import train

# Page Configuration
st.set_page_config(
    page_title="Student Performance & At-Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

BEST_REG_PATH = MODELS_DIR / "best_regressor.joblib"
BEST_CLF_PATH = MODELS_DIR / "best_classifier.joblib"
REG_RESULTS_PATH = REPORTS_DIR / "regression_results.csv"
CLF_RESULTS_PATH = REPORTS_DIR / "classification_results.csv"
SUMMARY_REPORT_PATH = REPORTS_DIR / "summary_report.txt"

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .stat-lbl {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-badge-danger {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        border: 1px solid #f87171;
    }
    .risk-badge-safe {
        background-color: #d1fae5;
        color: #047857;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        border: 1px solid #34d399;
    }
    .report-box {
        background-color: #0f172a;
        color: #e2e8f0;
        padding: 1.25rem;
        border-radius: 10px;
        font-family: monospace;
        font-size: 0.88rem;
        white-space: pre-wrap;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)


# Cached Resource / Data Loaders
@st.cache_data
def get_dataset():
    """Load and cache the student performance dataset."""
    try:
        return load_data(DATA_PATH)
    except FileNotFoundError:
        return None


@st.cache_resource
def get_models():
    """Load and cache trained regression and classification models."""
    if BEST_REG_PATH.is_file() and BEST_CLF_PATH.is_file():
        try:
            reg = joblib.load(BEST_REG_PATH)
            clf = joblib.load(BEST_CLF_PATH)
            return reg, clf
        except Exception:
            return None, None
    return None, None


@st.cache_data
def get_results():
    """Load and cache evaluation result CSV tables."""
    reg_df = pd.read_csv(REG_RESULTS_PATH) if REG_RESULTS_PATH.is_file() else None
    clf_df = pd.read_csv(CLF_RESULTS_PATH) if CLF_RESULTS_PATH.is_file() else None
    summary_text = SUMMARY_REPORT_PATH.read_text(encoding="utf-8") if SUMMARY_REPORT_PATH.is_file() else ""
    return reg_df, clf_df, summary_text


# Load resources
df = get_dataset()
best_reg, best_clf = get_models()
reg_df, clf_df, summary_text = get_results()
models_ready = best_reg is not None and best_clf is not None

# Sidebar Controls
st.sidebar.title("🎓 Student ML Portal")
st.sidebar.markdown("**Predict Performance & Identify At-Risk Students**")

if models_ready:
    st.sidebar.success("🟢 **Models Ready & Trained**")
else:
    st.sidebar.warning("⚠️ **Models Not Trained Yet**")

st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Actions")

if st.sidebar.button("🚀 Run Training Pipeline", use_container_width=True, type="primary"):
    with st.spinner("Executing end-to-end Machine Learning pipeline..."):
        try:
            train.run_pipeline()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.sidebar.success("✅ Training completed successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error during training: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Context")
st.sidebar.info("""
**Grading Scale**: 0 to 20
**At-Risk Definition**: Final grade G3 < 10
**Dataset**: UCI Student Performance (Math)
**Sample Size**: 395 students, 33 attributes
""")


# Header
st.markdown('<div class="main-title">Student Performance Prediction & At-Risk Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Predicting student final grades (G3) and identifying academic vulnerability using Scikit-Learn Machine Learning pipelines.</div>',
    unsafe_allow_html=True,
)

if not models_ready:
    st.warning("⚠️ **Notice:** Trained models and evaluation artifacts were not detected. Please click **'🚀 Run Training Pipeline'** in the left sidebar to train the models and generate all charts and reports.")

# Main Navigation Tabs
tab_overview, tab_analysis, tab_results, tab_predict = st.tabs([
    "📊 Overview",
    "📈 Data Analysis",
    "🏆 Model Results",
    "🎯 Predict a Student",
])


# ==============================================================================
# TAB 1: OVERVIEW
# ==============================================================================
with tab_overview:
    if df is None:
        st.error(f"Dataset file missing at `{DATA_PATH}`. Please place `student-mat.csv` into `data/`.")
    else:
        # Key Summary Metrics
        total_students = len(df)
        at_risk_count = int((df["G3"] < 10).sum())
        not_risk_count = total_students - at_risk_count
        avg_g3 = df["G3"].mean()
        pass_rate = (not_risk_count / total_students) * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-lbl">Total Students</div>
                <div class="stat-val">{total_students}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-lbl">Average Final Grade (G3)</div>
                <div class="stat-val">{avg_g3:.2f} <span style="font-size:1rem;color:#64748b;">/ 20</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-lbl">At-Risk Students (G3 &lt; 10)</div>
                <div class="stat-val" style="color:#ef4444;">{at_risk_count} <span style="font-size:1rem;color:#ef4444;">({at_risk_count/total_students*100:.1f}%)</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-lbl">Passing Rate (G3 ≥ 10)</div>
                <div class="stat-val" style="color:#10b981;">{pass_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📋 Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)

        st.markdown("### 🔍 Data Quality & Structure")
        dq1, dq2, dq3 = st.columns(3)
        with dq1:
            st.metric("Total Rows / Instances", df.shape[0])
            st.metric("Total Columns / Features", df.shape[1])
        with dq2:
            st.metric("Missing (Null) Values", int(df.isnull().sum().sum()))
            st.metric("Duplicate Records", int(df.duplicated().sum()))
        with dq3:
            st.metric("Categorical Features", int((df.dtypes == "object").sum()))
            st.metric("Numeric Features", int((df.dtypes != "object").sum()))

        st.markdown("---")
        st.markdown("### 📉 Grade & Risk Distributions")
        g1_col, g2_col = st.columns(2)
        with g1_col:
            hist_path = PLOTS_DIR / "g3_distribution.png"
            if hist_path.is_file():
                st.image(str(hist_path), caption="Distribution of G3 final grades with at-risk threshold at 10.", use_container_width=True)
            else:
                st.info("Run training to view the G3 distribution chart.")
        with g2_col:
            count_path = PLOTS_DIR / "at_risk_countplot.png"
            if count_path.is_file():
                st.image(str(count_path), caption="Class balance: Not At-Risk vs At-Risk students.", use_container_width=True)
            else:
                st.info("Run training to view the class balance chart.")


# ==============================================================================
# TAB 2: DATA ANALYSIS
# ==============================================================================
with tab_analysis:
    if df is not None:
        st.markdown("### 🌐 Correlation Heatmap")
        heatmap_path = PLOTS_DIR / "correlation_heatmap.png"
        if heatmap_path.is_file():
            st.image(str(heatmap_path), caption="Correlation matrix for numeric features in the dataset.", use_container_width=True)
        else:
            st.info("Run training to view the correlation heatmap.")

        st.markdown("---")
        st.markdown("### 🔬 Interactive Feature Explorer vs Final Grade (G3)")
        st.write("Select any feature in the dataset to explore its relationship with the final grade G3.")

        all_features = [c for c in df.columns if c not in ["G3", "at_risk"]]
        selected_feature = st.selectbox(
            "Choose a feature to plot against G3:",
            options=all_features,
            index=all_features.index("G2") if "G2" in all_features else 0,
        )

        is_numeric = df[selected_feature].dtype != "object"

        col_plot, col_info = st.columns([1.6, 1])

        with col_plot:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.set_theme(style="whitegrid")

            if is_numeric:
                sns.regplot(
                    data=df,
                    x=selected_feature,
                    y="G3",
                    ax=ax,
                    scatter_kws={"alpha": 0.6, "color": "#3b82f6"},
                    line_kws={"color": "#ef4444", "linewidth": 2},
                )
                ax.axhline(10, color="#64748b", linestyle="--", linewidth=1, label="Passing (10)")
                ax.set_title(f"Relationship: {selected_feature} vs Final Grade (G3)", fontweight="bold")
                ax.legend()
            else:
                order = df.groupby(selected_feature)["G3"].median().sort_values(ascending=False).index
                sns.boxplot(
                    data=df,
                    x=selected_feature,
                    y="G3",
                    order=order,
                    ax=ax,
                    palette="Blues_r",
                    hue=selected_feature,
                    legend=False,
                )
                ax.axhline(10, color="#ef4444", linestyle="--", linewidth=1, label="Passing (10)")
                ax.set_title(f"G3 Grade Distribution by {selected_feature}", fontweight="bold")
                ax.legend()

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_info:
            st.markdown(f"#### Summary: `{selected_feature}`")
            if is_numeric:
                corr_val = df[selected_feature].corr(df["G3"])
                st.metric("Correlation with G3", f"{corr_val:.3f}")
                st.markdown("**Descriptive Statistics:**")
                st.dataframe(df[[selected_feature, "G3"]].describe().T[["mean", "std", "min", "50%", "max"]], use_container_width=True)
            else:
                group_stats = df.groupby(selected_feature)["G3"].agg(["count", "mean", "median", "std"]).reset_index()
                st.markdown("**G3 Statistics by Category:**")
                st.dataframe(group_stats, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 💡 Analytical Note: Period Grades Dominance")
        st.info("""
        **Strong Correlation Alert:** Period grades **G1** ($r \\approx 0.80$) and **G2** ($r \\approx 0.90$) have massive correlation with final grade **G3**.
        While this yields high predictive accuracy later in the academic year, it cannot provide early intervention at enrollment.
        Check the **Model Results** tab to compare performance **With G1/G2** against the **Early-Warning Scenario (Without G1/G2)**.
        """)


# ==============================================================================
# TAB 3: MODEL RESULTS
# ==============================================================================
with tab_results:
    if reg_df is None or clf_df is None:
        st.warning("⚠️ No model results found. Please run the training pipeline first by clicking 'Run Training Pipeline' in the sidebar.")
    else:
        st.markdown("### 🏆 Performance Comparison Tables")

        scenario_filter = st.radio(
            "Filter Evaluation Scenario:",
            options=["All Scenarios", "Full Features (incl. G1, G2)", "Early Warning (excl. G1, G2)"],
            horizontal=True,
        )

        filtered_reg = reg_df if scenario_filter == "All Scenarios" else reg_df[reg_df["scenario"] == scenario_filter]
        filtered_clf = clf_df if scenario_filter == "All Scenarios" else clf_df[clf_df["scenario"] == scenario_filter]

        r_col, c_col = st.columns(2)
        with r_col:
            st.markdown("#### 📉 Regression Models (Target: G3 Grade)")
            st.dataframe(
                filtered_reg.style.format({"mae": "{:.3f}", "rmse": "{:.3f}", "r2": "{:.3f}"})
                .highlight_min(subset=["mae", "rmse"], color="#d1fae5")
                .highlight_max(subset=["r2"], color="#d1fae5"),
                use_container_width=True,
            )

        with c_col:
            st.markdown("#### 🎯 Classification Models (Target: At-Risk Class)")
            st.dataframe(
                filtered_clf.style.format({
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1": "{:.3f}",
                    "balanced_accuracy": "{:.3f}"
                })
                .highlight_max(subset=["recall", "f1", "balanced_accuracy"], color="#d1fae5"),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### 📊 Visual Model Comparisons")
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            reg_cmp_path = PLOTS_DIR / "regression_comparison.png"
            if reg_cmp_path.is_file():
                st.image(str(reg_cmp_path), caption="Regression Models Error Comparison (Lower is Better)", use_container_width=True)
        with cmp2:
            clf_cmp_path = PLOTS_DIR / "classification_comparison.png"
            if clf_cmp_path.is_file():
                st.image(str(clf_cmp_path), caption="Classification Models Risk Metrics Comparison (Higher is Better)", use_container_width=True)

        st.markdown("---")
        st.markdown("### 🎯 Model Error Analysis: Actual vs Predicted & Confusion Matrices")

        st.markdown("#### Regression: Actual vs Predicted Final Grade G3")
        a1, a2, a3 = st.columns(3)
        with a1:
            p = PLOTS_DIR / "reg_actual_vs_pred_linear_regression.png"
            if p.is_file():
                st.image(str(p), use_container_width=True)
        with a2:
            p = PLOTS_DIR / "reg_actual_vs_pred_decision_tree.png"
            if p.is_file():
                st.image(str(p), use_container_width=True)
        with a3:
            p = PLOTS_DIR / "reg_actual_vs_pred_random_forest.png"
            if p.is_file():
                st.image(str(p), use_container_width=True)

        st.markdown("#### Classification: Confusion Matrices on Test Set")
        c1, c2, c3 = st.columns(3)
        with c1:
            p = PLOTS_DIR / "clf_confusion_matrix_logistic_regression.png"
            if p.is_file():
                st.image(str(p), use_container_width=True)
        with c2:
            p = PLOTS_DIR / "clf_confusion_matrix_decision_tree.png"
            if p.is_file():
                st.image(str(p), use_container_width=True)
        with c3:
            p = PLOTS_DIR / "clf_confusion_matrix_random_forest.png"
            if p.is_file():
                st.image(str(p), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🌲 Top 15 Feature Importances (Random Forest)")
        fi_path = PLOTS_DIR / "feature_importance_rf.png"
        if fi_path.is_file():
            st.image(str(fi_path), caption="Top 15 most influential predictors for regression and classification.", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📝 Full Summary Report")
        with st.expander("View outputs/reports/summary_report.txt", expanded=False):
            st.markdown(f'<div class="report-box">{summary_text}</div>', unsafe_allow_html=True)


# ==============================================================================
# TAB 4: PREDICT A STUDENT
# ==============================================================================
with tab_predict:
    st.markdown("### 🔮 Predict Individual Student Grade & Risk Status")
    st.write("Enter student profile data and academic history below to generate real-time grade and vulnerability forecasts.")

    if not models_ready:
        st.warning("⚠️ Trained models are required for live prediction. Please click 'Run Training Pipeline' in the sidebar.")
    else:
        defaults = get_default_student()

        with st.form("student_prediction_form"):
            st.markdown("#### 1. Prior Academic Performance")
            p1, p2, p3, p4, p5 = st.columns(5)
            with p1:
                g1_val = st.number_input("G1 Grade (0-20)", min_value=0, max_value=20, value=int(defaults["G1"]))
            with p2:
                g2_val = st.number_input("G2 Grade (0-20)", min_value=0, max_value=20, value=int(defaults["G2"]))
            with p3:
                failures_val = st.selectbox("Past Failures", options=[0, 1, 2, 3], index=int(defaults["failures"]))
            with p4:
                absences_val = st.number_input("Absences (0-75)", min_value=0, max_value=75, value=int(defaults["absences"]))
            with p5:
                studytime_val = st.selectbox(
                    "Study Time",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {1: "< 2 hrs/wk", 2: "2 - 5 hrs/wk", 3: "5 - 10 hrs/wk", 4: "> 10 hrs/wk"}[x],
                    index=int(defaults["studytime"]) - 1,
                )

            st.markdown("#### 2. School & Institutional Support")
            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                school_val = st.selectbox("School", options=["GP", "MS"], format_func=lambda x: "GP (Gabriel Pereira)" if x == "GP" else "MS (Mousinho da Silveira)")
            with s2:
                schoolsup_val = st.selectbox("School Support", options=["no", "yes"], index=0)
            with s3:
                famsup_val = st.selectbox("Family Support", options=["yes", "no"], index=0)
            with s4:
                paid_val = st.selectbox("Extra Paid Classes", options=["no", "yes"], index=0)
            with s5:
                higher_val = st.selectbox("Desire Higher Ed", options=["yes", "no"], index=0)

            st.markdown("#### 3. Family Background & Demographics")
            f1, f2, f3, f4, f5 = st.columns(5)
            with f1:
                age_val = st.number_input("Age", min_value=15, max_value=22, value=int(defaults["age"]))
            with f2:
                medu_val = st.selectbox("Mother Education (Medu)", options=[0, 1, 2, 3, 4], format_func=lambda x: {0: "0: None", 1: "1: 4th grade", 2: "2: 5-9th grade", 3: "3: Secondary", 4: "4: Higher"}[x], index=int(defaults["Medu"]))
            with f3:
                fedu_val = st.selectbox("Father Education (Fedu)", options=[0, 1, 2, 3, 4], format_func=lambda x: {0: "0: None", 1: "1: 4th grade", 2: "2: 5-9th grade", 3: "3: Secondary", 4: "4: Higher"}[x], index=int(defaults["Fedu"]))
            with f4:
                mjob_val = st.selectbox("Mother Job", options=["teacher", "health", "services", "at_home", "other"], index=4)
            with f5:
                fjob_val = st.selectbox("Father Job", options=["teacher", "health", "services", "at_home", "other"], index=4)

            st.markdown("#### 4. Lifestyle & Health")
            l1, l2, l3, l4, l5 = st.columns(5)
            with l1:
                goout_val = st.selectbox("Going Out", options=[1, 2, 3, 4, 5], format_func=lambda x: f"{x} / 5", index=2)
            with l2:
                dalc_val = st.selectbox("Workday Alcohol", options=[1, 2, 3, 4, 5], format_func=lambda x: f"{x} / 5", index=0)
            with l3:
                walc_val = st.selectbox("Weekend Alcohol", options=[1, 2, 3, 4, 5], format_func=lambda x: f"{x} / 5", index=0)
            with l4:
                health_val = st.selectbox("Current Health", options=[1, 2, 3, 4, 5], format_func=lambda x: f"{x} / 5", index=2)
            with l5:
                internet_val = st.selectbox("Internet Access", options=["yes", "no"], index=0)

            with st.expander("⚙️ Additional Profile Details (Defaults Pre-filled)", expanded=False):
                e1, e2, e3, e4, e5 = st.columns(5)
                with e1:
                    sex_val = st.selectbox("Sex", options=["F", "M"], index=0)
                    address_val = st.selectbox("Address", options=["U", "R"], format_func=lambda x: "Urban (U)" if x == "U" else "Rural (R)")
                with e2:
                    famsize_val = st.selectbox("Family Size", options=["GT3", "LE3"], format_func=lambda x: "> 3 (GT3)" if x == "GT3" else "≤ 3 (LE3)")
                    pstatus_val = st.selectbox("Parent Cohabitation", options=["T", "A"], format_func=lambda x: "Together (T)" if x == "T" else "Apart (A)")
                with e3:
                    reason_val = st.selectbox("Reason for School", options=["course", "home", "reputation", "other"])
                    guardian_val = st.selectbox("Guardian", options=["mother", "father", "other"])
                with e4:
                    activities_val = st.selectbox("Extracurriculars", options=["yes", "no"], index=0)
                    nursery_val = st.selectbox("Attended Nursery", options=["yes", "no"], index=0)
                with e5:
                    romantic_val = st.selectbox("Romantic Relationship", options=["no", "yes"], index=0)
                    famrel_val = st.selectbox("Family Relationship", options=[1, 2, 3, 4, 5], index=3)
                    freetime_val = st.selectbox("Free Time", options=[1, 2, 3, 4, 5], index=2)
                    traveltime_val = st.selectbox("Travel Time", options=[1, 2, 3, 4], index=0)

            submit_btn = st.form_submit_button("⚡ Predict Student Outcome", type="primary", use_container_width=True)

        if submit_btn:
            # Build input dictionary
            student_data = {
                "school": school_val,
                "sex": sex_val,
                "age": age_val,
                "address": address_val,
                "famsize": famsize_val,
                "Pstatus": pstatus_val,
                "Medu": medu_val,
                "Fedu": fedu_val,
                "Mjob": mjob_val,
                "Fjob": fjob_val,
                "reason": reason_val,
                "guardian": guardian_val,
                "traveltime": traveltime_val,
                "studytime": studytime_val,
                "failures": failures_val,
                "schoolsup": schoolsup_val,
                "famsup": famsup_val,
                "paid": paid_val,
                "activities": activities_val,
                "nursery": nursery_val,
                "higher": higher_val,
                "internet": internet_val,
                "romantic": romantic_val,
                "famrel": famrel_val,
                "freetime": freetime_val,
                "goout": goout_val,
                "Dalc": dalc_val,
                "Walc": walc_val,
                "health": health_val,
                "absences": absences_val,
                "G1": g1_val,
                "G2": g2_val,
            }

            input_df = pd.DataFrame([student_data])

            # Generate predictions
            predicted_g3 = float(best_reg.predict(input_df)[0])
            predicted_g3_clamped = max(0.0, min(20.0, predicted_g3))

            predicted_risk = int(best_clf.predict(input_df)[0])
            risk_prob = (
                float(best_clf.predict_proba(input_df)[0][1])
                if hasattr(best_clf.named_steps["classifier"], "predict_proba")
                else (1.0 if predicted_risk == 1 else 0.0)
            )

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown(f"""
                <div class="stat-card" style="border-left: 5px solid #3b82f6;">
                    <div class="stat-lbl">Predicted Final Grade (G3)</div>
                    <div class="stat-val" style="color: #2563eb;">{predicted_g3_clamped:.2f} <span style="font-size: 1.1rem; color: #64748b;">/ 20</span></div>
                    <p style="margin-top: 0.5rem; color: #64748b; font-size: 0.9rem;">
                        Scale: 0 to 20 &bull; Passing Grade Threshold: 10.0
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(1.0, max(0.0, predicted_g3_clamped / 20.0)))

            with res_col2:
                if predicted_risk == 1 or predicted_g3_clamped < 10.0:
                    badge_html = f"""
                    <div class="risk-badge-danger">
                        🚨 AT RISK OF FAILING (G3 &lt; 10)
                    </div>
                    """
                    card_border = "#ef4444"
                    status_text = f"High probability of academic failure: <b>{risk_prob*100:.1f}%</b> risk estimate."
                    recommendation = "<b>Recommended Intervention:</b> Immediate tutoring, attendance follow-up, and academic counseling."
                else:
                    badge_html = f"""
                    <div class="risk-badge-safe">
                        🛡️ NOT AT RISK (PASSING EXPECTED)
                    </div>
                    """
                    card_border = "#10b981"
                    status_text = f"Low probability of academic failure: <b>{risk_prob*100:.1f}%</b> risk estimate."
                    recommendation = "<b>Current Outlook:</b> Student is maintaining passing trajectory. Continue positive reinforcement."

                st.markdown(f"""
                <div class="stat-card" style="border-left: 5px solid {card_border};">
                    <div class="stat-lbl">Vulnerability Classification</div>
                    <div style="margin: 0.5rem 0;">{badge_html}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.92rem; color: #334155;">{status_text}</div>
                    <div style="margin-top: 0.3rem; font-size: 0.88rem; color: #475569;">{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)

            # Grade progression trajectory
            st.markdown("#### 📈 Grade Trajectory (G1 &rarr; G2 &rarr; Predicted G3)")
            traj_df = pd.DataFrame({
                "Evaluation Period": ["Period 1 (G1)", "Period 2 (G2)", "Predicted Final (G3)"],
                "Grade": [g1_val, g2_val, round(predicted_g3_clamped, 2)],
            })
            st.line_chart(traj_df, x="Evaluation Period", y="Grade", color="#4f46e5")
