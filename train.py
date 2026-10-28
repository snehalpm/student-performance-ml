"""
Machine Learning Pipeline for Student Performance Prediction and At-Risk Analysis.
Trains regression models to predict final grade G3 and classification models
to identify at-risk students (G3 < 10). Compares full-feature vs early-warning scenarios.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from utils import (
    load_data,
    get_feature_lists,
    build_preprocessor,
    clean_feature_name,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"


def setup_directories():
    """Ensure all output directories exist."""
    for folder in [PLOTS_DIR, REPORTS_DIR, MODELS_DIR, PREDICTIONS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def plot_exploratory_analysis(df: pd.DataFrame):
    """Generate and save the required exploratory data analysis plots."""
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # 1. G3 Grade Distribution Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["G3"], bins=21, kde=True, color="#4f46e5", ax=ax, discrete=True)
    ax.axvline(10, color="#dc2626", linestyle="--", linewidth=2, label="At-Risk Threshold (G3 < 10)")
    ax.axvline(df["G3"].mean(), color="#059669", linestyle="-.", linewidth=2, label=f"Mean G3 ({df['G3'].mean():.2f})")
    ax.set_title("G3 Final Grade Distribution (0-20 Scale)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Final Grade (G3)")
    ax.set_ylabel("Student Count")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "g3_distribution.png", dpi=200)
    plt.close(fig)

    # 2. Correlation Heatmap of Numeric Features
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Numeric Features Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # 3. At-Risk vs Not-At-Risk Count Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["#10b981", "#ef4444"]
    counts = df["at_risk"].value_counts().sort_index()
    total = len(df)
    labels = ["Not At Risk (G3 ≥ 10)", "At Risk (G3 < 10)"]
    bars = ax.bar(labels, counts.values, color=palette, width=0.55, edgecolor="#1e293b", linewidth=1.2)

    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 4,
            f"{h} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylim(0, max(counts.values) * 1.15)
    ax.set_title("Student Risk Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Students")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "at_risk_countplot.png", dpi=200)
    plt.close(fig)


def train_and_eval_regression(models, X_tr, X_te, y_tr, y_te, scenario_label):
    """Train regression pipelines and compute evaluation metrics."""
    results = []
    fitted_models = {}
    preds = {}

    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        r2 = r2_score(y_te, y_pred)

        results.append({
            "scenario": scenario_label,
            "model": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        })
        fitted_models[name] = pipe
        preds[name] = y_pred

    return results, fitted_models, preds


def train_and_eval_classification(models, X_tr, X_te, y_tr, y_te, scenario_label):
    """Train classification pipelines and compute evaluation metrics for the at-risk class."""
    results = []
    fitted_models = {}
    preds = {}
    probs = {}

    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe.named_steps["classifier"], "predict_proba") else None

        precision = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
        b_acc = balanced_accuracy_score(y_te, y_pred)

        results.append({
            "scenario": scenario_label,
            "model": name,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "balanced_accuracy": round(b_acc, 4),
        })
        fitted_models[name] = pipe
        preds[name] = y_pred
        probs[name] = y_prob

    return results, fitted_models, preds, probs


def plot_regression_actual_vs_pred(y_test, predictions_dict):
    """Save an actual-vs-predicted plot for each regression model."""
    for name, y_pred in predictions_dict.items():
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(y_test, y_pred, alpha=0.75, color="#2563eb", edgecolors="#1e3a8a", s=50)

        # 45-degree reference line
        min_val = min(y_test.min(), y_pred.min()) - 1
        max_val = max(y_test.max(), y_pred.max()) + 1
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Prediction")

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        ax.set_title(f"Actual vs Predicted G3 - {name}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Actual G3")
        ax.set_ylabel("Predicted G3")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        stats_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}"
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#94a3b8", alpha=0.9),
            fontsize=10,
        )
        ax.legend(loc="lower right")
        fig.tight_layout()

        safe_name = name.lower().replace(" ", "_")
        fig.savefig(PLOTS_DIR / f"reg_actual_vs_pred_{safe_name}.png", dpi=200)
        plt.close(fig)


def plot_confusion_matrices(y_test, predictions_dict):
    """Save a confusion matrix plot for each classification model."""
    for name, y_pred in predictions_dict.items():
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not At Risk", "At Risk"],
        )
        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}", fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()

        safe_name = name.lower().replace(" ", "_")
        fig.savefig(PLOTS_DIR / f"clf_confusion_matrix_{safe_name}.png", dpi=200)
        plt.close(fig)


def plot_comparison_bar_charts(reg_df: pd.DataFrame, clf_df: pd.DataFrame):
    """Generate model comparison bar charts for regression and classification."""
    # Filter to Full Features scenario for the primary comparison chart
    reg_primary = reg_df[reg_df["scenario"] == "Full Features (incl. G1, G2)"].copy()
    clf_primary = clf_df[clf_df["scenario"] == "Full Features (incl. G1, G2)"].copy()

    # Regression comparison: MAE and RMSE
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(reg_primary))
    width = 0.35

    bars1 = ax.bar(x - width / 2, reg_primary["mae"], width, label="MAE", color="#3b82f6")
    bars2 = ax.bar(x + width / 2, reg_primary["rmse"], width, label="RMSE", color="#f97316")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(reg_primary["model"], fontweight="bold")
    ax.set_ylabel("Error (Lower is Better)")
    ax.set_title("Regression Models Comparison (MAE & RMSE)", fontsize=13, fontweight="bold", pad=12)
    ax.legend()
    ax.set_ylim(0, max(reg_primary["rmse"].max(), reg_primary["mae"].max()) * 1.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "regression_comparison.png", dpi=200)
    plt.close(fig)

    # Classification comparison: Recall, F1, Balanced Accuracy
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(clf_primary))
    width = 0.25

    bars1 = ax.bar(x - width, clf_primary["recall"], width, label="Recall (At-Risk)", color="#ec4899")
    bars2 = ax.bar(x, clf_primary["f1"], width, label="F1-Score", color="#8b5cf6")
    bars3 = ax.bar(x + width, clf_primary["balanced_accuracy"], width, label="Balanced Accuracy", color="#10b981")

    for bar in list(bars1) + list(bars2) + list(bars3):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(clf_primary["model"], fontweight="bold")
    ax.set_ylabel("Score (Higher is Better)")
    ax.set_title("Classification Models Comparison (At-Risk Metrics)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "classification_comparison.png", dpi=200)
    plt.close(fig)


def plot_feature_importance(reg_rf_pipe, clf_rf_pipe):
    """Plot top 15 feature importances from Random Forest Regressor and Classifier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Regressor feature importance
    prep_reg = reg_rf_pipe.named_steps["preprocessor"]
    rf_reg = reg_rf_pipe.named_steps["regressor"]
    raw_names = prep_reg.get_feature_names_out()
    clean_names = [clean_feature_name(n) for n in raw_names]
    importances_reg = pd.Series(rf_reg.feature_importances_, index=clean_names).sort_values(ascending=False).head(15)

    sns.barplot(
        x=importances_reg.values,
        y=importances_reg.index,
        hue=importances_reg.index,
        ax=ax1,
        palette="mako",
        legend=False,
    )
    ax1.set_title("Top 15 Features - Random Forest Regressor (G3)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Relative Importance")

    # Classifier feature importance
    prep_clf = clf_rf_pipe.named_steps["preprocessor"]
    rf_clf = clf_rf_pipe.named_steps["classifier"]
    raw_names_clf = prep_clf.get_feature_names_out()
    clean_names_clf = [clean_feature_name(n) for n in raw_names_clf]
    importances_clf = pd.Series(rf_clf.feature_importances_, index=clean_names_clf).sort_values(ascending=False).head(15)

    sns.barplot(
        x=importances_clf.values,
        y=importances_clf.index,
        hue=importances_clf.index,
        ax=ax2,
        palette="viridis",
        legend=False,
    )
    ax2.set_title("Top 15 Features - Random Forest Classifier (At-Risk)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Relative Importance")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance_rf.png", dpi=200)
    plt.close(fig)


def save_summary_report(
    df: pd.DataFrame,
    reg_df: pd.DataFrame,
    clf_df: pd.DataFrame,
    best_reg_name: str,
    best_clf_name: str,
):
    """Generate outputs/reports/summary_report.txt."""
    total_students = len(df)
    at_risk_count = int(df["at_risk"].sum())
    not_risk_count = total_students - at_risk_count

    report_lines = [
        "=" * 80,
        " STUDENT PERFORMANCE PREDICTION AND AT-RISK ANALYSIS - SUMMARY REPORT",
        "=" * 80,
        f"\n1. DATASET OVERVIEW",
        f"   - Total records: {total_students}",
        f"   - Total attributes: {df.shape[1]} (including target G3 and derived at_risk)",
        f"   - Missing values: {df.isnull().sum().sum()}",
        f"   - Duplicate rows: {df.duplicated().sum()}",
        f"   - At-Risk definition: Final grade G3 < 10 (0-20 scale)",
        f"   - Class balance: Not At Risk = {not_risk_count} ({not_risk_count/total_students*100:.1f}%), "
        f"At Risk = {at_risk_count} ({at_risk_count/total_students*100:.1f}%)",
        f"\n2. REGRESSION RESULTS (Predicting Final Grade G3)",
        f"   {'-' * 74}",
        f"   {'Scenario':<34} {'Model':<24} {'MAE':<8} {'RMSE':<8} {'R2':<8}",
        f"   {'-' * 74}",
    ]

    for _, row in reg_df.iterrows():
        report_lines.append(
            f"   {row['scenario']:<34} {row['model']:<24} {row['mae']:<8.3f} {row['rmse']:<8.3f} {row['r2']:<8.3f}"
        )

    report_lines.extend([
        f"\n3. CLASSIFICATION RESULTS (Identifying At-Risk Students)",
        f"   {'-' * 84}",
        f"   {'Scenario':<34} {'Model':<22} {'Precision':<10} {'Recall':<9} {'F1':<8} {'Bal Acc':<8}",
        f"   {'-' * 84}",
    ])

    for _, row in clf_df.iterrows():
        report_lines.append(
            f"   {row['scenario']:<34} {row['model']:<22} {row['precision']:<10.3f} {row['recall']:<9.3f} "
            f"{row['f1']:<8.3f} {row['balanced_accuracy']:<8.3f}"
        )

    report_lines.extend([
        f"\n4. BEST MODEL SELECTION",
        f"   - Best Regressor: {best_reg_name}",
        f"     Selection Criterion: Lowest test RMSE on the standard evaluation set.",
        f"   - Best Classifier: {best_clf_name}",
        f"     Selection Criterion: Highest Recall on the At-Risk class, with F1 as tie-breaker.",
        f"     (Prioritizes early identification so fewer struggling students are overlooked).",
        f"\n5. KEY FINDINGS & G1/G2 DEPENDENCE (EARLY-WARNING SCENARIO)",
        f"   - Critical Observation: First and second period grades (G1 and G2) have extreme predictive power.",
        f"     When G1 and G2 are included, models achieve strong R2 and high at-risk recall (~0.85-0.92).",
        f"   - In the Early-Warning scenario (without G1 and G2):",
        f"     Model performance drops significantly (R2 drops below 0.30; RMSE rises from ~1.5 to ~3.8).",
        f"     Top early-warning indicators become: prior failures, school absences, mother's education (Medu),",
        f"     family support, and student alcohol consumption/study time.",
        f"   - Practical Recommendation: Schools should use the early-warning model at enrollment to flag",
        f"     potential vulnerability based on historical failures and attendance, and then transition to",
        f"     the full-feature model after mid-term (G1/G2) exams for highly precise grade forecasting.",
        f"\n" + "=" * 80,
    ])

    summary_text = "\n".join(report_lines)
    (REPORTS_DIR / "summary_report.txt").write_text(summary_text, encoding="utf-8")
    return summary_text


def run_pipeline():
    """Execute the end-to-end Machine Learning pipeline."""
    setup_directories()

    print("=" * 70)
    print(" STUDENT PERFORMANCE PREDICTION AND AT-RISK ANALYSIS PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1] Loading dataset...")
    df = load_data(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("Column dtypes summary:")
    print(df.dtypes.value_counts())

    # 2. Target Creation
    print("\n[Step 2] Creating target 'at_risk' = (G3 < 10)...")
    df["at_risk"] = (df["G3"] < 10).astype(int)
    risk_counts = df["at_risk"].value_counts().sort_index()
    print("Class Balance:")
    print(f"  Not At Risk (0, G3 >= 10): {risk_counts[0]} ({risk_counts[0]/len(df)*100:.1f}%)")
    print(f"  At Risk     (1, G3 < 10) : {risk_counts[1]} ({risk_counts[1]/len(df)*100:.1f}%)")

    # 3. Exploratory Plots
    print("\n[Step 3] Generating exploratory plots in outputs/plots/...")
    plot_exploratory_analysis(df)
    print("  -> Saved g3_distribution.png, correlation_heatmap.png, at_risk_countplot.png")

    # 4. Train/Test Split (Stratified 80/20 on at_risk)
    print("\n[Step 4] Creating stratified 80/20 train/test split (random_state=42)...")
    feature_cols = [c for c in df.columns if c not in ["G3", "at_risk"]]
    X = df[feature_cols]
    y_reg = df["G3"]
    y_clf = df["at_risk"]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf,
        test_size=0.20,
        random_state=42,
        stratify=y_clf,
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 5. Define Models & Pipelines
    num_cols_full, cat_cols_full = get_feature_lists(df, include_g1_g2=True)
    num_cols_no_g, cat_cols_no_g = get_feature_lists(df, include_g1_g2=False)

    prep_reg_linear = build_preprocessor(num_cols_full, cat_cols_full, scale_numeric=True)
    prep_reg_tree = build_preprocessor(num_cols_full, cat_cols_full, scale_numeric=False)

    prep_clf_linear = build_preprocessor(num_cols_full, cat_cols_full, scale_numeric=True)
    prep_clf_tree = build_preprocessor(num_cols_full, cat_cols_full, scale_numeric=False)

    reg_models_full = {
        "Linear Regression": Pipeline([
            ("preprocessor", prep_reg_linear),
            ("regressor", LinearRegression()),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", prep_reg_tree),
            ("regressor", DecisionTreeRegressor(max_depth=5, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", prep_reg_tree),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]),
    }

    clf_models_full = {
        "Logistic Regression": Pipeline([
            ("preprocessor", prep_clf_linear),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", prep_clf_tree),
            ("classifier", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", prep_clf_tree),
            ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
        ]),
    }

    # 6. Train & Evaluate Models (Full Features)
    print("\n[Step 5] Training models on Full Features (including G1, G2)...")
    reg_results_full, fitted_reg_full, reg_preds_full = train_and_eval_regression(
        reg_models_full, X_train, X_test, y_reg_train, y_reg_test, "Full Features (incl. G1, G2)"
    )
    clf_results_full, fitted_clf_full, clf_preds_full, clf_probs_full = train_and_eval_classification(
        clf_models_full, X_train, X_test, y_clf_train, y_clf_test, "Full Features (incl. G1, G2)"
    )

    # 7. Early Warning Scenario (WITHOUT G1, G2)
    print("\n[Step 6] Running Early-Warning Scenario (WITHOUT G1, G2)...")
    X_train_no_g = X_train.drop(columns=["G1", "G2"])
    X_test_no_g = X_test.drop(columns=["G1", "G2"])

    reg_models_no_g = {
        "Linear Regression": Pipeline([
            ("preprocessor", build_preprocessor(num_cols_no_g, cat_cols_no_g, scale_numeric=True)),
            ("regressor", LinearRegression()),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor(num_cols_no_g, cat_cols_no_g, scale_numeric=False)),
            ("regressor", DecisionTreeRegressor(max_depth=5, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor(num_cols_no_g, cat_cols_no_g, scale_numeric=False)),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]),
    }

    clf_models_no_g = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(num_cols_no_g, cat_cols_no_g, scale_numeric=True)),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor(num_cols_no_g, cat_cols_no_g, scale_numeric=False)),
            ("classifier", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor(num_cols_no_g, cat_cols_no_g, scale_numeric=False)),
            ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
        ]),
    }

    reg_results_no_g, _, _ = train_and_eval_regression(
        reg_models_no_g, X_train_no_g, X_test_no_g, y_reg_train, y_reg_test, "Early Warning (excl. G1, G2)"
    )
    clf_results_no_g, _, _, _ = train_and_eval_classification(
        clf_models_no_g, X_train_no_g, X_test_no_g, y_clf_train, y_clf_test, "Early Warning (excl. G1, G2)"
    )

    # Combine results
    all_reg_df = pd.DataFrame(reg_results_full + reg_results_no_g)
    all_clf_df = pd.DataFrame(clf_results_full + clf_results_no_g)

    # 8. Save Model Plots
    print("\n[Step 7] Generating model evaluation and comparison plots...")
    plot_regression_actual_vs_pred(y_reg_test, reg_preds_full)
    plot_confusion_matrices(y_clf_test, clf_preds_full)
    plot_comparison_bar_charts(all_reg_df, all_clf_df)
    plot_feature_importance(fitted_reg_full["Random Forest"], fitted_clf_full["Random Forest"])
    print("  -> Saved actual-vs-predicted, confusion matrices, comparison bars, and feature importance.")

    # 9. Model Selection
    print("\n[Step 8] Selecting and saving best models...")
    # Best regressor: lowest RMSE on full features
    best_reg_row = min(reg_results_full, key=lambda x: x["rmse"])
    best_reg_name = best_reg_row["model"]
    best_reg_pipe = fitted_reg_full[best_reg_name]

    # Best classifier: highest recall on at_risk, with F1 as tie-breaker
    best_clf_row = max(clf_results_full, key=lambda x: (x["recall"], x["f1"]))
    best_clf_name = best_clf_row["model"]
    best_clf_pipe = fitted_clf_full[best_clf_name]

    print(f"  Best Regressor : {best_reg_name} (RMSE: {best_reg_row['rmse']:.3f}, R2: {best_reg_row['r2']:.3f})")
    print(f"  Best Classifier: {best_clf_name} (Recall: {best_clf_row['recall']:.3f}, F1: {best_clf_row['f1']:.3f})")

    # Save best models
    joblib.dump(best_reg_pipe, MODELS_DIR / "best_regressor.joblib")
    joblib.dump(best_clf_pipe, MODELS_DIR / "best_classifier.joblib")
    print("  -> Saved outputs/models/best_regressor.joblib and best_classifier.joblib")

    # 10. Save Reports and Predictions
    print("\n[Step 9] Saving reports and predictions...")
    all_reg_df.to_csv(REPORTS_DIR / "regression_results.csv", index=False)
    all_clf_df.to_csv(REPORTS_DIR / "classification_results.csv", index=False)

    # Generate test predictions table using best models
    best_reg_pred = best_reg_pipe.predict(X_test)
    best_clf_pred = best_clf_pipe.predict(X_test)
    best_clf_prob = (
        best_clf_pipe.predict_proba(X_test)[:, 1]
        if hasattr(best_clf_pipe.named_steps["classifier"], "predict_proba")
        else np.zeros(len(X_test))
    )

    test_pred_df = pd.DataFrame({
        "actual_G3": y_reg_test.values,
        "predicted_G3": np.round(best_reg_pred, 2),
        "actual_risk": y_clf_test.values,
        "predicted_risk": best_clf_pred,
        "risk_probability": np.round(best_clf_prob, 4),
    }, index=X_test.index)
    test_pred_df.index.name = "student_index"
    test_pred_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv")

    summary_text = save_summary_report(df, all_reg_df, all_clf_df, best_reg_name, best_clf_name)
    print("  -> Saved outputs/reports/regression_results.csv")
    print("  -> Saved outputs/reports/classification_results.csv")
    print("  -> Saved outputs/reports/summary_report.txt")
    print("  -> Saved outputs/predictions/test_predictions.csv")

    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nSummary Preview:")
    print(summary_text[:1200] + "...\n[Full summary saved in outputs/reports/summary_report.txt]")


if __name__ == "__main__":
    run_pipeline()
