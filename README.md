# Student Performance Prediction and At-Risk Analysis Using Machine Learning

An end-to-end Machine Learning project to predict secondary school students' final mathematics examination grades (**G3**) and accurately identify **At-Risk** students (**G3 < 10** on a 0–20 grading scale) using the canonical UCI Student Performance dataset.

---

## 📌 Project Overview

Identifying academically vulnerable students early allows educators, counselors, and institutions to take targeted, proactive intervention steps (such as tutoring and attendance monitoring) before final examinations.

This project addresses two core predictive modeling tasks:
1. **Regression Task**: Predict the exact final grade **G3** (continuous scale from 0 to 20).
2. **Classification Task**: Classify students into **Not At Risk** ($G3 \ge 10$) or **At Risk** ($G3 < 10$, where 10 represents the experimental passing threshold).
3. **Early-Warning Scenario Analysis**: Compare model performance when period grades (**G1** and **G2**) are available versus an enrollment early-warning scenario where only demographics, attendance, family background, and study habits are known.

---

## 🗂️ Project Directory Structure

```text
student_performance/
├── data/
│   └── student-mat.csv         # UCI Student Performance Math dataset (395 rows, 33 columns, ';'-delimited)
├── outputs/                    # Auto-generated pipeline artifacts (git-ignored)
│   ├── models/                 # Serialized scikit-learn pipelines
│   │   ├── best_regressor.joblib
│   │   └── best_classifier.joblib
│   ├── plots/                  # Visualizations, EDA charts, and confusion matrices
│   │   ├── g3_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── at_risk_countplot.png
│   │   ├── regression_comparison.png
│   │   ├── classification_comparison.png
│   │   ├── feature_importance_rf.png
│   │   ├── reg_actual_vs_pred_*.png
│   │   └── clf_confusion_matrix_*.png
│   ├── reports/                # Tabular benchmarks and summary reports
│   │   ├── regression_results.csv
│   │   ├── classification_results.csv
│   │   └── summary_report.txt
│   └── predictions/            # Out-of-sample evaluation predictions
│       └── test_predictions.csv
├── utils.py                    # Shared utilities (data loader, preprocessor builder, defaults)
├── train.py                    # Complete CLI training and evaluation pipeline
├── app.py                      # Interactive Streamlit web application
├── requirements.txt            # Minimal, unpinned dependency list
├── .gitignore                  # Ignores outputs/ and Python cache files
└── README.md                   # Documentation and run instructions
```

---

## ⚙️ Installation & Setup

### 1. Clone or Open the Repository
```bash
git clone <your-repository-url>
cd student_performance
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
All libraries are standard data science packages:
```bash
pip install -r requirements.txt
```
*(Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`)*

---

## 🚀 How to Run

### Step 1: Run the Complete Training Pipeline (CLI)
Execute `train.py` from the command line:
```bash
python train.py
```
This script will:
- Validate the presence of `data/student-mat.csv` and report dataset shape, missing values, duplicates, and class balance.
- Create output directories automatically.
- Generate and save exploratory data analysis plots.
- Perform a stratified 80/20 train/test split (`random_state=42`).
- Fit scikit-learn `Pipeline` + `ColumnTransformer` models for Linear Regression, Decision Trees, and Random Forests.
- Evaluate models under both **Full Features (including G1 & G2)** and **Early-Warning (excluding G1 & G2)** scenarios.
- Generate actual-vs-predicted scatter plots, confusion matrices, comparison bar charts, and top-15 feature importance charts.
- Export all evaluation tables (`.csv`), a full summary report (`.txt`), test predictions (`.csv`), and persist the best models (`.joblib`).

### Step 2: Launch the Streamlit Web Application
Run the web application locally:
```bash
streamlit run app.py
```
The application opens in your default browser at `http://localhost:8501`.

#### Web Application Features:
- **Tab 1: Overview**: Dataset summary metrics, interactive data preview table, data quality indicators, and G3 distribution plots.
- **Tab 2: Data Analysis**: Correlation heatmap, interactive feature selector with automated regression/box plots against final grade G3.
- **Tab 3: Model Results**: Performance comparison tables with scenario filters, side-by-side actual vs. predicted plots, confusion matrices, top-15 feature importance, and summary findings.
- **Tab 4: Predict a Student**: Intuitive form to enter student characteristics, predict final grade G3 on a gauge, view At-Risk status with probability badges, and review grade trajectories.
- **Sidebar "Run Training" Button**: Allows triggering the training pipeline directly from the GUI without restarting the server.

---

## 📊 Summary of Experimental Results

Evaluated on an 80/20 stratified test set ($N=79$ test instances, 26 At-Risk, 53 Not At-Risk):

### 1. Regression Models (Predicting G3 Final Grade)

| Evaluation Scenario | Model | MAE | RMSE | $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **Full Features (incl. G1, G2)** | Linear Regression | 1.486 | 2.079 | 0.796 |
| **Full Features (incl. G1, G2)** | Decision Tree ($depth=5$) | 0.879 | 1.557 | 0.885 |
| **Full Features (incl. G1, G2)** | **Random Forest ($n=200$)** | **0.814** | **1.235** | **0.928** |
| *Early Warning (excl. G1, G2)* | Linear Regression | 3.399 | 4.370 | 0.098 |
| *Early Warning (excl. G1, G2)* | Decision Tree ($depth=5$) | 2.964 | 4.194 | 0.169 |
| *Early Warning (excl. G1, G2)* | Random Forest ($n=200$) | 2.934 | 4.026 | 0.234 |

*Best Regressor*: **Random Forest** (lowest RMSE = 1.235, $R^2$ = 0.928).

---

### 2. Classification Models (Identifying At-Risk Students, $G3 < 10$)

| Evaluation Scenario | Model | Precision | Recall | F1-Score | Balanced Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full Features (incl. G1, G2)** | Logistic Regression | 0.815 | 0.846 | 0.830 | 0.876 |
| **Full Features (incl. G1, G2)** | Decision Tree ($depth=5$) | 0.786 | 0.846 | 0.815 | 0.867 |
| **Full Features (incl. G1, G2)** | **Random Forest ($n=200$)** | **0.821** | **0.885** | **0.852** | **0.895** |
| *Early Warning (excl. G1, G2)* | Logistic Regression | 0.529 | 0.692 | 0.600 | 0.695 |
| *Early Warning (excl. G1, G2)* | Decision Tree ($depth=5$) | 0.474 | 0.346 | 0.400 | 0.579 |
| *Early Warning (excl. G1, G2)* | Random Forest ($n=200$) | 0.600 | 0.462 | 0.522 | 0.655 |

*Best Classifier*: **Random Forest** (chosen by highest At-Risk Recall = **88.5%**, F1 = **0.852**).

---

## 🔍 Key Findings: The Impact of Period Grades (G1 & G2)

1. **Extreme Predictive Power of Period Grades**:
   - First and second period grades (**G1** and **G2**) strongly correlate with final performance ($r \approx 0.80$ and $r \approx 0.90$).
   - When G1 and G2 are known, Random Forest explains **92.8%** of the variance in final grades and catches **88.5%** of failing students.
2. **Early-Warning Scenario Reality**:
   - Prior to mid-term grades, predicting exact grade values is substantially more challenging ($R^2$ drops to 0.234).
   - In the early-warning setting, **Logistic Regression with class balancing** is the most effective classifier for catching vulnerable students (Recall = **69.2%**), prioritizing student safety over raw precision.
   - The top early-warning predictors are **prior failures (`failures`)**, **school absences (`absences`)**, **mother's education level (`Medu`)**, and **study time (`studytime`)**.
3. **Institutional Workflow Recommendation**:
   - **At Enrollment (Term Start)**: Screen incoming cohorts using the Early-Warning model to identify students with elevated risk profiles (focusing on historical failures and attendance).
   - **Post Mid-Terms (After G1/G2)**: Transition to the Full-Feature model for precise final score forecasting and intervention allocation.

---

## ⚠️ Limitations & Ethical Considerations

- **Dataset Sample Size**: The dataset contains 395 students from two secondary schools in Portugal (Math course). While sufficient for demonstration and proof-of-concept modeling, larger multi-institutional datasets are necessary before deploying in production academic environments.
- **Temporal Grade Dependence**: High accuracy in the full model is heavily driven by past grades in the same academic subject. Models must not be used punitively or to create self-fulfilling prophecies for struggling students.
- **Socioeconomic Demographics**: Demographic features (such as parents' education and jobs) reflect structural disparities; models should inform proactive academic support and resource allocation rather than automated admissions or tracking decisions.

---

## 📤 Pushing to GitHub

To push this project to a remote Git repository:

1. **Initialize Git** (if not already initialized):
   ```bash
   git init
   ```

2. **Verify `.gitignore`**:
   Ensure `outputs/` and `__pycache__/` are excluded so that generated binaries, heavy models, and plots are not committed:
   ```text
   __pycache__/
   outputs/
   venv/
   ```

3. **Stage, Commit, and Push**:
   ```bash
   git add .
   git commit -m "feat: complete student performance prediction and at-risk analysis ML project"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
