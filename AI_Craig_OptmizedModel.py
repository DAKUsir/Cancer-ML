# PCA TOGGLE
USE_PCA          = False    # Set False for clearest SHAP/LIME + feature importance
PCA_N_COMPONENTS = 0.95

import os
import io
import base64
import warnings
import gradio as gr
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    make_scorer
)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from imblearn.over_sampling import BorderlineSMOTE

# SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP loaded ✓")
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed → pip install shap")

# LIME (optional)
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
    print("LIME loaded ✓")
except ImportError:
    LIME_AVAILABLE = False
    print("WARNING: lime not installed → pip install lime")

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.figsize": (10, 7), "font.size": 12})
sns.set_style("whitegrid")


# Feature Groups

VERY_HIGH_RISK = [
    "Age", "Smoking_Years", "Heavy_Smoker", "Age_Smoking",
    "Family_Cancer_History", "Chronic_Inflammation",
    "Cancer_Risk_Score", "Age_Inflammation", "Age_Family"
]
HIGH_RISK = [
    "BMI", "Obese", "Metabolic_Risk", "Poor_Lifestyle",
    "Lifestyle_Score", "Age_BMI", "Inflammation_Risk",
    "Has_MetSyn", "MetSyn_Score", "Triple_Risk",
    "Obese_Inflamed", "Obese_Diabetic", "Inflammation_Score"
]
MEDIUM_RISK = [
    "Alcohol_Drinks_Week", "Diet_Quality_Score", "Exercise_Hours_Week",
    "Sun_Exposure_Hours_Week", "Diabetes", "Hypertension",
    "Blood_Glucose_mg_dL", "Stress_Level", "Sleep_Hours_Per_Day",
    "Diabetes_Hypertension", "BMI_Glucose", "BMI_Smoking",
    "Smoking_Alcohol", "All_Risk_Count", "Lifestyle_Metabolic_Risk",
    "Prediabetic", "Diabetic_Uncontrolled", "Healthy_Lifestyle"
]
LOW_RISK = [
    "Gender", "Height_cm", "Weight_kg", "Cholesterol_mg_dL",
    "Systolic_BP", "Diastolic_BP", "BP_Diff", "Resting_Heart_Rate",
    "Hemoglobin_g_dL", "Previous_Surgery", "Pregnancies",
    "Last_Checkup_Months_Ago", "CV_Risk", "MAP",
    "Low_Hemoglobin", "High_Cholesterol", "CV_Risk_Extended",
    "Low_Checkup", "Missed_Screening",
    "Log_Smoking", "Log_Alcohol", "Log_Last_Checkup"
]
POLY_SEED_FEATURES = [
    "Age", "Smoking_Years", "BMI", "Family_Cancer_History",
    "Chronic_Inflammation", "Cancer_Risk_Score"
]

FEATURE_LABELS = {
    "Age": "Age", "Smoking_Years": "Years Smoking",
    "Heavy_Smoker": "Heavy Smoker (>=15 yrs)", "Age_Smoking": "Age x Smoking",
    "Family_Cancer_History": "Family Cancer History",
    "Chronic_Inflammation": "Chronic Inflammation",
    "Cancer_Risk_Score": "Composite Risk Score",
    "Age_Inflammation": "Age x Inflammation", "Age_Family": "Age x Family History",
    "BMI": "BMI", "Obese": "Obese (BMI>=30)", "Metabolic_Risk": "Metabolic Risk",
    "Poor_Lifestyle": "Poor Lifestyle", "Lifestyle_Score": "Lifestyle Score",
    "Age_BMI": "Age x BMI", "Inflammation_Risk": "Inflammation + Family Risk",
    "Has_MetSyn": "Metabolic Syndrome", "MetSyn_Score": "Met. Syndrome Score",
    "Triple_Risk": "Triple Risk Flag", "Obese_Inflamed": "Obese + Inflamed",
    "Obese_Diabetic": "Obese + Diabetic", "Inflammation_Score": "Inflammation Score",
    "Alcohol_Drinks_Week": "Alcohol (drinks/week)", "Diet_Quality_Score": "Diet Quality",
    "Exercise_Hours_Week": "Exercise (hrs/week)", "Diabetes": "Diabetes",
    "Hypertension": "Hypertension", "Blood_Glucose_mg_dL": "Blood Glucose",
    "Stress_Level": "Stress Level", "Sleep_Hours_Per_Day": "Sleep (hrs/day)",
    "All_Risk_Count": "Total Risk Count", "BMI_Smoking": "BMI x Smoking",
    "Smoking_Alcohol": "Smoking x Alcohol",
}


# Helper Functions

def load_and_prepare_data(filepath, target_col):
    df = pd.read_excel(filepath)
    df = df.loc[:, ~df.columns.str.startswith("Profession")]
    df = df.drop(columns=["Profession"], errors="ignore")
    X = df.drop(columns=["PatientID", "Cancer_Type", "Cancer_Stage", target_col], errors="ignore")
    y = df[target_col].astype(int)
    return X, y


def engineer_features(X):
    X = X.copy()
    X["Gender"] = (X["Gender"] == "Male").astype(int)
    X["Obese"]             = (X["BMI"] >= 30).astype(int)
    X["Metabolic_Risk"]    = X["Diabetes"] + X["Hypertension"]
    X["Heavy_Smoker"]      = (X["Smoking_Years"] >= 15).astype(int)
    X["Poor_Lifestyle"]    = ((X["Exercise_Hours_Week"] < 2).astype(int) +
                               (X["Diet_Quality_Score"] <= 4).astype(int))
    X["CV_Risk"]           = ((X["Systolic_BP"] >= 140).astype(int) +
                               (X["Cholesterol_mg_dL"] >= 240).astype(int))
    X["Inflammation_Risk"] = (X["Chronic_Inflammation"] & X["Family_Cancer_History"]).astype(int)
    X["BP_Diff"]           = X["Systolic_BP"] - X["Diastolic_BP"]
    X["Age_BMI"]           = X["Age"] * X["BMI"]
    X["Age_Smoking"]       = X["Age"] * X["Smoking_Years"]
    X["Lifestyle_Score"]   = X["Exercise_Hours_Week"] + X["Diet_Quality_Score"] - X["Stress_Level"]
    X["Age_Group"]             = pd.cut(X["Age"], bins=[0, 35, 45, 55, 65, 200],
                                         labels=[0, 1, 2, 3, 4]).astype(float)
    X["Senior"]                = (X["Age"] >= 60).astype(int)
    X["Prime_Age_Cancer_Risk"] = ((X["Age"] >= 45) & (X["Age"] <= 75)).astype(int)
    X["Age_Inflammation"]      = X["Age"] * X["Chronic_Inflammation"]
    X["Age_Family"]            = X["Age"] * X["Family_Cancer_History"]
    X["Age_Metabolic"]         = X["Age"] * X["Metabolic_Risk"]
    X["BMI_Category"]   = pd.cut(X["BMI"], bins=[0, 18.5, 25, 30, 35, 200],
                                  labels=[0, 1, 2, 3, 4]).astype(float)
    X["Severely_Obese"] = (X["BMI"] >= 35).astype(int)
    X["BMI_Glucose"]    = X["BMI"] * X["Blood_Glucose_mg_dL"] / 100.0
    X["BMI_Smoking"]    = X["BMI"] * X["Smoking_Years"]
    X["Obese_Diabetic"] = (X["Obese"] & X["Diabetes"]).astype(int)
    X["Obese_Inflamed"] = (X["Obese"] & X["Chronic_Inflammation"]).astype(int)
    X["Smoking_Squared"] = X["Smoking_Years"] ** 2 / 100.0
    X["Ever_Smoked"]     = (X["Smoking_Years"] > 0).astype(int)
    X["Long_Smoker"]     = (X["Smoking_Years"] >= 20).astype(int)
    X["Smoking_Alcohol"] = X["Smoking_Years"] * X["Alcohol_Drinks_Week"]
    X["BP_Category"] = pd.cut(X["Systolic_BP"], bins=[0, 120, 130, 140, 160, 500],
                               labels=[0, 1, 2, 3, 4]).astype(float)
    X["MAP"]         = (X["Systolic_BP"] + 2 * X["Diastolic_BP"]) / 3.0
    ms1 = (X["BMI"] >= 30).astype(int)
    ms2 = (X["Systolic_BP"] >= 130).astype(int)
    ms3 = (X["Blood_Glucose_mg_dL"] >= 100).astype(int)
    ms4 = (X["Cholesterol_mg_dL"] >= 200).astype(int)
    X["MetSyn_Score"]          = ms1 + ms2 + ms3 + ms4
    X["Has_MetSyn"]            = (X["MetSyn_Score"] >= 3).astype(int)
    X["Prediabetic"]           = ((X["Blood_Glucose_mg_dL"] >= 100) &
                                   (X["Blood_Glucose_mg_dL"] < 126)).astype(int)
    X["Diabetic_Uncontrolled"] = (X["Diabetes"] &
                                   (X["Blood_Glucose_mg_dL"] >= 126)).astype(int)
    X["Diabetes_Hypertension"]    = (X["Diabetes"] & X["Hypertension"]).astype(int)
    X["Triple_Risk"]              = (X["Family_Cancer_History"] &
                                      X["Chronic_Inflammation"] & X["Obese"]).astype(int)
    X["Lifestyle_Metabolic_Risk"] = X["Poor_Lifestyle"] + X["Metabolic_Risk"] + X["Has_MetSyn"]
    X["All_Risk_Count"]           = (X["Obese"] + X["Heavy_Smoker"] +
                                      X["Chronic_Inflammation"] + X["Family_Cancer_History"] +
                                      X["Diabetes"] + X["Hypertension"] + X["Senior"])
    X["High_Cholesterol"] = (X["Cholesterol_mg_dL"] >= 240).astype(int)
    X["Low_Hemoglobin"]   = (((X["Gender"] == 1) & (X["Hemoglobin_g_dL"] < 13.5)) |
                               ((X["Gender"] == 0) & (X["Hemoglobin_g_dL"] < 12.0))).astype(int)
    X["CV_Risk_Extended"] = X["CV_Risk"] + X["High_Cholesterol"] + X["Has_MetSyn"]
    X["Sedentary"]         = (X["Exercise_Hours_Week"] == 0).astype(int)
    X["Sleep_Deprived"]    = (X["Sleep_Hours_Per_Day"] < 6).astype(int)
    X["High_Stress"]       = (X["Stress_Level"] >= 8).astype(int)
    X["Healthy_Lifestyle"] = ((X["Exercise_Hours_Week"] >= 3).astype(int) +
                               (X["Diet_Quality_Score"] >= 7).astype(int) +
                               (X["Sleep_Hours_Per_Day"] >= 7).astype(int) +
                               (X["Stress_Level"] <= 4).astype(int))
    X["Low_Checkup"]      = (X["Last_Checkup_Months_Ago"] >= 24).astype(int)
    X["Missed_Screening"] = (X["Last_Checkup_Months_Ago"] >= 36).astype(int)
    X["Inflammation_Score"] = (X["Chronic_Inflammation"] + X["Obese_Inflamed"] +
                                X["Inflammation_Risk"] + X["Metabolic_Risk"])
    X["Cancer_Risk_Score"]  = (X["Age_Group"].fillna(0) * 0.8 +
                                X["Smoking_Years"] * 0.05 +
                                X["Family_Cancer_History"] * 2.0 +
                                X["Chronic_Inflammation"] * 1.5 +
                                X["Obese"] * 0.8 + X["Has_MetSyn"] * 0.7 +
                                X["Metabolic_Risk"] * 0.5 +
                                X["Poor_Lifestyle"] * 0.4 +
                                X["Low_Hemoglobin"] * 0.5)
    X["Log_Smoking"]      = np.log1p(X["Smoking_Years"])
    X["Log_Alcohol"]      = np.log1p(X["Alcohol_Drinks_Week"])
    X["Log_Last_Checkup"] = np.log1p(X["Last_Checkup_Months_Ago"])
    return X


def add_polynomial_features(X, seed_features=None, degree=2):
    if seed_features is None:
        seed_features = POLY_SEED_FEATURES
    available = [c for c in seed_features if c in X.columns]
    if not available:
        print("  WARNING: No polynomial seed features found — skipping poly step.")
        return X
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    poly_arr  = poly.fit_transform(X[available])
    poly_cols = poly.get_feature_names_out(available)
    new_cols  = [c for c in poly_cols if " " in c]
    poly_df   = pd.DataFrame(
        poly_arr[:, [list(poly_cols).index(c) for c in new_cols]],
        columns=[f"POLY_{c.replace(' ', '_x_')}" for c in new_cols],
        index=X.index
    )
    print(f"  Polynomial features added: {len(poly_df.columns)}")
    return pd.concat([X, poly_df], axis=1)


def apply_feature_weighting(X):
    for col in VERY_HIGH_RISK:
        if col in X.columns: X[col] *= 1.5
    for col in HIGH_RISK:
        if col in X.columns: X[col] *= 1.2
    for col in MEDIUM_RISK:
        if col in X.columns: X[col] *= 1.0
    for col in LOW_RISK:
        if col in X.columns: X[col] *= 0.8
    for col in X.columns:
        if col.startswith("POLY_"):
            X[col] *= 1.3
    return X


def get_score_dict(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "auc":       roc_auc_score(y_true, y_score),
        "y_pred":    y_pred
    }


def threshold_table(y_true, y_score):
    rows = []
    for t in np.linspace(0.05, 0.95, 101):
        s = get_score_dict(y_true, y_score, t)
        rows.append({"threshold": round(t, 2), "recall": s["recall"],
                     "precision": s["precision"], "f1": s["f1"], "accuracy": s["accuracy"]})
    return pd.DataFrame(rows)


def get_model_probability(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
        return (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)


def select_optimal_threshold(y_true, y_score, min_recall=0.87):
    tt = threshold_table(y_true, y_score)
    cand = tt[tt["recall"] >= min_recall]
    if not cand.empty:
        return cand.sort_values("precision", ascending=False).iloc[0]["threshold"]
    return tt.sort_values("f1", ascending=False).iloc[0]["threshold"]


def apply_pca_fit(X_array, n_components=PCA_N_COMPONENTS):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_array)
    print(f"    PCA: {X_array.shape[1]} features -> {X_pca.shape[1]} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")
    return X_pca, pca


def apply_pca_transform(X_array, pca):
    return pca.transform(X_array)


# Hyperparameter Tuning

def _multi_score_tuning(estimator, param_grid, X_train, y_train,
                         n_iter=25, cv=5, label="Model"):
    scoring = {
        "recall":    make_scorer(recall_score),
        "f1":        make_scorer(f1_score),
        "precision": make_scorer(precision_score, zero_division=0),
    }
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator, param_distributions=param_grid,
        n_iter=n_iter, cv=skf, scoring=scoring,
        refit="recall", n_jobs=-1, random_state=42, verbose=0,
        return_train_score=False
    )
    search.fit(X_train, y_train)
    res = pd.DataFrame(search.cv_results_)
    for metric in ["recall", "f1", "precision"]:
        col = f"mean_test_{metric}"
        best_row = res.loc[res[col].idxmax()]
        print(f"    [{label}] Best {metric:9s}: {best_row[col]:.3f}  "
              f"params={best_row['params']}")
    print(f"    [{label}] Chosen (recall): {search.best_score_:.3f}  "
          f"params={search.best_params_}")
    return search.best_estimator_


def tune_l1(X_train, y_train, cwd):
    print("  Tuning Logistic L1...")
    return _multi_score_tuning(
        LogisticRegression(penalty="l1", class_weight=cwd, random_state=42),
        {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
         "solver": ["liblinear", "saga"], "max_iter": [500, 1000, 2000]},
        X_train, y_train, n_iter=20, label="L1")


def tune_l2(X_train, y_train, cwd):
    print("  Tuning Logistic L2...")
    return _multi_score_tuning(
        LogisticRegression(penalty="l2", class_weight=cwd, random_state=42),
        {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
         "solver": ["lbfgs", "liblinear"], "max_iter": [500, 1000, 2000]},
        X_train, y_train, n_iter=20, label="L2")


def tune_rf(X_train, y_train, cwd):
    print("  Tuning RandomForest...")
    return _multi_score_tuning(
        RandomForestClassifier(class_weight=cwd, random_state=42, n_jobs=-1),
        {"n_estimators": [200, 300, 400, 500], "max_depth": [8, 10, 12, 15, None],
         "min_samples_leaf": [1, 2, 4], "min_samples_split": [2, 5, 10],
         "max_features": ["sqrt", "log2"]},
        X_train, y_train, n_iter=25, label="RF")


def tune_et(X_train, y_train, cwd):
    print("  Tuning ExtraTrees...")
    return _multi_score_tuning(
        ExtraTreesClassifier(class_weight=cwd, random_state=42, n_jobs=-1),
        {"n_estimators": [300, 400, 500, 600], "max_depth": [8, 10, 12, 15, None],
         "min_samples_leaf": [1, 2, 4], "min_samples_split": [2, 4, 8],
         "max_features": ["sqrt", "log2"], "bootstrap": [True, False]},
        X_train, y_train, n_iter=30, label="ET")


def tune_gb(X_train, y_train):
    print("  Tuning GradientBoosting...")
    return _multi_score_tuning(
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.2],
         "max_depth": [3, 4, 5, 6], "min_samples_leaf": [1, 2, 4],
         "subsample": [0.7, 0.8, 1.0]},
        X_train, y_train, n_iter=25, label="GB")


def tune_svm(X_train, y_train, cwd):
    print("  Tuning SVM...")
    return _multi_score_tuning(
        SVC(kernel="rbf", probability=True, class_weight=cwd, random_state=42),
        {"C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
         "gamma": ["scale", "auto", 0.001, 0.01, 0.1]},
        X_train, y_train, n_iter=20, label="SVM")


def tune_knn(X_train, y_train):
    print("  Tuning KNN...")
    return _multi_score_tuning(
        KNeighborsClassifier(n_jobs=-1),
        {"n_neighbors": [3, 5, 7, 9, 11, 15, 21],
         "weights": ["uniform", "distance"],
         "metric": ["euclidean", "manhattan", "minkowski"], "p": [1, 2]},
        X_train, y_train, n_iter=20, label="KNN")


# Model Assembly — 9 models total

def initialize_models(class_weight_dict, best_l1, best_l2,
                       best_rf, best_et, best_gb, best_svm, best_knn):

    # v2 Ensemble_L2_ET — preserved unchanged
    et_cw_v2 = {k: v * 2.0 if k == 1 else v for k, v in class_weight_dict.items()}
    l2_ens = LogisticRegression(
        penalty="l2", class_weight=class_weight_dict, random_state=42,
        **{k: v for k, v in best_l2.get_params().items()
           if k not in ["penalty", "class_weight", "random_state"]}
    )
    et_ens = ExtraTreesClassifier(
        class_weight=et_cw_v2, random_state=42, n_jobs=-1,
        **{k: v for k, v in best_et.get_params().items()
           if k not in ["class_weight", "random_state", "n_jobs"]}
    )
    ensemble_v2 = VotingClassifier(
        estimators=[("l2", l2_ens), ("et", et_ens)],
        voting="soft", weights=[2, 1]
    )

    # v3 HighRecall_Ensemble — new best model
    def _p(m, drop=("class_weight", "random_state", "n_jobs", "penalty")):
        return {k: v for k, v in m.get_params().items() if k not in drop}

    et_cw_hr = {k: (v * 2.5 if k == 1 else v) for k, v in class_weight_dict.items()}
    ensemble_hr = VotingClassifier(
        estimators=[
            ("l2", LogisticRegression(
                penalty="l2", class_weight=class_weight_dict,
                random_state=1, **_p(best_l2))),
            ("l1", LogisticRegression(
                penalty="l1", class_weight=class_weight_dict,
                random_state=2, solver="liblinear",
                **{k: v for k, v in best_l1.get_params().items()
                   if k not in ("penalty", "class_weight", "random_state", "solver")})),
            ("et", ExtraTreesClassifier(
                class_weight=et_cw_hr, random_state=3, n_jobs=-1,
                **_p(best_et))),
            ("rf", RandomForestClassifier(
                class_weight=class_weight_dict, random_state=4, n_jobs=-1,
                **_p(best_rf))),
        ],
        voting="soft", weights=[3, 2, 2, 1]
    )

    return {
        "Logistic_L1":         best_l1,
        "Logistic_L2":         best_l2,
        "RandomForest":        best_rf,
        "ExtraTrees":          best_et,
        "GradientBoosting":    best_gb,
        "SVM":                 best_svm,
        "KNN":                 best_knn,
        "Ensemble_L2_ET":      ensemble_v2,
        "HighRecall_Ensemble": ensemble_hr,
    }


# Plotting

def plot_confusion_matrix(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Cancer", "Cancer"],
                yticklabels=["No Cancer", "Cancer"], ax=ax)
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(
        f"Confusion Matrix — {model_name}\n"
        f"TP={tp}  FP={fp}  FN={fn}  TN={tn}  |  "
        f"Recall={tp/(tp+fn):.3f}  Precision={tp/(tp+fp):.3f}"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"cm_{model_name.replace(' ', '_')}.png", dpi=100)
    plt.show()


def plot_all_roc_curves(roc_store, models):
    plt.figure(figsize=(12, 8))
    for name in models.keys():
        mean_fpr = np.linspace(0, 1, 100)
        tprs, aucs = [], []
        for fpr, tpr, auc in roc_store[name]:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr, mean_tpr,
                 label=f"{name} (AUC={np.mean(aucs):.3f}+/-{np.std(aucs):.3f})",
                 linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models (5-Fold CV)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_all_models.png", dpi=100)
    plt.show()


def plot_all_pr_curves(pr_store, models):
    plt.figure(figsize=(12, 8))
    for name in models.keys():
        mean_rec = np.linspace(0, 1, 100)
        all_prec, aps = [], []
        for y_true, y_prob in pr_store[name]:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            all_prec.append(np.interp(mean_rec, rec[::-1], prec[::-1]))
            aps.append(average_precision_score(y_true, y_prob))
        mean_prec = np.mean(all_prec, axis=0)
        plt.plot(mean_rec, mean_prec,
                 label=f"{name} (AP={np.mean(aps):.3f}+/-{np.std(aps):.3f})",
                 linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — All Models (5-Fold CV)")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pr_all_models.png", dpi=100)
    plt.show()


def plot_metrics_comparison(summary_cv):
    metrics = ["Avg_Recall", "Avg_F1", "Avg_Precision", "Avg_AUC", "Avg_Accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 6))
    for ax, metric in zip(axes, metrics):
        ax.barh(summary_cv["Model"], summary_cv[metric],
                color=plt.cm.Set2(np.linspace(0, 1, len(summary_cv))))
        ax.set_xlim(0, 1)
        ax.set_title(metric.replace("Avg_", ""))
        ax.set_xlabel("Score")
        for i, v in enumerate(summary_cv[metric]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    plt.suptitle("Model Comparison — 5-Fold CV Metrics", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("model_comparison_cv.png", dpi=100, bbox_inches="tight")
    plt.show()


def plot_test_metrics_comparison(test_df):
    metrics = ["Recall", "F1", "Precision", "AUC", "Accuracy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 6))
    for ax, metric in zip(axes, metrics):
        ax.barh(test_df["Model"], test_df[metric],
                color=plt.cm.Set3(np.linspace(0, 1, len(test_df))))
        ax.set_xlim(0, 1)
        ax.set_title(metric)
        ax.set_xlabel("Score")
        for i, v in enumerate(test_df[metric]):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    plt.suptitle("Model Comparison — Test Set Metrics", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("model_comparison_test.png", dpi=100, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_columns, model_name, top_n=25):
    """Feature importance with real column names — saved + shown."""
    et = None
    if hasattr(model, "named_estimators_"):
        et = model.named_estimators_.get("et")
    elif hasattr(model, "feature_importances_"):
        et = model
    if et is None:
        return
    imp = (pd.Series(et.feature_importances_, index=feature_columns)
           .sort_values(ascending=False).head(top_n))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(imp)))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean Decrease Impurity")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fi_{model_name.replace(' ', '_')}.png", dpi=100)
    plt.show()


def plot_feature_importance_pca(model_name, model_template, feature_columns,
                                 X_prepca, y_train, top_n=25):
    """When PCA=True: re-fits tree on pre-PCA data so feature names stay readable."""
    src = None
    if hasattr(model_template, "named_estimators_"):
        src = (model_template.named_estimators_.get("et") or
               model_template.named_estimators_.get("rf"))
    elif hasattr(model_template, "feature_importances_"):
        src = model_template
    if src is None:
        print(f"  Skipping FI for {model_name} — no tree estimator found.")
        return
    fresh = src.__class__(**src.get_params())
    print(f"  Re-fitting {src.__class__.__name__} on pre-PCA data for '{model_name}'...")
    fresh.fit(X_prepca, y_train)
    imp = (pd.Series(fresh.feature_importances_, index=feature_columns)
           .sort_values(ascending=False).head(top_n))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(imp)))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean Decrease Impurity")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}\n"
                 f"(re-fitted on original features)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"fi_{model_name.replace(' ', '_')}.png", dpi=100)
    plt.show()


# Artifact I/O

def save_artifacts(models, scaler, feature_columns, numeric_cols,
                   thresholds, summary_cv, pca_model,
                   X_train_raw, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(models,          os.path.join(output_dir, "models_dict.pkl"))
    joblib.dump(scaler,          os.path.join(output_dir, "final_scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(output_dir, "feature_columns.pkl"))
    joblib.dump(numeric_cols,    os.path.join(output_dir, "numeric_cols_full.pkl"))
    joblib.dump(thresholds,      os.path.join(output_dir, "model_thresholds.pkl"))
    joblib.dump(summary_cv,      os.path.join(output_dir, "summary_cv.pkl"))
    joblib.dump(pca_model,       os.path.join(output_dir, "pca_model.pkl"))
    joblib.dump(X_train_raw,     os.path.join(output_dir, "X_train_raw.pkl"))
    summary_cv.to_csv(os.path.join(output_dir, "model_cv_summary.csv"), index=False)
    print(f"Saved all artifacts to {output_dir}/")


def load_artifacts(artifact_dir="artifacts"):
    models_dict     = joblib.load(os.path.join(artifact_dir, "models_dict.pkl"))
    scaler          = joblib.load(os.path.join(artifact_dir, "final_scaler.pkl"))
    feature_columns = joblib.load(os.path.join(artifact_dir, "feature_columns.pkl"))
    numeric_cols    = joblib.load(os.path.join(artifact_dir, "numeric_cols_full.pkl"))
    thresholds      = joblib.load(os.path.join(artifact_dir, "model_thresholds.pkl"))
    pca_model       = joblib.load(os.path.join(artifact_dir, "pca_model.pkl"))
    X_train_raw     = joblib.load(os.path.join(artifact_dir, "X_train_raw.pkl"))
    return models_dict, scaler, feature_columns, numeric_cols, thresholds, pca_model, X_train_raw


# Main Training Pipeline

print("=" * 70)
print("Cancer Risk Prediction — v3 FINAL")
print("9 models: L1, L2, RF, ET, GB, SVM, KNN, Ensemble_L2_ET, HighRecall_Ensemble")
print("=" * 70)
print(f"PCA enabled: {USE_PCA}" +
      (f"  (n_components={PCA_N_COMPONENTS})" if USE_PCA else ""))

TARGET = "Cancer_Binary"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X, y = load_and_prepare_data(os.path.join(_BASE_DIR, "AI_Criag_CleanedDataset.xlsx"), TARGET)

print("\n[1] Feature Engineering")
X = engineer_features(X)
print(f"  Features after base engineering : {X.shape[1]}")

print("[2] Polynomial Features")
X = add_polynomial_features(X, seed_features=POLY_SEED_FEATURES, degree=2)
print(f"  Total features after polynomial : {X.shape[1]}")

X = apply_feature_weighting(X)

processed_df = X.copy()
processed_df["Cancer_Binary"] = y.values
processed_df.to_csv("processed_dataset.csv", index=False)
print(f"\n[3] Processed dataset saved -> processed_dataset.csv  "
      f"({processed_df.shape[0]} rows x {processed_df.shape[1]} cols)")

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = BorderlineSMOTE(random_state=42, sampling_strategy=0.8)
X_train_full, y_train_full = smote.fit_resample(X_train_full, y_train_full)
print(f"[4] After BorderlineSMOTE — Train: {len(X_train_full)}, "
      f"Cancer rate: {y_train_full.mean():.3f}")

cw_values = compute_class_weight("balanced",
                                  classes=np.unique(y_train_full), y=y_train_full)
class_weight_dict = dict(zip(np.unique(y_train_full), cw_values))

tune_scaler = StandardScaler()
num_cols_t  = X_train_full.select_dtypes(include=np.number).columns
X_train_for_tuning = X_train_full.copy()
X_train_for_tuning[num_cols_t] = tune_scaler.fit_transform(X_train_full[num_cols_t])

tune_pca = None
if USE_PCA:
    print("\nApplying PCA to tuning data...")
    X_tune_array, tune_pca = apply_pca_fit(X_train_for_tuning.values)
    X_train_for_tuning = pd.DataFrame(X_tune_array)

print("\n[5] Hyperparameter Tuning (primary=recall, secondary=f1, tertiary=precision)")
et_cw_boost = {k: v * 2.0 if k == 1 else v for k, v in class_weight_dict.items()}
best_l1  = tune_l1(X_train_for_tuning, y_train_full, class_weight_dict)
best_l2  = tune_l2(X_train_for_tuning, y_train_full, class_weight_dict)
best_rf  = tune_rf(X_train_for_tuning, y_train_full, class_weight_dict)
best_et  = tune_et(X_train_for_tuning, y_train_full, et_cw_boost)
best_gb  = tune_gb(X_train_for_tuning, y_train_full)
best_svm = tune_svm(X_train_for_tuning, y_train_full, class_weight_dict)
best_knn = tune_knn(X_train_for_tuning, y_train_full)

models = initialize_models(class_weight_dict,
                            best_l1, best_l2, best_rf, best_et,
                            best_gb, best_svm, best_knn)
print(f"\n  Total models: {len(models)}  ->  {list(models.keys())}")


# 5-Fold Stratified Cross-Validation

kf            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_metrics = {name: [] for name in models}
roc_store     = {name: [] for name in models}
pr_store      = {name: [] for name in models}

print("\n[6] 5-Fold Stratified Cross-Validation")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full, y_train_full), 1):
    print(f"  Fold {fold}/5")
    X_tr  = X_train_full.iloc[tr_idx].copy()
    X_val = X_train_full.iloc[val_idx].copy()
    y_tr  = y_train_full.iloc[tr_idx]
    y_val = y_train_full.iloc[val_idx]

    numeric_cols = X_tr.select_dtypes(include=np.number).columns
    scaler_fold  = StandardScaler()
    X_tr[numeric_cols]  = scaler_fold.fit_transform(X_tr[numeric_cols])
    X_val[numeric_cols] = scaler_fold.transform(X_val[numeric_cols])

    fold_pca = None
    if USE_PCA:
        X_tr_arr, fold_pca = apply_pca_fit(X_tr.values)
        X_val_arr          = apply_pca_transform(X_val.values, fold_pca)
        X_tr  = pd.DataFrame(X_tr_arr)
        X_val = pd.DataFrame(X_val_arr)

    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_score   = get_model_probability(model, X_val)
        base_auc  = roc_auc_score(y_val, y_score)
        threshold = select_optimal_threshold(y_val, y_score, min_recall=0.87)
        scores    = get_score_dict(y_val, y_score, threshold)
        model_metrics[name].append({
            "threshold": threshold, "accuracy": scores["accuracy"],
            "precision": scores["precision"], "recall": scores["recall"],
            "f1": scores["f1"], "auc": base_auc
        })
        fpr, tpr, _ = roc_curve(y_val, y_score)
        roc_store[name].append((fpr, tpr, base_auc))
        pr_store[name].append((y_val.values, y_score))


# Aggregate CV Results

summary = []
for name, metrics in model_metrics.items():
    dfm = pd.DataFrame(metrics)
    summary.append({
        "Model":         name,
        "Avg_Threshold": round(dfm["threshold"].mean(), 2),
        "Avg_Accuracy":  round(dfm["accuracy"].mean(), 3),
        "Avg_Precision": round(dfm["precision"].mean(), 3),
        "Avg_Recall":    round(dfm["recall"].mean(), 3),
        "Avg_F1":        round(dfm["f1"].mean(), 3),
        "Avg_AUC":       round(dfm["auc"].mean(), 3),
        "Std_Recall":    round(dfm["recall"].std(), 3),
        "Std_F1":        round(dfm["f1"].std(), 3),
    })

summary_cv = pd.DataFrame(summary).sort_values(
    by=["Avg_Recall", "Avg_F1", "Avg_Precision"], ascending=False
).reset_index(drop=True)

print("\n=== Cross-Validation Summary (sorted: Recall > F1 > Precision) ===")
print(summary_cv.to_string(index=False))
summary_cv.to_csv("model_cv_summary.csv", index=False)

plot_all_roc_curves(roc_store, models)
plot_all_pr_curves(pr_store, models)
plot_metrics_comparison(summary_cv)


# Final Training + Test Set Evaluation

print("\n[7] Final Training on Full Train Set + Test Set Evaluation")

final_scaler = StandardScaler()
num_cols     = X_train_full.select_dtypes(include=np.number).columns
X_train_scaled = X_train_full.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[num_cols] = final_scaler.fit_transform(X_train_full[num_cols])
X_test_scaled[num_cols]  = final_scaler.transform(X_test[num_cols])

# Keep pre-PCA scaled data so FI charts always show real column names
X_train_scaled_pre_pca = X_train_scaled.copy()

final_pca = None
if USE_PCA:
    print("\nApplying PCA to final train/test sets...")
    X_train_arr, final_pca = apply_pca_fit(X_train_scaled.values)
    X_test_arr             = apply_pca_transform(X_test_scaled.values, final_pca)
    X_train_scaled = pd.DataFrame(X_train_arr)
    X_test_scaled  = pd.DataFrame(X_test_arr)

test_results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train_full)
    y_prob    = get_model_probability(model, X_test_scaled)
    threshold = summary_cv.loc[summary_cv["Model"] == name, "Avg_Threshold"].values[0]
    y_pred    = (y_prob >= threshold).astype(int)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    test_results.append({
        "Model": name, "Threshold": threshold,
        "Accuracy": round(acc, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "F1": round(f1, 3), "AUC": round(auc, 3)
    })
    print(f"\n{'─'*60}")
    print(f"  {name}  (threshold={threshold})")
    print(f"{'─'*60}")
    print(classification_report(y_test, y_pred, digits=3,
                                 target_names=["No Cancer", "Cancer"]))
    plot_confusion_matrix(y_test, y_pred, name)

test_df = pd.DataFrame(test_results).sort_values(
    by=["Recall", "F1", "Precision"], ascending=False
).reset_index(drop=True)
print("\n=== TEST SET SUMMARY (sorted: Recall > F1 > Precision) ===")
print(test_df.to_string(index=False))
test_df.to_csv("model_test_summary.csv", index=False)

plot_test_metrics_comparison(test_df)

# Feature importance — real names always
FI_MODELS = ["ExtraTrees", "RandomForest", "GradientBoosting",
             "Ensemble_L2_ET", "HighRecall_Ensemble"]
if not USE_PCA:
    for mn in FI_MODELS:
        plot_feature_importance(models[mn], X_train_full.columns, mn, top_n=25)
else:
    print("\nPCA ON — re-fitting trees on pre-PCA data for interpretable FI plots:")
    for mn in FI_MODELS:
        plot_feature_importance_pca(mn, models[mn],
                                    X_train_full.columns.tolist(),
                                    X_train_scaled_pre_pca.values,
                                    y_train_full, top_n=25)


# Save Artifacts

thresholds        = summary_cv.set_index("Model")["Avg_Threshold"].to_dict()
feature_columns   = X_train_full.columns.tolist()
numeric_cols_full = X_train_full.select_dtypes(include=np.number).columns.tolist()
# LIME always operates in original (pre-PCA) feature space
X_train_raw_for_lime = X_train_scaled_pre_pca.values

save_artifacts(models, final_scaler, feature_columns, numeric_cols_full,
               thresholds, summary_cv, final_pca, X_train_raw_for_lime,
               output_dir=os.path.join(_BASE_DIR, "artifacts"))

print("\n[8] All done — HighRecall_Ensemble is the recommended model.")


# SHAP + LIME Utilities

def _get_tree_from_model(model):
    if hasattr(model, "named_estimators_"):
        for key in ("et", "rf"):
            if key in model.named_estimators_:
                return model.named_estimators_[key]
    if hasattr(model, "feature_importances_"):
        return model
    return None


def compute_shap_values(model, X_instance, X_background, feature_names):
    """Always operates in original (pre-PCA) feature space."""
    if not SHAP_AVAILABLE:
        return None, None, None
    try:
        tree_model = _get_tree_from_model(model)
        if tree_model is not None:
            explainer = shap.TreeExplainer(tree_model)
            sv = explainer.shap_values(X_instance)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) == 2 else sv[0]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            ev = explainer.expected_value
            if not np.isscalar(ev):
                ev = ev[1] if len(ev) == 2 else ev[0]
            return sv[0], ev, feature_names
        elif hasattr(model, "coef_"):
            bg = shap.kmeans(X_background, 50)
            explainer = shap.LinearExplainer(model, bg)
            sv = explainer.shap_values(X_instance)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) == 2 else sv[0]
            ev = explainer.expected_value
            if not np.isscalar(ev):
                ev = ev[0]
            return sv[0], ev, feature_names
        else:
            bg = shap.kmeans(X_background, 30)
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], bg)
            sv = explainer.shap_values(X_instance, nsamples=100)
            return sv[0], explainer.expected_value, feature_names
    except Exception as e:
        print(f"SHAP error: {e}")
        return None, None, None


def fig_to_base64(fig):
    """Convert figure to base64 URI for Gradio HTML output."""
    import matplotlib
    prev_backend = matplotlib.get_backend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{enc}"


def plot_shap_bar(shap_vals, feature_names, top_n=15, prob=None):
    sv  = np.array(shap_vals)
    idx = np.argsort(np.abs(sv))[-top_n:][::-1]
    vals  = sv[idx]
    names = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in idx]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    ax.barh(range(len(vals))[::-1], vals, color=colors, height=0.65, edgecolor="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10, color="white")
    ax.set_xlabel("SHAP Value (impact on cancer prediction)", color="white", fontsize=10)
    title = "SHAP Feature Contributions"
    if prob is not None: title += f"  |  Cancer Risk: {prob:.1f}%"
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=12)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333333")
    ax.legend(handles=[mpatches.Patch(color="#e74c3c", label="Increases risk"),
                        mpatches.Patch(color="#3498db", label="Decreases risk")],
              loc="lower right", facecolor="#1a1d27", edgecolor="#444",
              labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def plot_lime_bar(lime_list, top_n=12, prob=None):
    lime_list = lime_list[:top_n]
    labels = [item[0] for item in lime_list]
    values = [item[1] for item in lime_list]
    colors = ["#e67e22" if v > 0 else "#1abc9c" for v in values]
    short  = [" ".join(l.split()[:4]) for l in labels]
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    ax.barh(range(len(values))[::-1], values, color=colors, height=0.65, edgecolor="none")
    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(short[::-1], fontsize=9, color="white")
    ax.set_xlabel("LIME Contribution (Cancer class)", color="white", fontsize=10)
    title = "LIME Feature Explanation"
    if prob is not None: title += f"  |  Cancer Risk: {prob:.1f}%"
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=12)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333333")
    ax.legend(handles=[mpatches.Patch(color="#e67e22", label="Toward Cancer"),
                        mpatches.Patch(color="#1abc9c", label="Toward No Cancer")],
              loc="lower right", facecolor="#1a1d27", edgecolor="#444",
              labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def build_risk_summary(shap_vals, feature_names, prob):
    sv = np.array(shap_vals)
    pos = [i for i in np.argsort(np.abs(sv))[-20:][::-1] if sv[i] > 0][:5]
    neg = [i for i in np.argsort(np.abs(sv))[-10:][::-1] if sv[i] < 0][:3]
    lvl = "HIGH" if prob >= 50 else "MODERATE" if prob >= 30 else "LOW"
    em  = "warning" if lvl == "HIGH" else "moderate" if lvl == "MODERATE" else "ok"
    lines = [f"**Predicted Cancer Risk: {prob:.1f}%** ({lvl} RISK)\n"]
    if pos:
        lines.append("**Top Risk-Increasing Factors:**")
        for rank, i in enumerate(pos, 1):
            lines.append(f"  {rank}. **{FEATURE_LABELS.get(feature_names[i], feature_names[i])}**"
                         f" — SHAP: +{sv[i]:.4f}")
    if neg:
        lines.append("\n**Protective Factors:**")
        for i in neg:
            lines.append(f"  - **{FEATURE_LABELS.get(feature_names[i], feature_names[i])}**"
                         f" — SHAP: {sv[i]:.4f}")
    lines.append("\n*Screening tool only — consult a healthcare professional.*")
    return "\n".join(lines)


# Gradio — Load Artifacts + Cache LIME

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
(models_dict, scaler, feature_columns, numeric_cols_full,
 thresholds, pca_model, X_train_raw) = load_artifacts(ARTIFACT_DIR)
model_names = list(models_dict.keys())
print("Loaded models:", model_names)

_lime_cache = {}

def _lime_explainer(feat_names):
    key = tuple(feat_names)
    if key not in _lime_cache and LIME_AVAILABLE:
        _lime_cache[key] = lime.lime_tabular.LimeTabularExplainer(
            X_train_raw, feature_names=feat_names,
            class_names=["No Cancer", "Cancer"],
            mode="classification", random_state=42, discretize_continuous=True
        )
    return _lime_cache.get(key)


def _prepare_input(Age, Gender, Height_cm, Weight_kg, Exercise_Hours_Week,
                   Smoking_Years, Alcohol_Drinks_Week, Diet_Quality_Score,
                   Family_Cancer_History, Diabetes, Hypertension,
                   Sun_Exposure_Hours_Week, Blood_Glucose_mg_dL, Cholesterol_mg_dL,
                   Last_Checkup_Months_Ago, Pregnancies, Systolic_BP, Diastolic_BP,
                   Resting_Heart_Rate, Hemoglobin_g_dL, Chronic_Inflammation,
                   Previous_Surgery, Sleep_Hours_Per_Day, Stress_Level):
    BMI = Weight_kg / (Height_cm / 100) ** 2
    data = {
        "Age": Age, "Gender": Gender, "Height_cm": Height_cm,
        "Weight_kg": Weight_kg, "BMI": round(BMI, 1),
        "Exercise_Hours_Week": Exercise_Hours_Week,
        "Smoking_Years": Smoking_Years, "Alcohol_Drinks_Week": Alcohol_Drinks_Week,
        "Diet_Quality_Score": Diet_Quality_Score,
        "Family_Cancer_History": 1 if Family_Cancer_History else 0,
        "Diabetes": 1 if Diabetes else 0, "Hypertension": 1 if Hypertension else 0,
        "Sun_Exposure_Hours_Week": Sun_Exposure_Hours_Week,
        "Blood_Glucose_mg_dL": Blood_Glucose_mg_dL, "Cholesterol_mg_dL": Cholesterol_mg_dL,
        "Last_Checkup_Months_Ago": Last_Checkup_Months_Ago, "Pregnancies": Pregnancies,
        "Systolic_BP": Systolic_BP, "Diastolic_BP": Diastolic_BP,
        "Resting_Heart_Rate": Resting_Heart_Rate, "Hemoglobin_g_dL": Hemoglobin_g_dL,
        "Chronic_Inflammation": 1 if Chronic_Inflammation else 0,
        "Previous_Surgery": 1 if Previous_Surgery else 0,
        "Sleep_Hours_Per_Day": Sleep_Hours_Per_Day, "Stress_Level": Stress_Level
    }
    Xp = pd.DataFrame([data])
    Xp = engineer_features(Xp)
    Xp = add_polynomial_features(Xp, seed_features=POLY_SEED_FEATURES, degree=2)
    Xp = apply_feature_weighting(Xp)
    Xp = Xp.reindex(columns=feature_columns, fill_value=0)
    return Xp


# Gradio Prediction Functions

def predict_with_model(model_name,
                        Age, Gender, Height_cm, Weight_kg, Exercise_Hours_Week,
                        Smoking_Years, Alcohol_Drinks_Week, Diet_Quality_Score,
                        Family_Cancer_History, Diabetes, Hypertension,
                        Sun_Exposure_Hours_Week, Blood_Glucose_mg_dL, Cholesterol_mg_dL,
                        Last_Checkup_Months_Ago, Pregnancies, Systolic_BP, Diastolic_BP,
                        Resting_Heart_Rate, Hemoglobin_g_dL, Chronic_Inflammation,
                        Previous_Surgery, Sleep_Hours_Per_Day, Stress_Level):

    Xp = _prepare_input(Age, Gender, Height_cm, Weight_kg, Exercise_Hours_Week,
                         Smoking_Years, Alcohol_Drinks_Week, Diet_Quality_Score,
                         Family_Cancer_History, Diabetes, Hypertension,
                         Sun_Exposure_Hours_Week, Blood_Glucose_mg_dL, Cholesterol_mg_dL,
                         Last_Checkup_Months_Ago, Pregnancies, Systolic_BP, Diastolic_BP,
                         Resting_Heart_Rate, Hemoglobin_g_dL, Chronic_Inflammation,
                         Previous_Surgery, Sleep_Hours_Per_Day, Stress_Level)

    # Scale in original feature space (always)
    X_scaled   = scaler.transform(Xp[numeric_cols_full].values)   # (1, n_features)
    feat_names = numeric_cols_full                                  # real names

    # Apply PCA only for prediction — SHAP/LIME stay in original space
    X_input = apply_pca_transform(X_scaled, pca_model) if pca_model is not None else X_scaled

    model    = models_dict[model_name]
    prob     = get_model_probability(model, X_input)[0]
    thresh   = thresholds.get(model_name, 0.5)
    label    = "HIGH Cancer Risk" if prob >= thresh else "LOW Cancer Risk"
    prob_pct = round(float(prob * 100), 2)

    # SHAP (original feature space via wrapper if PCA was used)
    shap_html  = ("<p style='color:#888;padding:12px;'>"
                  "SHAP not available — run: pip install shap</p>")
    summary_md = ""
    if SHAP_AVAILABLE:
        shap_model = model
        if pca_model is not None:
            class _PCAWrap:
                def __init__(self, m, pca):
                    self.m = m; self.pca = pca
                def predict_proba(self, X):
                    return self.m.predict_proba(apply_pca_transform(X, self.pca))
            shap_model = _PCAWrap(model, pca_model)
        sv, ev, fn = compute_shap_values(shap_model, X_scaled, X_train_raw, feat_names)
        if sv is not None:
            shap_html  = (f'<img src="{fig_to_base64(plot_shap_bar(sv, fn, 15, prob_pct))}"'
                          f' style="width:100%;border-radius:8px;margin-top:8px;">')
            summary_md = build_risk_summary(sv, fn, prob_pct)

    # LIME (original feature space; wrapper handles PCA internally)
    lime_html = ("<p style='color:#888;padding:12px;'>"
                 "LIME not available — run: pip install lime</p>")
    if LIME_AVAILABLE:
        try:
            exp_obj = _lime_explainer(feat_names)
            if exp_obj:
                if pca_model is not None:
                    pred_fn = lambda x: np.column_stack([
                        1 - model.predict_proba(apply_pca_transform(x, pca_model))[:, 1],
                            model.predict_proba(apply_pca_transform(x, pca_model))[:, 1]
                    ])
                else:
                    pred_fn = lambda x: model.predict_proba(x)
                exp = exp_obj.explain_instance(
                    X_scaled[0], pred_fn, num_features=12, num_samples=800, labels=(1,))
                lime_list = exp.as_list(label=1)
                if lime_list:
                    lime_html = (f'<img src="{fig_to_base64(plot_lime_bar(lime_list, 12, prob_pct))}"'
                                 f' style="width:100%;border-radius:8px;margin-top:8px;">')
        except Exception as e:
            lime_html = f"<p style='color:#e74c3c;padding:12px;'>LIME error: {e}</p>"

    if not summary_md:
        lvl = "HIGH" if prob_pct >= 50 else "MODERATE" if prob_pct >= 30 else "LOW"
        summary_md = (f"**Predicted Cancer Risk: {prob_pct:.1f}%** ({lvl} RISK)\n\n"
                      "Install `shap` and `lime` for detailed explanations.\n\n"
                      "*Screening tool only — consult a healthcare professional.*")

    return label, prob_pct, float(thresh), summary_md, shap_html, lime_html


def predict_batch(model_name, csv_file):
    if csv_file is None:
        return None, "Please upload a CSV file"
    try:
        df_input  = pd.read_csv(csv_file.name)
        df_output = df_input.copy()
        Xp        = df_input.copy()

        if "Gender" in Xp.columns:
            if Xp["Gender"].dtype == "object" or Xp["Gender"].dtype.name == "category":
                Xp["Gender"] = Xp["Gender"].astype(str).str.strip()
                gmap = {"male":"Male","Male":"Male","MALE":"Male","M":"Male","m":"Male",
                        "female":"Female","Female":"Female","FEMALE":"Female",
                        "F":"Female","f":"Female","1":"Male","1.0":"Male",
                        "0":"Female","0.0":"Female"}
                Xp["Gender"] = Xp["Gender"].map(gmap).fillna("Female")
            else:
                Xp["Gender"] = Xp["Gender"].apply(
                    lambda v: "Male" if int(v) == 1 else "Female")

        for col in ["Family_Cancer_History","Diabetes","Hypertension",
                    "Chronic_Inflammation","Previous_Surgery"]:
            if col in Xp.columns:
                if Xp[col].dtype == "object" or Xp[col].dtype.name == "category":
                    Xp[col] = Xp[col].astype(str).str.strip().str.lower()
                    Xp[col] = Xp[col].map({"yes":1,"true":1,"1":1,"1.0":1,
                                            "no":0,"false":0,"0":0,"0.0":0}
                                           ).fillna(0).astype(int)
                else:
                    Xp[col] = Xp[col].astype(int)

        Xp = engineer_features(Xp)
        Xp = add_polynomial_features(Xp, seed_features=POLY_SEED_FEATURES, degree=2)
        Xp = apply_feature_weighting(Xp)
        Xp = Xp.reindex(columns=feature_columns, fill_value=0)
        X_scaled = scaler.transform(Xp[numeric_cols_full])
        if pca_model is not None:
            X_scaled = apply_pca_transform(X_scaled, pca_model)

        model  = models_dict[model_name]
        probs  = get_model_probability(model, X_scaled)
        thresh = thresholds.get(model_name, 0.5)
        preds  = ["High Cancer Risk" if p >= thresh else "Low Cancer Risk"
                  for p in probs]
        df_output["Prediction"]         = preds
        df_output["Risk_Probability_%"] = (probs * 100).round(2)
        df_output["Threshold_Used"]     = thresh
        out = "batch_predictions.csv"
        df_output.to_csv(out, index=False)
        return out, f"Successfully processed {len(df_output)} records"
    except Exception as e:
        return None, f"Error: {str(e)}"


# Gradio UI

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #080b12 !important;
    color: #d4d8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.block, .form {
    background: #0e1320 !important;
    border: 1px solid #1e2438 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease !important;
}
.block:hover { border-color: #2e3550 !important; }

.label-wrap span, label span {
    color: #7b84a8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

input[type="number"], input[type="text"], textarea, select {
    background: #131825 !important;
    border: 1px solid #1e2438 !important;
    border-radius: 8px !important;
    color: #e0e4f4 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
input[type="number"]:focus, input[type="text"]:focus, textarea:focus {
    border-color: #dc3232 !important;
    box-shadow: 0 0 0 3px rgba(220,50,50,0.12) !important;
    outline: none !important;
}

input[type="range"]    { accent-color: #dc3232 !important; }
input[type="checkbox"] { accent-color: #dc3232 !important; transform: scale(1.2) !important; }
input[type="radio"]    { accent-color: #dc3232 !important; }

.wrap-inner, .multiselect {
    background: #131825 !important;
    border-color: #1e2438 !important;
    border-radius: 8px !important;
}

button.primary, .btn-primary {
    background: linear-gradient(135deg, #b02020 0%, #dc3232 60%, #e84a4a 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(220,50,50,0.35) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(220,50,50,0.5) !important;
}
button.primary:active { transform: translateY(0) !important; }

button.secondary {
    background: #131825 !important;
    border: 1px solid #1e2438 !important;
    border-radius: 8px !important;
    color: #7b84a8 !important;
}

.tabs > .tab-nav {
    border-bottom: 1px solid #1e2438 !important;
    gap: 4px !important;
}
.tabs > .tab-nav button {
    color: #7b84a8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.03em !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease !important;
}
.tabs > .tab-nav button:hover { color: #c0c8e0 !important; background: #0e1320 !important; }
.tabs > .tab-nav button.selected {
    color: #f05050 !important;
    border-bottom-color: #dc3232 !important;
    background: #0e1320 !important;
    font-weight: 600 !important;
}

.output-class, .output-textbox, .textbox {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}

h1, h2, h3 { color: #f05050 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080b12; }
::-webkit-scrollbar-thumb { background: #1e2438; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #dc3232; }
"""

_HEADER = (
    '<div style="background:linear-gradient(135deg,#0a0d18 0%,#120810 40%,#0e1320 100%);'
    'padding:28px 32px 24px;border-radius:16px;margin-bottom:10px;border:1px solid #1e2438;'
    'position:relative;overflow:hidden;">'

    # decorative glow
    '<div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;'
    'background:radial-gradient(circle,rgba(220,50,50,0.15) 0%,transparent 70%);'
    'pointer-events:none;"></div>'

    # title row
    '<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">'
    '<div style="background:linear-gradient(135deg,#b02020,#e84a4a);border-radius:10px;'
    'padding:10px 12px;font-size:1.5rem;line-height:1;box-shadow:0 4px 16px rgba(220,50,50,0.4);">&#129516;</div>'
    '<div>'
    '<div style="margin:0;font-size:1.65rem;font-weight:700;color:#f0f2fa;letter-spacing:-0.01em;'
    'line-height:1.1;font-family:DM Sans,sans-serif;">Cancer Risk Prediction</div>'
    '<div style="margin:3px 0 0;color:#7b84a8;font-size:0.8rem;letter-spacing:0.03em;'
    'text-transform:uppercase;font-weight:500;">AI-Powered Screening Tool &nbsp;&#183;&nbsp; 9 Machine Learning Models</div>'
    '</div></div>'

    # disclaimer
    '<div style="background:rgba(220,50,50,0.08);border:1px solid rgba(220,50,50,0.2);'
    'border-radius:8px;padding:8px 14px;font-size:0.8rem;color:#c06060;'
    'display:flex;align-items:center;gap:8px;">'
    '<span style="font-size:1rem;">&#9877;&#65039;</span>'
    '<span>This tool is for <strong>screening purposes only</strong> and does not constitute '
    'medical advice. Always consult a qualified healthcare professional.</span>'
    '</div>'
    '</div>'

    # model guide card
    '<div style="background:#0e1320;border:1px solid #1e2438;border-radius:14px;'
    'padding:18px 22px;margin-bottom:6px;">'
    '<div style="margin:0 0 12px;font-size:0.78rem;font-weight:600;color:#7b84a8;'
    'text-transform:uppercase;letter-spacing:0.05em;">Model Guide &mdash; Which Should I Use?</div>'
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'

    # card 1 — highlighted
    '<div style="background:#131825;border:1px solid #dc3232;border-radius:10px;padding:10px 14px;">'
    '<div style="font-size:0.82rem;font-weight:700;color:#f05050;margin-bottom:3px;">'
    '&#11088; HighRecall_Ensemble &mdash; Recommended</div>'
    '<div style="font-size:0.75rem;color:#9098b8;line-height:1.5;">'
    'Best for screening. Combines 4 models with boosted minority-class weights '
    'to catch the most cancer cases. Lowest false-negative rate.</div>'
    '</div>'

    # card 2
    '<div style="background:#131825;border:1px solid #1e2438;border-radius:10px;padding:10px 14px;">'
    '<div style="font-size:0.82rem;font-weight:700;color:#c0c8e0;margin-bottom:3px;">'
    '&#129352; Ensemble_L2_ET &mdash; Strong Runner-Up</div>'
    '<div style="font-size:0.75rem;color:#9098b8;line-height:1.5;">'
    'Logistic + ExtraTrees soft-vote. Better precision than the top model &mdash; '
    'fewer false alarms.</div>'
    '</div>'

    # card 3
    '<div style="background:#131825;border:1px solid #1e2438;border-radius:10px;padding:10px 14px;">'
    '<div style="font-size:0.82rem;font-weight:700;color:#c0c8e0;margin-bottom:3px;">'
    '&#128208; SVM &mdash; Highest AUC (0.931)</div>'
    '<div style="font-size:0.75rem;color:#9098b8;line-height:1.5;">'
    'Best at ranking patients by risk score. Use for risk stratification '
    'rather than binary decisions.</div>'
    '</div>'

    # card 4
    '<div style="background:#131825;border:1px solid #1e2438;border-radius:10px;padding:10px 14px;">'
    '<div style="font-size:0.82rem;font-weight:700;color:#c0c8e0;margin-bottom:3px;">'
    '&#128202; Logistic Models &mdash; Most Explainable</div>'
    '<div style="font-size:0.75rem;color:#9098b8;line-height:1.5;">'
    'L1 (feature selection) and L2 (stable coefficients). Best when '
    'transparency and interpretability matter.</div>'
    '</div>'

    '</div></div>'
)

_inputs = [
    gr.Dropdown(
        choices=model_names,
        value="HighRecall_Ensemble",
        label="Prediction Model",
        info="HighRecall_Ensemble is recommended for most users.",
    ),
    gr.Number(label="Age (years)", value=50, minimum=1, maximum=120),
    gr.Radio(["Male", "Female"], label="Biological Sex", value="Male"),
    gr.Number(label="Height (cm)", value=170, minimum=50, maximum=250),
    gr.Number(label="Weight (kg)", value=70, minimum=10, maximum=300),
    gr.Number(label="Exercise Hours / Week", value=3, minimum=0, maximum=168,
              info="Total hours of moderate-to-vigorous physical activity per week."),
    gr.Number(label="Smoking Years", value=0, minimum=0, maximum=100,
              info="Total years of active smoking (0 if never smoked)."),
    gr.Number(label="Alcohol Drinks / Week", value=0, minimum=0,
              info="Standard drinks per week (1 drink = approx 14g pure alcohol)."),
    gr.Slider(1, 10, value=5, step=1,
              label="Diet Quality Score",
              info="1 = very poor (processed, high-fat)  to  10 = excellent (whole foods, balanced)."),
    gr.Checkbox(label="Family History of Cancer", value=False,
                info="First-degree relative (parent, sibling, child) diagnosed with any cancer."),
    gr.Checkbox(label="Diagnosed with Diabetes", value=False),
    gr.Checkbox(label="Diagnosed with Hypertension", value=False),
    gr.Number(label="Sun Exposure Hours / Week", value=5, minimum=0,
              info="Average weekly hours of direct sun exposure."),
    gr.Number(label="Blood Glucose (mg/dL)", value=100, minimum=30, maximum=600,
              info="Fasting blood glucose. Normal: <100  |  Pre-diabetic: 100-125  |  Diabetic: >=126"),
    gr.Number(label="Cholesterol (mg/dL)", value=180, minimum=50, maximum=600,
              info="Total cholesterol. Desirable: <200  |  Borderline: 200-239  |  High: >=240"),
    gr.Number(label="Last Medical Checkup (months ago)", value=12, minimum=0,
              info="Months since the most recent full medical examination."),
    gr.Number(label="Number of Pregnancies", value=0, minimum=0,
              info="Enter 0 if not applicable."),
    gr.Number(label="Systolic Blood Pressure (mmHg)", value=120, minimum=50, maximum=300,
              info="Upper BP reading. Normal: <120  |  Elevated: 120-129  |  High: >=130"),
    gr.Number(label="Diastolic Blood Pressure (mmHg)", value=80, minimum=30, maximum=200,
              info="Lower BP reading. Normal: <80  |  High: >=80"),
    gr.Number(label="Resting Heart Rate (bpm)", value=70, minimum=20, maximum=250),
    gr.Number(label="Hemoglobin (g/dL)", value=14, minimum=1, maximum=25,
              info="Normal: Men 13.5-17.5  |  Women 12.0-15.5"),
    gr.Checkbox(label="Chronic Inflammation Condition", value=False,
                info="e.g. Crohn's disease, rheumatoid arthritis, chronic hepatitis, IBD."),
    gr.Checkbox(label="History of Surgery", value=False),
    gr.Number(label="Average Sleep Hours / Day", value=7, minimum=0, maximum=24,
              info="Recommended: 7-9 hours for adults."),
    gr.Slider(1, 10, value=5, step=1,
              label="Stress Level",
              info="1 = very low stress  to  10 = extremely high / chronic stress."),
]

_outputs = [
    gr.Textbox(
        label="Risk Assessment",
        info="Binary result based on the model's tuned decision threshold.",
    ),
    gr.Number(
        label="Cancer Risk Probability (%)",
        info="Model's estimated probability of cancer (0-100%). Values above the threshold = HIGH RISK.",
    ),
    gr.Number(
        label="Decision Threshold",
        info="Per-model cutoff tuned on validation data. Above this = HIGH RISK.",
    ),
    gr.Markdown(label="Risk Summary & Key Drivers"),
    gr.HTML(label="SHAP — Feature Impact Chart"),
    gr.HTML(label="LIME — Local Explanation Chart"),
]

single_interface = gr.Interface(
    fn=predict_with_model,
    inputs=_inputs,
    outputs=_outputs,
    title="",
    description=_HEADER,
    css=_CSS,
)

_BATCH_HEADER = (
    '<div style="background:#0e1320;border:1px solid #1e2438;border-radius:14px;'
    'padding:20px 24px;margin-bottom:8px;">'
    '<div style="font-size:1rem;font-weight:700;color:#f0f2fa;margin-bottom:6px;">Batch Prediction</div>'
    '<div style="font-size:0.82rem;color:#7b84a8;line-height:1.6;margin-bottom:12px;">'
    'Upload a CSV file to score multiple patients at once. '
    'The file must include the following columns:</div>'
    '<div style="background:#131825;border:1px solid #1e2438;border-radius:8px;'
    'padding:10px 14px;font-family:monospace;font-size:0.72rem;'
    'color:#9098b8;line-height:1.9;word-break:break-all;">'
    'Age, Gender, Height_cm, Weight_kg, BMI, Exercise_Hours_Week, Smoking_Years, '
    'Alcohol_Drinks_Week, Diet_Quality_Score, Family_Cancer_History, Diabetes, '
    'Hypertension, Sun_Exposure_Hours_Week, Blood_Glucose_mg_dL, Cholesterol_mg_dL, '
    'Last_Checkup_Months_Ago, Pregnancies, Systolic_BP, Diastolic_BP, '
    'Resting_Heart_Rate, Hemoglobin_g_dL, Chronic_Inflammation, Previous_Surgery, '
    'Sleep_Hours_Per_Day, Stress_Level'
    '</div>'
    '<div style="margin-top:10px;font-size:0.75rem;color:#c06060;">'
    '&#9877;&#65039; Results are for screening only — not a medical diagnosis.</div>'
    '</div>'
)

batch_interface = gr.Interface(
    fn=predict_batch,
    inputs=[
        gr.Dropdown(
            choices=model_names,
            value="HighRecall_Ensemble",
            label="Prediction Model",
            info="HighRecall_Ensemble recommended — highest cancer detection rate.",
        ),
        gr.File(label="Upload Patient CSV", file_types=[".csv"]),
    ],
    outputs=[
        gr.File(label="Download Results CSV"),
        gr.Textbox(label="Status"),
    ],
    title="",
    description=_BATCH_HEADER,
    css=_CSS,
)

demo = gr.TabbedInterface(
    [single_interface, batch_interface],
    ["Single Patient Prediction", "Batch Upload"],
    title="Cancer Risk Prediction — AI Screening Tool",
)

demo.launch(share=True)