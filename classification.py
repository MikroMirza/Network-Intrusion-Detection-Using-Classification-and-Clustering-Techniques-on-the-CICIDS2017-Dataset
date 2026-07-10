import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

from preprocessing import load_and_clean, preprocess_features, train_test_split_data, balance_classes

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier

def build_models(random_state: int = 42):
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=random_state),
            {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=random_state, n_jobs=-1, eval_metric="mlogloss"),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.3],
            }
        )
    }

    return models

def tune_model(name, estimator, param_grid, X_train, y_train,
                search: str, n_iter: int, cv: int, random_state: int = 42):
    if search == "grid":
        searcher = GridSearchCV(estimator, param_grid, cv=cv,
                                 scoring="f1_weighted", n_jobs=-1, verbose=1)
    else:
        searcher = RandomizedSearchCV(estimator, param_grid, n_iter=n_iter, cv=cv,
                                       scoring="f1_weighted", n_jobs=-1,
                                       random_state=random_state, verbose=1)
    searcher.fit(X_train, y_train)
    print(f"Best params for {name}: {searcher.best_params_}")
    print(f"Best CV F1-weighted score: {searcher.best_score_}")
    return searcher.best_estimator_, searcher.best_params_

def evaluate_model(name, model, X_test, y_test, label_encoder, out_dir):
    y_pred = model.predict(X_test)
    n_classes = len(label_encoder.classes_)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    roc_auc = None
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    metrics["ROC-AUC"] = roc_auc

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(max(6, n_classes), max(5, n_classes * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"), dpi=150)
    plt.close()

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

    return metrics, report, y_proba


def plot_roc_curves(results, y_test, label_encoder, out_dir):
    """1 vs All curves"""
    n_classes = len(label_encoder.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes)) if n_classes > 2 else None

    plt.figure(figsize=(8, 7))
    for name, (metrics, report, y_proba) in results.items():
        if y_proba is None:
            continue
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_val})")
        else:
            all_fpr = np.unique(np.concatenate([
                roc_curve(y_test_bin[:, i], y_proba[:, i])[0] for i in range(n_classes)
            ]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
            mean_tpr /= n_classes
            roc_auc_val = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, label=f"{name} (macro AUC={roc_auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (macro-average for multiclass)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name, out_dir, top_n=20):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)

    if importances is None:
        print(f"{model_name} does not show feature importances")
        return

    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, max(4, top_n * 0.3)))
    sns.barplot(x=imp_series.values, y=imp_series.index, orient="h")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()
    return imp_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--balance", type=str, default="none",
                         choices=["none", "undersample", "oversample"])
    parser.add_argument("--search", type=str, default="random", choices=["grid", "random"])
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="classification_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_and_clean(args.data_path, args.sample_frac)
    X, y, label_encoder, feature_names = preprocess_features(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    if args.balance != "none":
        X_train, y_train = balance_classes(X_train, y_train, method=args.balance)

    models = build_models()
    results = {}
    fitted_models = {}
    best_params_all = {}

    for name, (estimator, param_grid) in models.items():
        best_model, best_params = tune_model(
            name, estimator, param_grid, X_train, y_train,
            search=args.search, n_iter=args.n_iter, cv=args.cv,
        )
        metrics, report, y_proba = evaluate_model(name, best_model, X_test, y_test, label_encoder, args.out_dir)
        results[name] = (metrics, report, y_proba)
        fitted_models[name] = best_model
        best_params_all[name] = best_params

    comparison_df = pd.DataFrame({name: metrics for name, (metrics, _, _) in results.items()}).T
    print(comparison_df.to_string())
    comparison_df.to_csv(os.path.join(args.out_dir, "model_comparison.csv"))
    with open(os.path.join(args.out_dir, "model_comparison.txt"), "w") as f:
        f.write(comparison_df.to_string())

    plot_roc_curves(results, y_test, label_encoder, args.out_dir)

    best_name = comparison_df["F1-score"].astype(float).idxmax()
    best_model = fitted_models[best_name]
    print(f"\nBest model by weighted F1-score: {best_name}")

    imp_series = plot_feature_importance(best_model, feature_names, best_name, args.out_dir)

    with open(os.path.join(args.out_dir, "best_model_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Best hyperparameters: {best_params_all[best_name]}\n\n")
        f.write("Classification report:\n")
        f.write(results[best_name][1])
        if imp_series is not None:
            f.write("\n\nTop features:\n")
            f.write(imp_series.to_string())


if __name__ == "__main__":
    main()