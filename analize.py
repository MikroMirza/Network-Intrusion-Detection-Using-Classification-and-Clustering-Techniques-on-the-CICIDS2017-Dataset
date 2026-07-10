import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


def load_data(data_path: str, sample_frac: float, random_state: int = 42) -> pd.DataFrame:
    if os.path.isfile(data_path):
        csv_files = [data_path]
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, encoding="latin1")
        df.columns = [c.strip() for c in df.columns]
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)
        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)
    print("Data loaded")
    print(f"Dataframe shape: {full_df.shape}\n")
    return full_df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    label_col = None
    for cand in ["Label", "label", " Label"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col != "Label":
        df = df.rename(columns={label_col: "Label"})

    #Replace infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df


def report_summary_stats(df: pd.DataFrame, out_dir: str) -> None:
    lines = []
    lines.append("Summary")
    lines.append(f"Shape: {df.shape}")

    lines.append("Dtypes:")
    lines.append(df.dtypes.value_counts().to_string())

    n_dupes = df.duplicated().sum()
    lines.append(f"Duplicate rows: {n_dupes}\n")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    lines.append("Columns with missing values:")
    if missing.empty:
        lines.append("  None")
    else:
        for col, cnt in missing.items():
            lines.append(f"  {col}: {cnt} ({cnt / len(df) * 100}%)")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "summary_stats.txt"), "w") as f:
        f.write(text)


def plot_class_distribution(df: pd.DataFrame, out_dir: str) -> None:
    counts = df["Label"].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, orient="h")
    plt.xscale("log")
    plt.xlabel("Count (log scale)")
    plt.ylabel("Attack Category")
    plt.title("Class Distribution (Normal vs Attack Types)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    plt.close()


def plot_missing_values(df: pd.DataFrame, out_dir: str) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("\nNo missing values to plot.")
        return
    plt.figure(figsize=(10, max(4, len(missing) * 0.3)))
    sns.barplot(x=missing.values, y=missing.index, orient="h")
    plt.xlabel("Missing value count")
    plt.title("Missing Values by Column")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "missing_values.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: str, top_n: int = 25) -> pd.Index:
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    variances = numeric_df.var().sort_values(ascending=False)
    top_features = variances.head(top_n).index
    corr = numeric_df[top_features].fillna(0).corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                xticklabels=True, yticklabels=True, annot=True)
    plt.title(f"Correlation Heatmap (Top {top_n} Highest-Variance Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    return top_features


def analyze_outliers(df: pd.DataFrame, out_dir: str) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    lines = ["Outlier summary"]
    outlier_counts = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((series < lower) | (series > upper)).sum()
        pct = n_outliers / len(series) * 100
        outlier_counts[col] = pct
        lines.append(f"{col}: {n_outliers:,} outliers ({pct:.2f}%)")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "outlier_summary.txt"), "w") as f:
        f.write(text)
    print("Outlier summary written")

    # boxplots for the features with highest outlier rates (most informative)
    top_outlier_cols = sorted(outlier_counts, key=outlier_counts.get, reverse=True)[:6]
    if top_outlier_cols:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, col in zip(axes.flatten(), top_outlier_cols):
            sns.boxplot(x=df[col].dropna(), ax=ax)
            ax.set_title(col, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "boxplots_top_features.png"), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample_frac", type=float, default=1.0,
                         help="Fraction of rows to sample")
    parser.add_argument("--out_dir", type=str, default="eda_outputs",
                         help="Destination folder")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.data_path, args.sample_frac)
    df = basic_cleaning(df)

    report_summary_stats(df, args.out_dir)
    plot_class_distribution(df, args.out_dir)
    plot_missing_values(df, args.out_dir)
    top_features = plot_correlation_heatmap(df, args.out_dir)
    analyze_outliers(df, args.out_dir)

    print(f"Data saved to: {os.path.abspath(args.out_dir)}\n")


if __name__ == "__main__":
    main()