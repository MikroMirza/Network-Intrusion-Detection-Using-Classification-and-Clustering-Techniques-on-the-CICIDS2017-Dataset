import glob
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def load_and_clean(data_path: str, sample_frac: float = 1.0,
                    random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if os.path.isfile(data_path):
        csv_files = [data_path]
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")

    frames = []
    for f in csv_files:
        d = pd.read_csv(f, low_memory=False, encoding="latin1")
        #Strip whitespaces
        d.columns = [c.strip() for c in d.columns]
        if sample_frac < 1.0:
            d = d.sample(frac=sample_frac, random_state=random_state)
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)

    #Standardize label column title
    label_col = None
    for cand in ["Label", "label", " Label"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError("Could not find the label column")
    if label_col != "Label":
        df = df.rename(columns={label_col: "Label"})

    #Replace infinity with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    #Drop duplicates and NaN
    n_before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    n_after = len(df)

    return df.reset_index(drop=True)

def preprocess_features(df: pd.DataFrame, variance_threshold: float = 0.0,
                         scale: bool = True):
    y_raw = df["Label"].values
    X_df = df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    if variance_threshold is not None:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(X_df)
        keep_cols = X_df.columns[selector.get_support()]
        dropped = set(X_df.columns) - set(keep_cols)
        X_df = X_df[keep_cols]

    feature_names = list(X_df.columns)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df.values)
    else:
        X = X_df.values

    return X, y, label_encoder, feature_names


def train_test_split_data(X, y, test_size: float = 0.2,
                           random_state: int = RANDOM_STATE):
    """Stratified train/test split, preserving class distribution (per proposal Sec. 5)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def balance_classes(X, y, method: str = "undersample",
                     random_state: int = RANDOM_STATE):
    """
    method:
      "undersample"
      "oversample"
      "none"
    """
    if method == "none":
        return X, y

    if method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == "oversample":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_res, y_res = sampler.fit_resample(X, y)
    print(f"Balanced classes via {method}: {len(y):,} -> {len(y_res):,} samples")
    return X_res, y_res