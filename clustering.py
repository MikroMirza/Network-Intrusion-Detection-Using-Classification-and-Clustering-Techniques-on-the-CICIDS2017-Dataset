import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

from preprocessing import load_and_clean, preprocess_features

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def run_kmeans(X, n_clusters, random_state=RANDOM_STATE):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, model

def run_dbscan(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X)
    return labels, model

def run_gmm(X, n_components, random_state=RANDOM_STATE):
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(X)
    return labels, model

def evaluate_clustering(name, X, cluster_labels, true_labels, out_dir):
    lines = [f"{name}"]

    n_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()

    valid_mask = cluster_labels != -1
    unique_valid = set(cluster_labels[valid_mask])
    if len(unique_valid) > 1 and valid_mask.sum() > 1:
        sil = silhouette_score(X[valid_mask], cluster_labels[valid_mask])
        db = davies_bouldin_score(X[valid_mask], cluster_labels[valid_mask])
        lines.append(f"Silhouette Score: {sil}")
        lines.append(f"Davies-Bouldin Index: {db}")
    else:
        sil, db = None, None
        lines.append("fewer than 2 clusters, can't calculate Silhouette and Davies-Bouldin")

    ari = adjusted_rand_score(true_labels, cluster_labels)
    lines.append(f"Adjusted Rand Index: {ari}")

    text = "\n".join(lines)
    print("\n" + text)
    return {"name": name, "silhouette": sil, "davies_bouldin": db, "ari": ari,
            "n_clusters_found": n_found, "n_noise": int(n_noise)}, text


def plot_clusters_pca(X_pca, cluster_labels, title, out_path):
    plt.figure(figsize=(9, 7))
    unique_labels = sorted(set(cluster_labels))
    palette = sns.color_palette("tab20", len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = cluster_labels == lab
        label_name = "Noise" if lab == -1 else f"Cluster {lab}"
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=6, alpha=0.6,
                    color=palette[i], label=label_name)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def cluster_label_crosstab(cluster_labels, true_labels, label_encoder, name, out_dir):
    true_names = label_encoder.inverse_transform(true_labels)
    ct = pd.crosstab(pd.Series(cluster_labels, name="Cluster"),
                      pd.Series(true_names, name="True Label"))
    text = ct.to_string()
    with open(os.path.join(out_dir, f"cluster_label_crosstab_{name}.txt"), "w") as f:
        f.write(text)
    return ct


def main():
    parser = argparse.ArgumentParser(description="Clustering analysis for CICIDS2017")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--n_clusters", type=int, default=None,
                         help="Default: number of unique labels in the data")
    parser.add_argument("--pca_components", type=int, default=2)
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="clustering_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_and_clean(args.data_path, args.sample_frac)
    X, y, label_encoder, feature_names = preprocess_features(df)

    n_clusters = args.n_clusters or len(label_encoder.classes_)

    #find pc components so we can actually visualize the clusters
    #still train the algorithms on the whole feature set
    pca = PCA(n_components=max(args.pca_components, 2), random_state=RANDOM_STATE)
    X_pca_full = pca.fit_transform(X)
    X_pca_2d = X_pca_full[:, :2]
    explained = pca.explained_variance_ratio_[:2].sum()

    #truth plot
    plot_clusters_pca(X_pca_2d, y, "True Attack Labels (PCA projection)",
                       os.path.join(args.out_dir, "true_labels_pca.png"))

    all_metrics = []

    #kmeans
    km_labels, _ = run_kmeans(X, n_clusters)
    m, _ = evaluate_clustering("KMeans", X, km_labels, y, args.out_dir)
    all_metrics.append(m)
    plot_clusters_pca(X_pca_2d, km_labels, "K-Means Clusters (PCA projection)",
                       os.path.join(args.out_dir, "clusters_kmeans.png"))
    cluster_label_crosstab(km_labels, y, label_encoder, "kmeans", args.out_dir)

    #dbscan
    db_labels, _ = run_dbscan(X, args.dbscan_eps, args.dbscan_min_samples)
    m, _ = evaluate_clustering("DBSCAN", X, db_labels, y, args.out_dir)
    all_metrics.append(m)
    plot_clusters_pca(X_pca_2d, db_labels, "DBSCAN Clusters (PCA projection)",
                       os.path.join(args.out_dir, "clusters_dbscan.png"))
    cluster_label_crosstab(db_labels, y, label_encoder, "dbscan", args.out_dir)
    if m["n_clusters_found"] <= 1:
        print("DBSCAN found one cluster. Adjust --dbscan_eps and --dbscan_min_samples")

    #gmm
    gmm_labels, _ = run_gmm(X, n_clusters)
    m, _ = evaluate_clustering("GaussianMixture", X, gmm_labels, y, args.out_dir)
    all_metrics.append(m)
    plot_clusters_pca(X_pca_2d, gmm_labels, "Gaussian Mixture Clusters (PCA projection)",
                       os.path.join(args.out_dir, "clusters_gmm.png"))
    cluster_label_crosstab(gmm_labels, y, label_encoder, "gmm", args.out_dir)

    # --- summary ---
    metrics_df = pd.DataFrame(all_metrics).set_index("name")
    print("Summary")
    print(metrics_df.to_string())
    metrics_df.to_csv(os.path.join(args.out_dir, "clustering_metrics.csv"))
    with open(os.path.join(args.out_dir, "clustering_metrics.txt"), "w") as f:
        f.write(metrics_df.to_string())

    best_by_ari = metrics_df["ari"].astype(float).idxmax()
    print(f"Algorithm that clusters most closely with true attack labels: {best_by_ari}\n")

if __name__ == "__main__":
    main()