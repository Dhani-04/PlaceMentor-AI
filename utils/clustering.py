"""K-Means workflow for PlaceMentor AI.

The scaler and the K-Means model live in ONE scikit-learn Pipeline, so the
exact same standardisation is applied during training and prediction.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import (
    FEATURES,
    build_input_frame,
    generate_simulated_dataset,
    load_and_validate,
    save_dataset,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "kmeans_pipeline.joblib"
OUTPUT_PATH = BASE_DIR / "outputs" / "clustered_students.csv"

RANDOM_STATE = 42
DEFAULT_K = 3

CAREER_READY = "Career-ready profile"
DEVELOPING = "Developing talent"
PRIORITY = "Skill-building priority"


# --------------------------------------------------------------------------
# Naming clusters (descriptive labels assigned AFTER clustering)
# --------------------------------------------------------------------------
def name_clusters(centroids_scaled: np.ndarray) -> tuple[dict[int, str], dict[int, float]]:
    """Give each cluster a descriptive name from its actual centroid.

    The composite score is the mean standardised value of all features at the
    centroid. Highest -> Career-ready, lowest -> Skill-building priority,
    everything in between -> Developing talent. K-Means itself learns no labels.
    """
    scores = centroids_scaled.mean(axis=1)
    order = list(np.argsort(-scores))
    labels: dict[int, str] = {}
    for rank, cid in enumerate(order):
        if rank == 0:
            labels[int(cid)] = CAREER_READY
        elif rank == len(order) - 1:
            labels[int(cid)] = PRIORITY
        else:
            labels[int(cid)] = DEVELOPING
    return labels, {int(c): float(scores[c]) for c in range(len(scores))}


# --------------------------------------------------------------------------
# Model evaluation (Elbow + Silhouette)
# --------------------------------------------------------------------------
def evaluate_k_range(df: pd.DataFrame, k_min: int = 2, k_max: int = 8,
                     seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Inertia (WCSS) and silhouette score for each k in [k_min, k_max]."""
    X = StandardScaler().fit_transform(df[FEATURES].astype(float))
    rows = []
    for k in range(k_min, min(k_max, len(df) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, km.labels_)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def dataframe_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def fit_bundle(df: pd.DataFrame, k: int = DEFAULT_K, seed: int = RANDOM_STATE) -> dict:
    """Fit StandardScaler + K-Means and collect everything the app needs."""
    X = df[FEATURES].astype(float)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])
    pipeline.fit(X)

    scaler = pipeline.named_steps["scaler"]
    kmeans = pipeline.named_steps["kmeans"]
    X_scaled = scaler.transform(X)
    labels = kmeans.labels_

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    cluster_labels, composite = name_clusters(centroids_scaled)

    own_dist = np.linalg.norm(X_scaled - centroids_scaled[labels], axis=1)
    typical = {int(c): float(own_dist[labels == c].mean()) for c in range(k)}

    return {
        "pipeline": pipeline,
        "features": list(FEATURES),
        "k": int(k),
        "cluster_labels": cluster_labels,
        "composite_scores": composite,
        "typical_distance": typical,
        "centroids_scaled": centroids_scaled,
        "centroids_original": centroids_original,
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "inertia": float(kmeans.inertia_),
        "n_samples": int(len(df)),
        "data_hash": dataframe_hash(df[FEATURES]),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


def save_bundle(bundle: dict, path: Path = MODEL_PATH) -> bool:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
        return True
    except OSError:
        return False


def load_bundle(path: Path = MODEL_PATH) -> dict | None:
    """Load the saved model; return None if it is missing or unusable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    needed = {"pipeline", "features", "k", "cluster_labels", "sklearn_version", "data_hash"}
    if not isinstance(bundle, dict) or not needed.issubset(bundle):
        return None
    if bundle["sklearn_version"] != sklearn.__version__:
        return None  # trained with another scikit-learn version: retrain
    if bundle["features"] != FEATURES:
        return None
    return bundle


def save_clustered_output(df: pd.DataFrame, bundle: dict, path: Path = OUTPUT_PATH) -> bool:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        add_clusters(df, bundle).to_csv(path, index=False)
        return True
    except OSError:
        return False


def ensure_project_assets(k: int = DEFAULT_K, force_retrain: bool = False):
    """Make sure dataset, model and output file exist and agree with each other.

    Returns (dataframe, bundle, notes). Missing pieces are rebuilt automatically,
    which is what makes the app work on a fresh Streamlit Cloud deployment.
    """
    notes: list[str] = []

    if not DATASET_PATH.exists():
        save_dataset(generate_simulated_dataset(), DATASET_PATH)
        notes.append("Simulated dataset was missing and has been regenerated.")

    df, _report = load_and_validate(DATASET_PATH)

    bundle = None if force_retrain else load_bundle()
    if bundle is not None and bundle["data_hash"] != dataframe_hash(df[FEATURES]):
        bundle = None
    if bundle is None:
        bundle = fit_bundle(df, k=k)
        save_bundle(bundle)
        save_clustered_output(df, bundle)
        notes.append("K-Means model was trained automatically.")
    elif not OUTPUT_PATH.exists():
        save_clustered_output(df, bundle)

    return df, bundle, notes


# --------------------------------------------------------------------------
# Using the trained model
# --------------------------------------------------------------------------
def add_clusters(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return a copy of df with Cluster, Profile Group and Centroid Distance."""
    pipe = bundle["pipeline"]
    X = df[FEATURES].astype(float)
    scaled = pipe.named_steps["scaler"].transform(X)
    kmeans = pipe.named_steps["kmeans"]
    labels = kmeans.predict(scaled)
    dists = np.linalg.norm(scaled - kmeans.cluster_centers_[labels], axis=1)

    out = df.copy()
    out["Cluster"] = labels.astype(int)
    out["Profile Group"] = [bundle["cluster_labels"][int(c)] for c in labels]
    out["Centroid Distance"] = np.round(dists, 3)
    return out


def predict_profile(bundle: dict, values: dict) -> dict:
    """Assign one student's input to the nearest cluster centroid."""
    pipe = bundle["pipeline"]
    frame = build_input_frame(values)
    scaled = pipe.named_steps["scaler"].transform(frame)  # transform only, never fit
    kmeans = pipe.named_steps["kmeans"]
    distances = kmeans.transform(scaled)[0]
    cid = int(np.argmin(distances))
    typical = bundle["typical_distance"][cid]
    dist = float(distances[cid])
    ratio = dist / typical if typical else 1.0
    if ratio <= 1.0:
        closeness = "Close to the centre of this group"
    elif ratio <= 1.5:
        closeness = "Moderately away from the centre of this group"
    else:
        closeness = "Far from the centre of this group (a less typical profile)"
    return {
        "cluster": cid,
        "profile": bundle["cluster_labels"][cid],
        "distance": dist,
        "typical_distance": typical,
        "closeness": closeness,
        "all_distances": {int(i): float(d) for i, d in enumerate(distances)},
        "scaled": scaled[0],
    }


def fit_pca(df: pd.DataFrame, bundle: dict):
    """PCA on the SCALED data, for two-dimensional visualisation only."""
    scaler = bundle["pipeline"].named_steps["scaler"]
    scaled = scaler.transform(df[FEATURES].astype(float))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)
    return pca, coords


def cluster_summary(clustered: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Per-cluster size and feature averages, strongest group first."""
    grouped = clustered.groupby("Cluster")[FEATURES].mean().round(2)
    grouped.insert(0, "Students", clustered.groupby("Cluster").size())
    grouped.insert(1, "Share (%)", (grouped["Students"] / len(clustered) * 100).round(1))
    grouped.insert(0, "Profile Group", [bundle["cluster_labels"][int(c)] for c in grouped.index])
    grouped["Composite"] = [round(bundle["composite_scores"][int(c)], 2) for c in grouped.index]
    return grouped.sort_values("Composite", ascending=False)
