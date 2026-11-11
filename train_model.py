"""Train the PlaceMentor AI K-Means model.

Usage (from the project folder):
    python train_model.py                 # train with k = 3
    python train_model.py --regenerate    # also rebuild the simulated dataset
    python train_model.py --k 4           # try another cluster count
"""
from __future__ import annotations

import argparse

from utils.clustering import (
    DATASET_PATH,
    DEFAULT_K,
    MODEL_PATH,
    OUTPUT_PATH,
    add_clusters,
    cluster_summary,
    evaluate_k_range,
    fit_bundle,
    save_bundle,
    save_clustered_output,
)
from utils.preprocessing import generate_simulated_dataset, load_and_validate, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PlaceMentor AI K-Means model.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="number of clusters (default 3)")
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerate the simulated dataset before training")
    args = parser.parse_args()

    # STEP 1: load (or create) the dataset
    if args.regenerate or not DATASET_PATH.exists():
        save_dataset(generate_simulated_dataset(), DATASET_PATH)
        print(f"Simulated dataset written to {DATASET_PATH}")

    # STEP 2 and 3: validate, handle missing values, select features
    df, report = load_and_validate(DATASET_PATH)
    print(f"Loaded {report['rows']} records | missing filled: {report['missing_values_filled']}"
          f" | values clipped: {report['values_clipped']}")

    # STEP 4 to 6: scaling happens inside evaluate_k_range; inertia + silhouette for k = 2..8
    evaluation = evaluate_k_range(df)
    print("\nElbow and silhouette analysis (k = 2 to 8)")
    print(evaluation.round(3).to_string(index=False))

    # STEP 7 to 9: final model, labels, descriptive names
    bundle = fit_bundle(df, k=args.k)
    print(f"\nFinal model: k = {bundle['k']} | silhouette = {bundle['silhouette']:.3f}"
          f" | inertia = {bundle['inertia']:.1f}")

    clustered = add_clusters(df, bundle)
    print("\nCluster summary")
    print(cluster_summary(clustered, bundle).to_string())

    # STEP 11: persist model + scaler (one pipeline) and the clustered dataset
    if save_bundle(bundle):
        print(f"\nModel saved to {MODEL_PATH}")
    if save_clustered_output(df, bundle):
        print(f"Clustered dataset saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
