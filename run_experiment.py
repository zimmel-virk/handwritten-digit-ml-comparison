"""Run the full handwritten-digit model comparison."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import classification_metrics_numpy
from src.models import (
    build_decision_tree,
    fit_standardizer_numpy,
    knn_predict,
    poly_features_degree2_numpy,
    predict_scores,
    train_linear_regression_multiclass_gd,
    transform_standardizer_numpy,
)
from src.preprocessing import load_and_split_digits


def save_confusion_matrix(cm, title, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def record(name, y_true, y_pred, output_dir):
    metrics = classification_metrics_numpy(y_true, y_pred, 10)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        f"Confusion Matrix — {name}",
        output_dir / f"{name}_confusion_matrix.png",
    )
    serializable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    with (output_dir / f"{name}_metrics.json").open("w") as f:
        json.dump(serializable, f, indent=2)
    return {"model": name, **serializable}


def main():
    output_dir = Path("results/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_split_digits()
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("Dataset:", data["X"].shape)
    print("Any NaNs in X?", np.isnan(data["X"]).any())
    print("Any NaNs in y?", np.isnan(data["y"]).any())
    print("PCA explained variance:", data["pca"].explained_variance_ratio_.sum())

    results = []

    # 1) kNN on scaled raw features
    y_knn = knn_predict(
        data["X_test_scaled"],
        data["X_train_scaled"],
        y_train,
        k=3,
    )
    results.append(record("knn_scratch", y_test, y_knn, output_dir))

    # 2) kNN in 20-component PCA space
    y_knn_pca = knn_predict(
        data["X_test_pca"],
        data["X_train_pca"],
        y_train,
        k=3,
    )
    results.append(record("knn_pca20", y_test, y_knn_pca, output_dir))

    # 3) Multiclass linear regression from scratch on PCA features
    W_lin, _ = train_linear_regression_multiclass_gd(
        data["X_train_pca"],
        y_train,
        data["X_test_pca"],
        y_test,
        n_classes=10,
        lr=0.03,
        epochs=400,
        batch_size=256,
        l2=1e-2,
        seed=42,
    )
    y_lin = np.argmax(predict_scores(data["X_test_pca"], W_lin), axis=1)
    results.append(record("linear_regression_pca", y_test, y_lin, output_dir))

    # 4) Degree-2 polynomial regression from scratch on PCA features
    X_train_poly_raw = poly_features_degree2_numpy(data["X_train_pca"])
    X_test_poly_raw = poly_features_degree2_numpy(data["X_test_pca"])
    mu, sigma = fit_standardizer_numpy(X_train_poly_raw)
    X_train_poly = transform_standardizer_numpy(X_train_poly_raw, mu, sigma)
    X_test_poly = transform_standardizer_numpy(X_test_poly_raw, mu, sigma)

    W_poly, _ = train_linear_regression_multiclass_gd(
        X_train_poly,
        y_train,
        X_test_poly,
        y_test,
        n_classes=10,
        lr=0.01,
        epochs=600,
        batch_size=256,
        l2=5e-2,
        seed=42,
    )
    y_poly = np.argmax(predict_scores(X_test_poly, W_poly), axis=1)
    results.append(record("polynomial_regression_pca", y_test, y_poly, output_dir))

    # 5) Decision Tree library baseline
    tree = build_decision_tree()
    tree.fit(data["X_train"], y_train)
    y_tree = tree.predict(data["X_test"])
    results.append(record("decision_tree", y_test, y_tree, output_dir))

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "model_comparison.csv", index=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["model"], df["accuracy"])
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("Handwritten Digit Model Comparison")
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, df["accuracy"]):
        plt.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.2%}", ha="center")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()

    print(df[["model", "accuracy", "macro_f1", "weighted_f1"]].to_string(index=False))
    print("\nGenerated results saved to results/generated/")


if __name__ == "__main__":
    main()
