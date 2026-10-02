"""From-scratch and library models reconstructed from the coursework appendix."""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def knn_predict_one(x, X_train, y_train, k):
    distances = np.sqrt(((X_train - x) ** 2).sum(axis=1))
    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_idx]
    counts = np.bincount(nearest_labels, minlength=10)
    return int(np.argmax(counts))


def knn_predict(X_test, X_train, y_train, k=3):
    preds = np.empty(X_test.shape[0], dtype=int)
    for i, x in enumerate(X_test):
        preds[i] = knn_predict_one(x, X_train, y_train, k)
    return preds


def one_hot_numpy(y, n_classes=10):
    y = np.asarray(y).astype(int).ravel()
    Y = np.zeros((len(y), n_classes), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def add_bias_numpy(X):
    X = np.asarray(X, dtype=np.float64)
    return np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])


def train_linear_regression_multiclass_gd(
    X_train,
    y_train,
    X_eval,
    y_eval,
    n_classes=10,
    lr=0.01,
    epochs=600,
    batch_size=256,
    l2=0.0,
    grad_clip=1.0,
    lr_decay=0.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    Xb_train = add_bias_numpy(X_train)
    Xb_eval = add_bias_numpy(X_eval)
    Y_train = one_hot_numpy(y_train, n_classes)
    Y_eval = one_hot_numpy(y_eval, n_classes)

    n, d1 = Xb_train.shape
    W = rng.normal(0.0, 0.01, size=(d1, n_classes)).astype(np.float64)
    history = {"train_mse": [], "eval_mse": [], "train_acc": [], "eval_acc": []}

    for epoch in range(epochs):
        lr_epoch = lr / (1.0 + lr_decay * epoch) if lr_decay > 0 else lr
        idx = rng.permutation(n)
        Xs = Xb_train[idx]
        Ys = Y_train[idx]

        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = Ys[start:start + batch_size]

            scores = xb @ W
            diff = scores - yb
            grad = (2.0 / xb.shape[0]) * (xb.T @ diff)

            if l2 > 0.0:
                reg = 2.0 * l2 * W
                reg[0, :] = 0.0
                grad += reg

            grad = np.clip(grad, -grad_clip, grad_clip)
            W -= lr_epoch * grad

        train_scores = Xb_train @ W
        eval_scores = Xb_eval @ W

        train_mse = float(np.mean((Y_train - train_scores) ** 2))
        eval_mse = float(np.mean((Y_eval - eval_scores) ** 2))
        train_acc = float(np.mean(np.argmax(train_scores, axis=1) == y_train))
        eval_acc = float(np.mean(np.argmax(eval_scores, axis=1) == y_eval))

        history["train_mse"].append(train_mse)
        history["eval_mse"].append(eval_mse)
        history["train_acc"].append(train_acc)
        history["eval_acc"].append(eval_acc)

    return W, history


def fit_standardizer_numpy(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return mu, sigma


def transform_standardizer_numpy(X, mu, sigma):
    return (np.asarray(X, dtype=np.float64) - mu) / sigma


def poly_features_degree2_numpy(X):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    parts = [X, X ** 2]
    interactions = []
    for i in range(d):
        for j in range(i + 1, d):
            interactions.append((X[:, i] * X[:, j]).reshape(n, 1))
    if interactions:
        parts.append(np.hstack(interactions))
    return np.hstack(parts)


def predict_scores(X, W):
    return add_bias_numpy(X) @ W


def build_decision_tree():
    return DecisionTreeClassifier(random_state=RANDOM_STATE)
