"""Small runnable demo for the educational nano TabICLv2 implementation.

Run with:
    uv run python -m nanotabicl.demo
"""

from __future__ import annotations

import numpy as np

from nanotabicl import (
    NanoTabICLClassifier,
    NanoTabICLRegressor,
    sample_scm_bnn_task,
)


def classification_demo() -> None:
    n_train, n_test, n_features = 512, 128, 6
    x, y = sample_scm_bnn_task(
        n_rows=n_train + n_test, n_features=n_features,
        task="classification", seed=7,
    )
    clf = NanoTabICLClassifier(temperature=0.7)
    clf.fit(x[:n_train], y[:n_train])
    pred = clf.predict(x[n_train:])
    prob = clf.predict_proba(x[n_train:])[:3]

    acc = float(np.mean(pred == y[n_train:]))
    print("=== Classification demo (random weights) ===")
    print(f"Accuracy: {acc:.3f}")
    print("First 3 probability rows:")
    print(np.round(prob, 3))
    print()


def regression_demo() -> None:
    n_train, n_test, n_features = 1024, 128, 4
    x, y = sample_scm_bnn_task(
        n_rows=n_train + n_test, n_features=n_features,
        task="regression", seed=21,
    )
    reg = NanoTabICLRegressor()
    reg.fit(x[:n_train], y[:n_train])
    pred = reg.predict(x[n_train:])
    rmse = float(np.sqrt(np.mean((pred - y[n_train:]) ** 2)))

    print("=== Regression demo (random weights) ===")
    print(f"RMSE: {rmse:.3f}")
    print("First 5 predictions:", np.round(pred[:5], 3))
    print("First 5 targets:    ", np.round(y[n_train : n_train + 5], 3))
    print()


def main() -> None:
    classification_demo()
    regression_demo()


if __name__ == "__main__":
    main()
