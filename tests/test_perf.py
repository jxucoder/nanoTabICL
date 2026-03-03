"""Performance benchmark for NanoTabICL on synthetic SCM+BNN tasks.

Run with:
    uv run python tests/test_perf.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from nanotabicl import NanoTabICLClassifier, NanoTabICLRegressor, sample_scm_bnn_task


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class MajorityClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray) -> MajorityClassifier:
        vals, counts = np.unique(y, return_counts=True)
        self._majority = vals[np.argmax(counts)]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._majority)


class MeanRegressor:
    def fit(self, X: np.ndarray, y: np.ndarray) -> MeanRegressor:
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._mean)


class KNNClassifier:
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        self._X = X
        self._y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None, :] - self._X[None, :, :], axis=2)
        idx = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        neighbor_labels = self._y[idx]
        return np.array(
            [np.bincount(row.astype(int)).argmax() for row in neighbor_labels]
        )


class KNNRegressor:
    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNRegressor:
        self._X = X
        self._y = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None, :] - self._X[None, :, :], axis=2)
        idx = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        return np.mean(self._y[idx], axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    metric: float
    elapsed_ms: float


def bench_classification(
    n_train: int, n_test: int, n_features: int, seed: int,
) -> list[BenchResult]:
    X, y = sample_scm_bnn_task(
        n_rows=n_train + n_test, n_features=n_features,
        task="classification", seed=seed,
    )
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    models: list[tuple[str, object]] = [
        ("NanoTabICL", NanoTabICLClassifier()),
        ("Majority", MajorityClassifier()),
        ("5-NN", KNNClassifier(k=5)),
    ]

    results: list[BenchResult] = []
    for name, m in models:
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)  # type: ignore[attr-defined]
        pred = m.predict(X_te)  # type: ignore[attr-defined]
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(BenchResult(name, accuracy(y_te, pred), elapsed))
    return results


def bench_regression(
    n_train: int, n_test: int, n_features: int, seed: int,
) -> list[BenchResult]:
    X, y = sample_scm_bnn_task(
        n_rows=n_train + n_test, n_features=n_features,
        task="regression", seed=seed,
    )
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    models: list[tuple[str, object]] = [
        ("NanoTabICL", NanoTabICLRegressor()),
        ("Mean", MeanRegressor()),
        ("5-NN", KNNRegressor(k=5)),
    ]

    results: list[BenchResult] = []
    for name, m in models:
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)  # type: ignore[attr-defined]
        pred = m.predict(X_te)  # type: ignore[attr-defined]
        elapsed = (time.perf_counter() - t0) * 1000
        r2 = r_squared(y_te, pred)
        results.append(BenchResult(name, r2, elapsed))
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_table(metric_name: str, rows: list[BenchResult]) -> None:
    name_w = max(len(r.name) for r in rows)
    print(f"  {'Model':<{name_w}}  {metric_name:>10}  {'Time (ms)':>10}")
    print(f"  {'-' * name_w}  {'-' * 10}  {'-' * 10}")
    for r in rows:
        print(f"  {r.name:<{name_w}}  {r.metric:>10.4f}  {r.elapsed_ms:>10.1f}")
    print()


def main() -> None:
    print("=" * 64)
    print("  NanoTabICL Performance Benchmark (random weights)")
    print("=" * 64)
    print()

    seeds = [0, 7, 42]
    n_features_list = [4, 6]

    print("-" * 64)
    print("  Classification accuracy")
    print("-" * 64)
    agg: dict[str, list[float]] = {}
    for nf in n_features_list:
        for seed in seeds:
            results = bench_classification(256, 64, nf, seed)
            print(f"\n  seed={seed}, n_features={nf}")
            print_table("Accuracy", results)
            for r in results:
                agg.setdefault(r.name, []).append(r.metric)

    print("-" * 64)
    print("  Summary (mean ± std)")
    print("-" * 64)
    for name, vals in agg.items():
        arr = np.array(vals)
        print(f"  {name:<12}  {arr.mean():.4f} ± {arr.std():.4f}")
    print()

    print("-" * 64)
    print("  Regression R²")
    print("-" * 64)
    agg_reg: dict[str, list[float]] = {}
    for nf in n_features_list:
        for seed in seeds:
            results = bench_regression(256, 64, nf, seed)
            print(f"\n  seed={seed}, n_features={nf}")
            print_table("R²", results)
            for r in results:
                agg_reg.setdefault(r.name, []).append(r.metric)

    print("-" * 64)
    print("  Summary (mean ± std)")
    print("-" * 64)
    for name, vals in agg_reg.items():
        arr = np.array(vals)
        print(f"  {name:<12}  {arr.mean():.4f} ± {arr.std():.4f}")
    print()


if __name__ == "__main__":
    main()
