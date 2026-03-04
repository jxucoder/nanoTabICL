"""Modal GPU training & eval for nanoTabICL.

Usage:
    modal run train_modal.py
"""

import modal

app = modal.App("nanotabicl")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scikit-learn")
    .add_local_dir("src", remote_path="/root/src")
)

volume = modal.Volume.from_name("nanotabicl-checkpoints", create_if_missing=True)
CHECKPOINT_PATH = "/checkpoints/nano_tabicl.pt"


@app.function(
    gpu="A10G",
    timeout=7200,
    image=image,
    volumes={"/checkpoints": volume},
)
def train():
    import sys

    sys.path.insert(0, "/root/src")

    from nanotabicl import (
        NanoTabICL,
        NanoTabICLConfig,
        PretrainConfig,
        pretrain,
        save_checkpoint,
    )

    cfg = NanoTabICLConfig(
        d_model=256,
        nhead=8,
        ff_factor=2,
        col_num_blocks=2,
        col_num_inds=32,
        row_num_blocks=4,
        row_num_cls=4,
        icl_num_blocks=8,
        max_classes=10,
        dropout=0.1,
    )
    model = NanoTabICL(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"nanoTabICL  params={n_params:,}  d_model={cfg.d_model}")

    print("\nPretraining on synthetic tasks …")
    losses = pretrain(
        model,
        PretrainConfig(
            steps=50_000,
            batch_size=16,
            lr=1e-4,
            weight_decay=1e-2,
            n_rows=(64, 512),
            n_features=(3, 30),
            n_classes=(2, 10),
            device="cuda",
            log_every=1000,
            warmup_steps=500,
            use_amp=True,
        ),
    )

    save_checkpoint(model, CHECKPOINT_PATH)
    volume.commit()
    print(f"\nSaved checkpoint → {CHECKPOINT_PATH}")

    avg_loss = sum(losses[-1000:]) / len(losses[-1000:])
    print(f"Final avg loss (last 1000 steps): {avg_loss:.4f}")


@app.function(
    gpu="A10G",
    timeout=1200,
    image=image,
    volumes={"/checkpoints": volume},
)
def evaluate():
    import sys

    sys.path.insert(0, "/root/src")

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR

    from nanotabicl import (
        NanoTabICLClassifier,
        NanoTabICLRegressor,
        sample_scm_bnn_task,
    )

    volume.reload()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    n_train, n_test = 512, 128
    method_names = ["nanoTabICL", "KNN-5", "KNN-10", "SVM", "RF", "LogReg/Ridge"]

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  CLASSIFICATION")
    print("=" * 72)
    cls_features = [4, 6, 8, 10, 12]
    cls_classes = [2, 3, 4, 5, 6]
    all_cls: dict[str, list[float]] = {m: [] for m in method_names}

    for i, (nf, nc) in enumerate(zip(cls_features, cls_classes)):
        seed = 100 + i
        X, y = sample_scm_bnn_task(
            n_rows=n_train + n_test,
            n_features=nf,
            task="classification",
            n_classes=nc,
            seed=seed,
        )
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]

        scaler = StandardScaler().fit(X_tr)
        Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

        row: dict[str, float] = {}

        icl = NanoTabICLClassifier(checkpoint=CHECKPOINT_PATH, device="cuda")
        icl.fit(X_tr, y_tr)
        row["nanoTabICL"] = float(np.mean(icl.predict(X_te) == y_te))

        for k in (5, 10):
            knn = KNeighborsClassifier(n_neighbors=k).fit(Xs_tr, y_tr)
            row[f"KNN-{k}"] = float(np.mean(knn.predict(Xs_te) == y_te))

        svm = SVC(kernel="rbf").fit(Xs_tr, y_tr)
        row["SVM"] = float(np.mean(svm.predict(Xs_te) == y_te))

        rf = RandomForestClassifier(n_estimators=100, random_state=seed).fit(X_tr, y_tr)
        row["RF"] = float(np.mean(rf.predict(X_te) == y_te))

        lr = LogisticRegression(max_iter=1000).fit(Xs_tr, y_tr)
        row["LogReg/Ridge"] = float(np.mean(lr.predict(Xs_te) == y_te))

        for m in method_names:
            all_cls[m].append(row[m])

        parts = "  ".join(f"{m}={row[m]:.3f}" for m in method_names)
        print(f"  Task {i+1} (feat={nf}, cls={nc}):  {parts}")

    print("-" * 72)
    avg_parts = "  ".join(
        f"{m}={np.mean(all_cls[m]):.3f}" for m in method_names
    )
    print(f"  Average:  {avg_parts}")

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  REGRESSION")
    print("=" * 72)
    reg_features = [4, 6, 8, 10, 12]
    all_reg: dict[str, list[float]] = {m: [] for m in method_names}

    for i, nf in enumerate(reg_features):
        seed = 200 + i
        X, y = sample_scm_bnn_task(
            n_rows=n_train + n_test,
            n_features=nf,
            task="regression",
            seed=seed,
        )
        X_tr, X_te = X[:n_train], X[n_train:]
        y_tr, y_te = y[:n_train], y[n_train:]

        scaler = StandardScaler().fit(X_tr)
        Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)

        def rmse(preds):
            return float(np.sqrt(np.mean((preds - y_te) ** 2)))

        row: dict[str, float] = {}

        # NOTE: Cannot reuse the classification checkpoint here because the
        # model architecture differs (max_classes=10 vs 0).  Use a fresh
        # (untrained) regressor so the benchmark still runs end-to-end.
        reg = NanoTabICLRegressor(device="cuda")
        reg.fit(X_tr, y_tr)
        row["nanoTabICL"] = rmse(reg.predict(X_te))

        for k in (5, 10):
            knn = KNeighborsRegressor(n_neighbors=k).fit(Xs_tr, y_tr)
            row[f"KNN-{k}"] = rmse(knn.predict(Xs_te))

        svm = SVR(kernel="rbf").fit(Xs_tr, y_tr)
        row["SVM"] = rmse(svm.predict(Xs_te))

        rf = RandomForestRegressor(n_estimators=100, random_state=seed).fit(X_tr, y_tr)
        row["RF"] = rmse(rf.predict(X_te))

        ridge = Ridge().fit(Xs_tr, y_tr)
        row["LogReg/Ridge"] = rmse(ridge.predict(Xs_te))

        for m in method_names:
            all_reg[m].append(row[m])

        parts = "  ".join(f"{m}={row[m]:.4f}" for m in method_names)
        print(f"  Task {i+1} (feat={nf}):  {parts}")

    print("-" * 72)
    avg_parts = "  ".join(
        f"{m}={np.mean(all_reg[m]):.4f}" for m in method_names
    )
    print(f"  Average:  {avg_parts}")


@app.local_entrypoint()
def main():
    train.remote()
    evaluate.remote()
    print("Done.")
