"""Modal GPU training for nanoTabICL classification.

Usage:
    modal run experiments/classification/train_modal.py
    modal run --detach experiments/classification/train_modal.py
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
    timeout=14400,
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
            steps=200_000,
            batch_size=8,
            lr=1e-4,
            weight_decay=1e-2,
            n_rows=(64, 1024),
            n_features=(3, 30),
            n_classes=(2, 10),
            device="cuda",
            log_every=2000,
            warmup_steps=2000,
            use_amp=True,
        ),
    )

    save_checkpoint(model, CHECKPOINT_PATH)
    volume.commit()
    print(f"\nSaved checkpoint → {CHECKPOINT_PATH}")

    avg_loss = sum(losses[-1000:]) / len(losses[-1000:])
    print(f"Final avg loss (last 1000 steps): {avg_loss:.4f}")


@app.local_entrypoint()
def main():
    train.remote()
    print("Training done.")
