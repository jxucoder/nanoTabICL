"""nanoTabICL – minimal single-file TabICLv2.

Three-stage architecture pretrained on synthetic SCM+MLP tasks:
  Stage 1  ColEmbedding   – per-column ISAB with target-aware affine output
  Stage 2  RowInteraction – CLS-aggregated transformer over features
  Stage 3  ICLearning     – transformer over rows, train-only context, MLP decoder

Run pretraining + eval:
    uv run python -m nanotabicl.nano_model
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Task = Literal["classification", "regression"]

# ── Synthetic prior ──────────────────────────────────────────────


def sample_scm_bnn_task(
    *,
    n_rows: int,
    n_features: int,
    task: Task,
    n_classes: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic task (SCM features + random MLP target)."""
    rng = np.random.default_rng(seed)
    X = np.empty((n_rows, n_features), dtype=np.float32)
    X[:, 0] = rng.normal(size=n_rows).astype(np.float32)
    for j in range(1, n_features):
        parent = int(rng.integers(0, j))
        X[:, j] = (0.7 * X[:, parent] + 0.7 * rng.normal(size=n_rows)).astype(
            np.float32
        )
    hidden = max(4, min(16, n_features * 2))
    w1 = rng.normal(0, 1 / math.sqrt(n_features), (n_features, hidden)).astype(
        np.float32
    )
    b1 = rng.normal(0, 0.2, (hidden,)).astype(np.float32)
    w2 = rng.normal(0, 1 / math.sqrt(hidden), (hidden,)).astype(np.float32)
    b2 = float(rng.normal(0, 0.2))
    raw = np.tanh(X @ w1 + b1) @ w2 + b2
    if task == "classification":
        # quantile-based bucketing for proper multi-class support
        quantiles = np.linspace(0, 1, n_classes + 1)[1:-1]
        thresholds = np.quantile(raw, quantiles)
        y = np.digitize(raw, thresholds).astype(np.int64)
    elif task == "regression":
        y = (raw + 0.05 * rng.normal(size=n_rows)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported task: {task!r}")
    return X, y


Prior = Literal["mlp", "linear", "tree"]


def _scm_features_torch(
    n_rows: int, n_features: int, device: torch.device
) -> Tensor:
    """Generate SCM-structured features (shared across priors)."""
    X = torch.empty(n_rows, n_features, device=device)
    X[:, 0] = torch.randn(n_rows, device=device)
    for j in range(1, n_features):
        p = torch.randint(0, j, (1,)).item()
        X[:, j] = 0.7 * X[:, p] + 0.7 * torch.randn(n_rows, device=device)
    return X


def _target_mlp(X: Tensor) -> Tensor:
    """Random single-hidden-layer MLP target."""
    n_features = X.shape[1]
    device = X.device
    h = max(4, min(16, n_features * 2))
    w1 = torch.randn(n_features, h, device=device) / math.sqrt(n_features)
    b1 = torch.randn(h, device=device) * 0.2
    w2 = torch.randn(h, device=device) / math.sqrt(h)
    return torch.tanh(X @ w1 + b1) @ w2


def _target_linear(X: Tensor) -> Tensor:
    """Random linear target."""
    n_features = X.shape[1]
    device = X.device
    w = torch.randn(n_features, device=device) / math.sqrt(n_features)
    b = torch.randn(1, device=device) * 0.2
    return X @ w + b


def _target_tree(X: Tensor, max_depth: int = 4) -> Tensor:
    """Random axis-aligned decision tree target."""
    n_rows, n_features = X.shape
    device = X.device
    values = torch.zeros(n_rows, device=device)
    # simulate a random decision tree by recursive splits
    indices = torch.arange(n_rows, device=device)
    _tree_split(X, values, indices, n_features, device, depth=0, max_depth=max_depth)
    return values


def _tree_split(
    X: Tensor, values: Tensor, indices: Tensor,
    n_features: int, device: torch.device,
    depth: int, max_depth: int,
) -> None:
    """Recursively assign values via random axis-aligned splits."""
    if depth >= max_depth or len(indices) < 4:
        values[indices] = torch.randn(1, device=device).item()
        return
    feat = torch.randint(0, n_features, (1,)).item()
    col = X[indices, feat]
    threshold = col.median()
    left = indices[col <= threshold]
    right = indices[col > threshold]
    if len(left) == 0 or len(right) == 0:
        values[indices] = torch.randn(1, device=device).item()
        return
    _tree_split(X, values, left, n_features, device, depth + 1, max_depth)
    _tree_split(X, values, right, n_features, device, depth + 1, max_depth)


_TARGET_FNS = {"mlp": _target_mlp, "linear": _target_linear, "tree": _target_tree}


def _sample_cls_torch(
    n_rows: int, n_features: int, n_classes: int, device: torch.device,
    prior: Prior = "mlp",
) -> tuple[Tensor, Tensor]:
    X = _scm_features_torch(n_rows, n_features, device)
    raw = _TARGET_FNS[prior](X)
    # quantile-based bucketing for proper multi-class support
    quantiles = torch.linspace(0, 1, n_classes + 1, device=device)[1:-1]
    thresholds = torch.quantile(raw, quantiles)
    y = torch.bucketize(raw, thresholds)
    return X, y


def _sample_reg_torch(
    n_rows: int, n_features: int, device: torch.device,
    prior: Prior = "mlp",
) -> tuple[Tensor, Tensor]:
    X = _scm_features_torch(n_rows, n_features, device)
    raw = _TARGET_FNS[prior](X)
    return X, raw + 0.05 * torch.randn(n_rows, device=device)


# ── Config ───────────────────────────────────────────────────────


@dataclass
class NanoTabICLConfig:
    d_model: int = 64
    nhead: int = 4
    ff_factor: int = 2
    col_num_blocks: int = 1
    col_num_inds: int = 16
    row_num_blocks: int = 2
    row_num_cls: int = 4
    icl_num_blocks: int = 3
    max_classes: int = 10
    dropout: float = 0.0


# ── Building blocks ──────────────────────────────────────────────


class _MHA(nn.Module):
    def __init__(self, d: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.nhead = nhead
        self.hd = d // nhead
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.drop = dropout
        nn.init.zeros_(self.wo.weight)
        nn.init.zeros_(self.wo.bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        def _r(x: Tensor) -> Tensor:
            return x.unflatten(-1, (self.nhead, self.hd)).transpose(-3, -2)

        o = F.scaled_dot_product_attention(
            _r(self.wq(q)),
            _r(self.wk(k)),
            _r(self.wv(v)),
            dropout_p=self.drop if self.training else 0.0,
        )
        return self.wo(o.transpose(-3, -2).flatten(-2))


class _Block(nn.Module):
    """Pre-norm transformer block (matches TabICLv2 MultiheadAttentionBlock)."""

    def __init__(self, d: int, nhead: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = _MHA(d, nhead, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        nn.init.zeros_(self.ff[2].weight)
        nn.init.zeros_(self.ff[2].bias)

    def forward(
        self,
        src: Tensor,
        *,
        ctx: Tensor | None = None,
        train_size: int | None = None,
    ) -> Tensor:
        h = self.ln(src)
        if ctx is not None:
            kv = self.ln(ctx)
        elif train_size is not None:
            kv = h[..., :train_size, :]
        else:
            kv = h
        x = src + self.attn(h, kv, kv)
        return x + self.ff(self.ln2(x))


class _ISAB(nn.Module):
    """Induced Self-Attention Block (matches TabICLv2 SetTransformer)."""

    def __init__(
        self, d: int, nhead: int, ff: int, n_ind: int, dropout: float = 0.0
    ):
        super().__init__()
        self.inds = nn.Parameter(torch.empty(n_ind, d))
        nn.init.trunc_normal_(self.inds, std=0.02)
        self.s1 = _Block(d, nhead, ff, dropout)
        self.s2 = _Block(d, nhead, ff, dropout)

    def forward(self, src: Tensor, train_size: int | None = None) -> Tensor:
        I = self.inds.expand(*src.shape[:-2], -1, -1)
        ctx = src[..., :train_size, :] if train_size is not None else src
        H = self.s1(I, ctx=ctx)
        return self.s2(src, ctx=H)


# ── Stage 1: Column Embedding ────────────────────────────────────


class _ColEmbedding(nn.Module):
    """Distribution-aware column embedding via shared ISAB + affine output."""

    def __init__(self, cfg: NanoTabICLConfig):
        super().__init__()
        d, ff = cfg.d_model, cfg.d_model * cfg.ff_factor
        self.max_classes = cfg.max_classes
        self.in_proj = nn.Linear(1, d)
        self.y_enc = (
            nn.Linear(cfg.max_classes, d) if cfg.max_classes > 0
            else nn.Linear(1, d)
        )
        self.isabs = nn.ModuleList(
            [_ISAB(d, cfg.nhead, ff, cfg.col_num_inds, cfg.dropout)
             for _ in range(cfg.col_num_blocks)]
        )
        self.w_proj = nn.Linear(d, d)
        self.w_ln = nn.LayerNorm(d)
        self.b_proj = nn.Linear(d, d)
        self.b_ln = nn.LayerNorm(d)

    def _encode_y(self, y: Tensor) -> Tensor:
        if self.max_classes > 0:
            return self.y_enc(F.one_hot(y.long(), self.max_classes).float())
        return self.y_enc(y.unsqueeze(-1).float())

    def forward(self, X: Tensor, y_train: Tensor, train_size: int) -> Tensor:
        B, T, H = X.shape
        feats = X.transpose(1, 2).unsqueeze(-1)
        src = self.in_proj(feats)
        src[:, :, :train_size] = (
            src[:, :, :train_size] + self._encode_y(y_train).unsqueeze(1)
        )
        d = src.shape[-1]
        src = src.reshape(B * H, T, d)
        for isab in self.isabs:
            src = isab(src, train_size=train_size)
        src = src.reshape(B, H, T, d)
        W = self.w_ln(self.w_proj(src))
        bias = self.b_ln(self.b_proj(src))
        return (feats * W + bias).permute(0, 2, 1, 3)


# ── Stage 2: Row Interaction ─────────────────────────────────────


class _RowInteraction(nn.Module):
    """Feature interaction within rows + CLS-token aggregation."""

    def __init__(self, cfg: NanoTabICLConfig):
        super().__init__()
        d, ff = cfg.d_model, cfg.d_model * cfg.ff_factor
        self.n_cls = cfg.row_num_cls
        self.cls = nn.Parameter(torch.empty(cfg.row_num_cls, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nb = cfg.row_num_blocks
        self.blocks = nn.ModuleList(
            [_Block(d, cfg.nhead, ff, cfg.dropout) for _ in range(max(0, nb - 1))]
        )
        self.cls_block = _Block(d, cfg.nhead, ff, cfg.dropout)
        self.out_ln = nn.LayerNorm(d)

    def forward(self, emb: Tensor) -> Tensor:
        B, T, H, d = emb.shape
        C = self.n_cls
        tokens = torch.cat([self.cls.expand(B, T, C, d), emb], dim=2)
        tokens = tokens.reshape(B * T, H + C, d)
        for blk in self.blocks:
            tokens = blk(tokens)
        cls_out = self.cls_block(tokens[:, :C], ctx=tokens)
        return self.out_ln(cls_out).reshape(B, T, C * d)


# ── Stage 3: ICL Learning ────────────────────────────────────────


class _ICLearning(nn.Module):
    """Dataset-level ICL: train-only context attention + MLP decoder."""

    def __init__(self, cfg: NanoTabICLConfig):
        super().__init__()
        icl_d = cfg.d_model * cfg.row_num_cls
        ff = icl_d * cfg.ff_factor
        self.max_classes = cfg.max_classes
        self.y_enc = (
            nn.Linear(cfg.max_classes, icl_d) if cfg.max_classes > 0
            else nn.Linear(1, icl_d)
        )
        out_dim = cfg.max_classes if cfg.max_classes > 0 else 1
        self.blocks = nn.ModuleList(
            [_Block(icl_d, cfg.nhead, ff, cfg.dropout)
             for _ in range(cfg.icl_num_blocks)]
        )
        self.ln = nn.LayerNorm(icl_d)
        self.dec = nn.Sequential(
            nn.Linear(icl_d, icl_d * 2), nn.GELU(), nn.Linear(icl_d * 2, out_dim)
        )

    def _encode_y(self, y: Tensor) -> Tensor:
        if self.max_classes > 0:
            return self.y_enc(F.one_hot(y.long(), self.max_classes).float())
        return self.y_enc(y.unsqueeze(-1).float())

    def forward(self, R: Tensor, y_train: Tensor) -> Tensor:
        ts = y_train.shape[1]
        R = R.clone()
        R[:, :ts] = R[:, :ts] + self._encode_y(y_train)
        for blk in self.blocks:
            R = blk(R, train_size=ts)
        return self.dec(self.ln(R))[:, ts:]


# ── Full model ───────────────────────────────────────────────────


class NanoTabICL(nn.Module):
    """Minimal TabICLv2: ColEmbed → RowInteraction → ICLearning."""

    def __init__(self, cfg: NanoTabICLConfig | None = None):
        super().__init__()
        self.cfg = cfg or NanoTabICLConfig()
        self.col = _ColEmbedding(self.cfg)
        self.row = _RowInteraction(self.cfg)
        self.icl = _ICLearning(self.cfg)

    def forward(self, X: Tensor, y_train: Tensor) -> Tensor:
        """
        X: (B, T, H) – train rows first, then test rows.
        y_train: (B, train_size) – training labels.
        Returns: (B, test_size, out_dim) – predictions for test rows.
        """
        ts = y_train.shape[1]
        return self.icl(self.row(self.col(X, y_train, ts)), y_train)


NanoTabICLv2 = NanoTabICL


# ── Pretraining ──────────────────────────────────────────────────


@dataclass
class PretrainConfig:
    steps: int = 500
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 4
    n_rows: tuple[int, int] = (64, 256)
    n_features: tuple[int, int] = (3, 15)
    n_classes: tuple[int, int] = (2, 10)
    train_ratio: float = 0.75
    device: str = "auto"
    seed: int = 42
    log_every: int = 100


def _resolve_device(s: str) -> torch.device:
    if s != "auto":
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_PRIORS: list[Prior] = ["mlp", "linear", "tree"]


def pretrain(model: NanoTabICL, cfg: PretrainConfig | None = None) -> list[float]:
    """Pretrain on synthetic tasks with diverse priors (MLP, linear, tree)."""
    cfg = cfg or PretrainConfig()
    dev = _resolve_device(cfg.device)
    model.to(dev).train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    torch.manual_seed(cfg.seed)
    is_cls = model.cfg.max_classes > 0
    losses: list[float] = []

    for step in range(1, cfg.steps + 1):
        nr = torch.randint(cfg.n_rows[0], cfg.n_rows[1] + 1, (1,)).item()
        nf = torch.randint(cfg.n_features[0], cfg.n_features[1] + 1, (1,)).item()
        ts = max(2, int(nr * cfg.train_ratio))
        if nr - ts < 1:
            continue
        prior = _PRIORS[torch.randint(0, len(_PRIORS), (1,)).item()]
        Xs, Ys = [], []
        for _ in range(cfg.batch_size):
            if is_cls:
                nc = min(
                    torch.randint(cfg.n_classes[0], cfg.n_classes[1] + 1, (1,)).item(),
                    model.cfg.max_classes,
                )
                x, y = _sample_cls_torch(nr, nf, nc, dev, prior=prior)
            else:
                x, y = _sample_reg_torch(nr, nf, dev, prior=prior)
            Xs.append(x)
            Ys.append(y)
        X = torch.stack(Xs)
        Y = torch.stack(Ys)

        # per-column normalisation (train rows only to avoid test leakage)
        X_train = X[:, :ts, :]
        mu = X_train.mean(dim=1, keepdim=True)
        sd = X_train.std(dim=1, keepdim=True).clamp(min=1e-6)
        X = (X - mu) / sd

        logits = model(X, Y[:, :ts])
        if is_cls:
            nc = int(Y.max().item()) + 1
            loss = F.cross_entropy(
                logits[:, :, :nc].reshape(-1, nc), Y[:, ts:].reshape(-1)
            )
        else:
            loss = F.mse_loss(logits.squeeze(-1), Y[:, ts:])

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % cfg.log_every == 0:
            avg = sum(losses[-cfg.log_every :]) / cfg.log_every
            print(f"  step {step:>5d}/{cfg.steps}  loss={avg:.4f}")

    model.eval()
    return losses


# ── Save / load ──────────────────────────────────────────────────


def save_checkpoint(model: NanoTabICL, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.cfg), "state_dict": model.state_dict()}, path
    )


def load_checkpoint(path: str, device: str = "cpu") -> NanoTabICL:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = NanoTabICL(NanoTabICLConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


# ── Inference wrappers ───────────────────────────────────────────


class NanoTabICLClassifier:
    """sklearn-like classifier backed by NanoTabICL."""

    def __init__(
        self,
        config: NanoTabICLConfig | None = None,
        checkpoint: str | None = None,
        temperature: float = 0.9,
        device: str = "auto",
    ):
        self.temperature = temperature
        self.device = _resolve_device(device)
        if checkpoint:
            self.model = load_checkpoint(checkpoint, str(self.device))
        else:
            cfg = config or NanoTabICLConfig()
            if cfg.max_classes < 1:
                cfg.max_classes = 10
            self.model = NanoTabICL(cfg)
        self.model.to(self.device).eval()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> NanoTabICLClassifier:
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.classes_, enc = torch.unique(y_t, sorted=True, return_inverse=True)
        self._n_cls = len(self.classes_)
        self._x_mu = X_t.mean(0, keepdim=True)
        self._x_sd = X_t.std(0, keepdim=True).clamp(min=1e-6)
        self._X = ((X_t - self._x_mu) / self._x_sd).to(self.device)
        self._y = enc.to(self.device)
        self._fitted = True
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit first")
        Xn = (
            (torch.as_tensor(np.asarray(X), dtype=torch.float32) - self._x_mu)
            / self._x_sd
        ).to(self.device)
        logits = self.model(
            torch.cat([self._X, Xn]).unsqueeze(0), self._y.unsqueeze(0)
        )
        logits = logits[0, :, : self._n_cls]
        return F.softmax(logits / self.temperature, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_.cpu().numpy()[np.argmax(proba, axis=1)]


class NanoTabICLRegressor:
    """sklearn-like regressor backed by NanoTabICL."""

    def __init__(
        self,
        config: NanoTabICLConfig | None = None,
        checkpoint: str | None = None,
        device: str = "auto",
    ):
        self.device = _resolve_device(device)
        if checkpoint:
            self.model = load_checkpoint(checkpoint, str(self.device))
        else:
            cfg = config or NanoTabICLConfig(max_classes=0)
            cfg.max_classes = 0
            self.model = NanoTabICL(cfg)
        self.model.to(self.device).eval()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> NanoTabICLRegressor:
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self._x_mu = X_t.mean(0, keepdim=True)
        self._x_sd = X_t.std(0, keepdim=True).clamp(min=1e-6)
        self._y_mu = y_t.mean()
        self._y_sd = y_t.std().clamp(min=1e-6)
        self._X = ((X_t - self._x_mu) / self._x_sd).to(self.device)
        self._y = ((y_t - self._y_mu) / self._y_sd).to(self.device)
        self._fitted = True
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit first")
        Xn = (
            (torch.as_tensor(np.asarray(X), dtype=torch.float32) - self._x_mu)
            / self._x_sd
        ).to(self.device)
        out = self.model(
            torch.cat([self._X, Xn]).unsqueeze(0), self._y.unsqueeze(0)
        )
        preds = out[0, :, 0] * self._y_sd + self._y_mu
        return preds.cpu().numpy()


# ── CLI: pretrain + eval ─────────────────────────────────────────

if __name__ == "__main__":
    cfg = NanoTabICLConfig()
    model = NanoTabICL(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"nanoTabICL  params={n_params:,}  d_model={cfg.d_model}")

    print("\nPretraining on synthetic classification tasks …")
    pretrain(model, PretrainConfig(steps=500, log_every=100))
    save_checkpoint(model, "nano_tabicl.pt")
    print("Saved → nano_tabicl.pt")

    print("\nEval on held-out synthetic task:")
    X, y = sample_scm_bnn_task(n_rows=640, n_features=6, task="classification", seed=7)
    clf = NanoTabICLClassifier(checkpoint="nano_tabicl.pt")
    clf.fit(X[:512], y[:512])
    acc = float((clf.predict(X[512:]) == y[512:]).mean())
    print(f"  accuracy = {acc:.3f}")
