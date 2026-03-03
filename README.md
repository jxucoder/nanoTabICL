# nanoTabICL

Minimal, single-file implementation of the **TabICLv2** architecture for
tabular in-context learning – pretraining and inference in one place.

## What is implemented

The nano model strictly follows the TabICLv2 three-stage pipeline:

1. **Column embedding** – per-column Induced Self-Attention Block (ISAB /
   SetTransformer-style) with target-aware affine output.
2. **Row interaction** – multi-block transformer over features within each
   row, with learnable CLS tokens aggregated via cross-attention.
3. **ICL prediction** – transformer over rows where all queries attend only
   to training rows, plus an MLP decoder.

Additional components:

- **Diverse synthetic priors** (MLP, linear, decision tree) for generating pretraining tasks.
- **Pretraining loop** (AdamW + cosine LR schedule on synthetic classification/regression tasks).
- **Save / load** pretrained checkpoints.
- **sklearn-like wrappers** (`NanoTabICLClassifier`, `NanoTabICLRegressor`).

This is a *nano* educational reimplementation – no ensembling, no Flash
Attention, no KV cache, no mixed-radix multi-class logic.

## Quickstart

```bash
uv sync
uv run python -m nanotabicl.demo          # quick demo (random weights)
uv run python -m nanotabicl.nano_model     # pretrain + eval
```

Run smoke tests:

```bash
uv run python -m unittest discover -s tests
```

## Website demo (GitHub Pages ready)

A beginner-friendly TabICLv2 explainer + interactive toy demo lives in `docs/`:

- `docs/index.html` – walkthrough for someone with basic ML background.
- `docs/app.js` – synthetic tabular task + context-based toy predictor.
- `docs/styles.css` – page styling.

Preview locally:

```bash
python -m http.server 8000 --directory docs
# open http://localhost:8000
```

Deploy later on GitHub Pages:

1. Push this repo with `docs/`.
2. Go to `Settings -> Pages`.
3. Select `Deploy from a branch`, choose `main` and `/docs`.

## Pretraining

```python
from nanotabicl import NanoTabICL, NanoTabICLConfig, PretrainConfig, pretrain, save_checkpoint

model = NanoTabICL(NanoTabICLConfig(d_model=64))
pretrain(model, PretrainConfig(steps=2000))
save_checkpoint(model, "nano_tabicl.pt")
```

## Inference

```python
from nanotabicl import NanoTabICLClassifier, sample_scm_bnn_task

X, y = sample_scm_bnn_task(n_rows=640, n_features=6, task="classification", seed=0)

clf = NanoTabICLClassifier(checkpoint="nano_tabicl.pt")
clf.fit(X[:512], y[:512])
pred = clf.predict(X[512:])
proba = clf.predict_proba(X[512:])
```

## Evaluation

Results from a 115K-param model pretrained for 1500 steps on CPU (~2.5 min):

**Pretrained vs baselines** (20 binary classification tasks):

| Model | Mean accuracy | Std |
|---|---|---|
| **Pretrained** | **0.674** | 0.132 |
| Random weights | 0.491 | 0.040 |
| Majority class | 0.464 | 0.021 |

**Label ablation** – scrambling training labels drops accuracy to chance,
confirming the model learns from the label-feature association via ICL
attention rather than memorising patterns from pretraining:

| Condition | Accuracy |
|---|---|
| Real labels | 0.698 |
| Scrambled labels | 0.530 |

**Scaling** – accuracy with more training rows:

| n_train | Accuracy |
|---|---|
| 16 | 0.869 |
| 64 | 0.865 |
| 256 | 0.873 |
| 512 | 0.873 |

> The flat scaling curve (0.865–0.873) suggests that this particular
> evaluation task is relatively easy and saturates quickly. With harder
> tasks or more features the gap between small and large context sizes
> would be more pronounced. Re-running with different seeds and task
> difficulties is recommended for a fuller picture.

The real TabICLv2 has ~50M+ params and trains for days on GPUs with much
richer synthetic priors.  The architecture here is faithful; the scale is
intentionally tiny.

## Package layout

- `src/nanotabicl/nano_model.py` – architecture, pretraining, inference (single file).
- `src/nanotabicl/demo.py` – toy classification and regression demos.
- `docs/` – static website for TabICLv2 explanation and toy browser demo.

## Next ideas

- Add optional visualization of attention maps.
- Add multi-GPU / Modal cloud pretraining script.
