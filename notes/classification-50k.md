# Classification Experiment — 50K Steps (2025-03-06)

## Architecture (d_model=256, 73.7M params)

- 3-stage pipeline: ColEmbedding (ISAB) → RowInteraction (CLS-token transformer) → ICLearning (row-level transformer)
- ICL stage attends test rows **only** to train rows (train-only context)
- Classification uses `max_classes=10`; regression uses `max_classes=0` (separate architecture)

## Training Config

- steps=50,000, batch_size=16, lr=1e-4, warmup=500
- n_rows=(64, 512), n_features=(3, 30), n_classes=(2, 10)
- Cosine LR schedule, AMP enabled, A10G GPU

## Results

| Method | nanoTabICL | KNN-5 | KNN-10 | SVM | RF | LogReg |
|--------|-----------|-------|--------|-----|-----|--------|
| Avg accuracy | 0.709 | 0.650 | 0.680 | 0.755 | 0.691 | 0.797 |

### Per-task breakdown

| Task | feat | cls | nanoTabICL | KNN-5 | KNN-10 | SVM | RF | LogReg |
|------|------|-----|-----------|-------|--------|-----|-----|--------|
| 1 | 4 | 2 | 0.953 | 0.930 | 0.914 | 0.930 | 0.930 | 0.930 |
| 2 | 6 | 3 | 0.891 | 0.805 | 0.828 | 0.891 | 0.812 | 0.875 |
| 3 | 8 | 4 | 0.750 | 0.656 | 0.766 | 0.828 | 0.734 | 0.867 |
| 4 | 10 | 5 | 0.484 | 0.430 | 0.453 | 0.562 | 0.477 | 0.727 |
| 5 | 12 | 6 | 0.469 | 0.430 | 0.438 | 0.562 | 0.500 | 0.586 |

## Key Findings

- **Underfitting is the main issue.** 50K steps × bs16 = 800K examples for 73.7M params is far too few. Loss was still decreasing (1.09) when training stopped.
- Degrades sharply above 4 classes.
- Regression: untrained model (can't reuse cls checkpoint), so results are not meaningful.
- Cosine LR schedule decays to ~0 by end of run — wasted the last portion of training.
- Train context length (64-512 rows at 0.75 ratio) should match eval (512 train rows) more closely.

## Next Steps

- Train for ≥200K steps (target ≤0.7 loss) before evaluating
- Use `n_rows=(64, 1024)` so the model sees long-context examples matching eval
- Consider checkpointing every N steps for resumability
- Always run with `--detach` on Modal to survive network drops
