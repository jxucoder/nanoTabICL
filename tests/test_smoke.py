from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch

from nanotabicl import (
    NanoTabICL,
    NanoTabICLClassifier,
    NanoTabICLConfig,
    NanoTabICLRegressor,
    load_checkpoint,
    save_checkpoint,
)


class NanoTabICLSmokeTests(unittest.TestCase):
    def test_classifier_shapes(self) -> None:
        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(80, 6)).astype(np.float32)
        y_train = (x_train[:, 0] - x_train[:, 1] > 0).astype(np.int64)
        x_test = rng.normal(size=(16, 6)).astype(np.float32)

        model = NanoTabICLClassifier()
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)

        self.assertEqual(pred.shape, (16,))
        self.assertEqual(proba.shape, (16, 2))
        self.assertTrue(np.allclose(np.sum(proba, axis=1), 1.0, atol=1e-5))

    def test_regressor_shapes(self) -> None:
        rng = np.random.default_rng(1)
        x_train = rng.normal(size=(80, 5)).astype(np.float32)
        y_train = np.tanh(0.8 * x_train[:, 0] - 0.5 * x_train[:, 1]).astype(
            np.float32
        )
        x_test = rng.normal(size=(20, 5)).astype(np.float32)

        model = NanoTabICLRegressor()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        self.assertEqual(pred.shape, (20,))
        self.assertFalse(np.isnan(pred).any())

    def test_output_changes_with_input(self) -> None:
        """Verify outputs change after a few training steps."""
        from nanotabicl import PretrainConfig, pretrain

        cfg = NanoTabICLConfig(d_model=32, nhead=2, icl_num_blocks=1)
        model = NanoTabICL(cfg)
        # Briefly train so the model is no longer at zero-init
        pretrain(
            model,
            PretrainConfig(
                steps=5, batch_size=2, log_every=100, device="cpu",
                n_rows=(32, 64), n_features=(3, 5), n_classes=(2, 3),
            ),
        )
        model.eval()
        torch.manual_seed(0)
        X1 = torch.randn(1, 20, 4)
        X2 = torch.randn(1, 20, 4)
        y = torch.zeros(1, 15, dtype=torch.long)
        with torch.no_grad():
            out1 = model(X1, y)
            out2 = model(X2, y)
        self.assertFalse(torch.allclose(out1, out2, atol=1e-6))

    def test_gradients_flow(self) -> None:
        """Verify gradients propagate through all three stages."""
        cfg = NanoTabICLConfig(d_model=32, nhead=2, icl_num_blocks=1)
        model = NanoTabICL(cfg).train()
        X = torch.randn(1, 20, 4)
        y = torch.zeros(1, 15, dtype=torch.long)
        logits = model(X, y)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(
                    p.grad, f"No gradient for {name}"
                )

    def test_checkpoint_roundtrip(self) -> None:
        """Verify save/load produces identical outputs."""
        cfg = NanoTabICLConfig(d_model=32, nhead=2, icl_num_blocks=1)
        model = NanoTabICL(cfg).eval()
        X = torch.randn(1, 20, 4)
        y = torch.zeros(1, 15, dtype=torch.long)
        with torch.no_grad():
            out_before = model(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        save_checkpoint(model, path)
        loaded = load_checkpoint(path)
        with torch.no_grad():
            out_after = loaded(X, y)

        self.assertTrue(
            torch.allclose(out_before, out_after, atol=1e-6),
            "Checkpoint round-trip changed model outputs",
        )

    def test_pretrain_loss_decreases(self) -> None:
        """Verify loss decreases over a short pretraining run."""
        from nanotabicl import PretrainConfig, pretrain

        cfg = NanoTabICLConfig(d_model=32, nhead=2, icl_num_blocks=1)
        model = NanoTabICL(cfg)
        losses = pretrain(
            model,
            PretrainConfig(
                steps=40, batch_size=2, log_every=100, device="cpu",
                n_rows=(32, 64), n_features=(3, 5), n_classes=(2, 3),
            ),
        )
        # compare first and last 10-step averages
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        self.assertLess(
            late, early, "Loss did not decrease during pretraining"
        )


class SklearnWrapperIntegrationTests(unittest.TestCase):
    """End-to-end integration tests for the sklearn-compatible wrappers."""

    def test_classifier_fit_predict_cycle(self) -> None:
        """Full fit/predict/predict_proba cycle on synthetic data."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 4)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)

        clf = NanoTabICLClassifier()
        clf.fit(x[:80], y[:80])

        pred = clf.predict(x[80:])
        proba = clf.predict_proba(x[80:])

        self.assertEqual(pred.shape, (20,))
        self.assertEqual(proba.shape, (20, 2))
        # all predictions should be valid class labels
        self.assertTrue(set(pred.tolist()).issubset({0, 1}))
        # probabilities should sum to 1
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0, atol=1e-5))

    def test_classifier_multiclass(self) -> None:
        """Classifier handles 3+ classes."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 5)).astype(np.float32)
        y = (x[:, 0] > 0.5).astype(np.int64) + (x[:, 1] > 0).astype(np.int64)

        clf = NanoTabICLClassifier()
        clf.fit(x[:100], y[:100])
        pred = clf.predict(x[100:])
        proba = clf.predict_proba(x[100:])

        n_classes = len(np.unique(y[:100]))
        self.assertEqual(pred.shape, (20,))
        self.assertEqual(proba.shape[0], 20)
        self.assertEqual(proba.shape[1], n_classes)

    def test_classifier_single_feature(self) -> None:
        """Classifier works with a single feature."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 1)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)

        clf = NanoTabICLClassifier()
        clf.fit(x[:40], y[:40])
        pred = clf.predict(x[40:])
        self.assertEqual(pred.shape, (20,))

    def test_regressor_fit_predict_cycle(self) -> None:
        """Full fit/predict cycle for regression."""
        rng = np.random.default_rng(99)
        x = rng.normal(size=(100, 3)).astype(np.float32)
        y = (0.5 * x[:, 0] - 0.3 * x[:, 1]).astype(np.float32)

        reg = NanoTabICLRegressor()
        reg.fit(x[:80], y[:80])
        pred = reg.predict(x[80:])

        self.assertEqual(pred.shape, (20,))
        self.assertFalse(np.isnan(pred).any())
        self.assertFalse(np.isinf(pred).any())

    def test_predict_before_fit_raises(self) -> None:
        """Calling predict before fit raises RuntimeError."""
        clf = NanoTabICLClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict(np.zeros((5, 3), dtype=np.float32))

        reg = NanoTabICLRegressor()
        with self.assertRaises(RuntimeError):
            reg.predict(np.zeros((5, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
