"""Проверки RAD и воспроизводимого продолжения обучения."""

import os

os.environ["DDE_BACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_workflows
from lle.pinn.sampling import residual_probabilities
from lle.runs.checkpoint import load_checkpoint
from lle.runs import runs
from lle.workflows import pinn_gaussian
from lle.pinn.training import TrainingCheckpoint


class SamplingTest(unittest.TestCase):
    setUp = test_workflows.WorkflowsTest.setUp
    config = test_workflows.WorkflowsTest.config
    assert_same = test_workflows.WorkflowsTest.assert_same

    def rad_config(self, steps=6, warmup=2):
        return self.config(
            steps,
            sampling={
                "rad": {
                    "candidate_points": 32,
                    "batch_size": 7,
                    "warmup_steps": warmup,
                }
            },
            checkpoint={"every_steps": 1},
        )

    def test_probabilities(self):
        np.testing.assert_allclose(
            residual_probabilities([0, 1, 3], 1, 0.5), [1 / 6, 7 / 24, 13 / 24]
        )
        np.testing.assert_allclose(residual_probabilities([0, 0], 2, 0), [0.5, 0.5])
        np.testing.assert_allclose(
            residual_probabilities([1e300, 1e300], 4, 0), [0.5, 0.5]
        )
        with self.assertRaises(FloatingPointError):
            residual_probabilities([np.nan], 1, 0.5)

    def test_resume_and_extend(self):
        for warmup in (0, 2, 5):
            config = self.rad_config(warmup=warmup)
            full = pinn_gaussian.train(config)
            full_state = load_checkpoint(full.path / "checkpoint.pt")
            expected_steps = list(range(warmup, 7, 2))
            self.assertEqual(
                [r["step"] for r in full_state["sampling"]["history"]], expected_steps
            )
            for split in (1, 2, 3):
                with self.subTest(warmup=warmup, split=split):
                    short = pinn_gaussian.train(self.rad_config(split, warmup))
                    extended = pinn_gaussian.extend(short, 6 - split)
                    self.assert_same(full, extended)
                    original = TrainingCheckpoint.save

                    def interrupt(callback):
                        original(callback)
                        if callback.model.train_state.step == split:
                            raise RuntimeError("Test interruption")

                    with patch.object(TrainingCheckpoint, "save", interrupt):
                        with self.assertRaisesRegex(RuntimeError, "Test interruption"):
                            pinn_gaussian.train(config)
                    resumed = pinn_gaussian.resume(runs.list_runs()[-1])
                    self.assert_same(full, resumed)
                    saved = load_checkpoint(resumed.path / "checkpoint.pt")
                    for key in ("train_x_all", "train_x_bc", "test_x"):
                        np.testing.assert_array_equal(
                            full_state["data"][key], saved["data"][key]
                        )
                    self.assertEqual(
                        full_state["sampling"]["rng"], saved["sampling"]["rng"]
                    )

    def test_fixed_ic_and_test_points(self):
        before = pinn_gaussian.train(self.rad_config(1))
        after = pinn_gaussian.extend(before, 2)
        left = load_checkpoint(before.path / "checkpoint.pt")
        right = load_checkpoint(after.path / "checkpoint.pt")
        for key in ("train_x_bc", "test_x"):
            np.testing.assert_array_equal(left["data"][key], right["data"][key])
        self.assertEqual(right["data"]["train_x_all"].shape, (16, 2))
        with np.load(after.path / "sampling_histograms.npz") as hist:
            np.testing.assert_array_equal(hist["counts"].sum(axis=(1, 2)), [8])
        self.assertTrue((after.path / "sampling.csv").exists())

    def test_transfer_resets_sampling(self):
        source = pinn_gaussian.train(self.rad_config(3))
        child = pinn_gaussian.transfer(source, self.rad_config(1))
        state = load_checkpoint(child.path / "checkpoint.pt")
        self.assertEqual(state["sampling"]["history"], [])
        self.assertIsNone(state["sampling"]["last_update"])

    def test_cli(self):
        args = [
            "scripts/pinn_gaussian.py",
            "train",
            "--config",
            "experiments/06_pinn-rad-sampling/rad_config.yaml",
            "domain.time_end=0.02",
            "recording.interval=0.01",
            "recording.num_points=8",
            "domain.interior_points=8",
            "domain.initial_points=8",
            "network.hidden_layers=1",
            "network.hidden_width=8",
            "training.iterations=2",
            "training.resample_every=1",
            "sampling.rad.candidate_points=8",
            "sampling.rad.replace=false",
            "sampling.rad.batch_size=7",
            "sampling.rad.warmup_steps=1",
        ]
        with patch.object(sys, "argv", args):
            runpy.run_path(args[0], run_name="__main__")
        saved = load_checkpoint(runs.list_runs()[-1].path / "checkpoint.pt")
        self.assertEqual([r["step"] for r in saved["sampling"]["history"]], [1, 2])
        self.assertEqual([r["unique_points"] for r in saved["sampling"]["history"]], [8, 8])

    def test_legacy_checkpoint(self):
        source = pinn_gaussian.train(self.config(3))
        full = pinn_gaussian.train(self.config(6))
        from lle.runs.checkpoint import restore_rng, save_checkpoint

        saved = load_checkpoint(source.path / "checkpoint.pt")
        saved.pop("sampling")
        saved.pop("anchors")
        restore_rng(saved)
        save_checkpoint(source.path / "checkpoint.pt", saved, saved["config"])
        extended = pinn_gaussian.extend(source, 3)
        self.assert_same(full, extended)

    def test_known_complex_solution(self):
        from lle.physics.equation import LLEParameters
        from lle.pinn.model import evaluate_residual, lle_residual

        class ExactField(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.amplitude = torch.nn.Parameter(
                    torch.tensor(0.7, dtype=torch.float64)
                )

            def forward(self, points):
                t = points[:, 1:2]
                # Однородное поле с затуханием и нелинейной фазой, pump=0.
                amplitude = self.amplitude * torch.exp(-t)
                phase = self.amplitude**2 * (1 - torch.exp(-2 * t)) / 2 - 3 * t
                zero = points[:, :1] ** 2 * 0
                return torch.cat(
                    (
                        amplitude * torch.cos(phase) + zero,
                        amplitude * torch.sin(phase) + zero,
                    ),
                    dim=1,
                )

        points = np.random.default_rng(1).uniform([-3, 0], [3, 2], (32, 2))
        residual = evaluate_residual(
            ExactField(), points, lle_residual(LLEParameters(3, 0, -0.1))
        )
        np.testing.assert_allclose(residual, 0, atol=1e-14)

    def test_invalid_configuration(self):
        from pydantic import ValidationError

        for values in (
            {"candidate_points": 4},
            {"batch_size": 0},
            {"power": 0.0},
            {"uniform_fraction": 1.1},
            {"warmup_steps": -1},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                self.config(sampling={"rad": values})
        with self.assertRaisesRegex(ValidationError, "fixed loss weights"):
            self.config(sampling={"rad": {}}, loss={"causal": {}})
