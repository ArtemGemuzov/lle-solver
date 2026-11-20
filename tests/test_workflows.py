"""Проверки API: .venv/Scripts/python.exe -m unittest discover -s tests -v."""

import os
os.environ["DDE_BACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import contextlib
import io
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lle.config import load_config
from lle.numerics.grid import PeriodicGrid
from lle.numerics.integrators import rk4_step
from lle.numerics.lle import LugiatoLefever
from lle.numerics.split_schemes import StrangSplitStep
from lle.numerics.steps import FourierStep, IntegratorStep
from lle.pinn.checkpoint import PinnCheckpoint, load_checkpoint
from lle.pinn.model import build_model, initial_weights
from lle.pinn.training import PinnTraining
from lle.runs import runs
from lle.storage import load_evolution, save_evolution
from lle.workflows import pinn_gaussian, ssfm_gaussian


class WorkflowsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(runs, "RUNS_ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        torch.set_default_device("cpu")

    def config(self, steps=6, **updates):
        values = {
            "domain": {"time_end": .02, "interior_points": 8, "initial_points": 8},
            "network": {"hidden_layers": 1, "hidden_width": 8, "spatial_harmonics": 2},
            "training": {"iterations": steps, "resample_every": 2, "log_every": 1},
            "recording": {"interval": .01, "num_points": 8},
            "checkpoint": {"every_steps": 3},
        }
        values.update(updates)
        return pinn_gaussian.PinnGaussianConfig.model_validate(values)

    def assert_same(self, a, b):
        left = load_checkpoint(a.path / "checkpoint.pt")
        right = load_checkpoint(b.path / "checkpoint.pt")
        for key in left["model"]:
            torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)
        np.testing.assert_array_equal(load_evolution(a.path).values, load_evolution(b.path).values)

    def test_ssfm_and_storage(self):
        config = load_config(ssfm_gaussian.SsfmGaussianConfig, overrides=[
            "grid.num_points=32", "solver.time_end=0.02", "recording.interval=0.01"
        ])
        field = ssfm_gaussian.simulate(config)
        save_evolution(self.root, field)
        np.testing.assert_array_equal(load_evolution(self.root).values, field.values)
        # Однородное поле без накачки: известны затухание и нелинейная фаза.
        grid = PeriodicGrid(16, 2 * np.pi)
        equation = LugiatoLefever(3., 0., -.1)
        scheme = StrangSplitStep(FourierStep(equation.linear_operator, grid.wavenumbers),
                                IntegratorStep(equation.nonlinear_term, rk4_step))
        start = .7 + .2j
        actual = scheme.advance(np.full(16, start, dtype=np.complex128), 100, .001)
        expected = start * np.exp(-.1 + 1j * (abs(start)**2 * (1-np.exp(-.2))/2 - .3))
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)

    def test_extend_matches_uninterrupted(self):
        full = pinn_gaussian.train(self.config())
        short = pinn_gaussian.train(self.config(3))
        extended = pinn_gaussian.extend(short.id, 3)
        self.assert_same(full, extended)
        self.assertEqual(extended.load_metadata()["parent_run_id"], short.id)
        self.assertEqual(short.load_metadata()["status"], "complete")

    def test_resume_matches_uninterrupted(self):
        for loss in ({}, {"relobralo": {}}, {"causal": {"bands": 2}},
                     {"causal": {"bands": 2, "schedule": {"epsilons": [.1, 1.], "threshold": .999999}}}):
            with self.subTest(loss=loss):
                config = self.config(loss=loss)
                full = pinn_gaussian.train(config)
                original = PinnCheckpoint.save
                def interrupt(checkpoint):
                    original(checkpoint)
                    if checkpoint.model.train_state.step == 3:
                        raise RuntimeError("Test interruption")
                with patch.object(PinnCheckpoint, "save", interrupt):
                    with self.assertRaisesRegex(RuntimeError, "Test interruption"):
                        pinn_gaussian.train(config)
                interrupted = runs.list_runs()[-1]
                self.assertEqual(interrupted.load_metadata()["status"], "incomplete")
                resumed = pinn_gaussian.resume(interrupted.id)
                self.assertEqual(resumed.id, interrupted.id)
                self.assert_same(full, resumed)
                with self.assertRaisesRegex(ValueError, "Only incomplete"):
                    pinn_gaussian.resume(resumed)

    def test_transfer_preserves_complex_prediction(self):
        source_config = self.config(3)
        source = pinn_gaussian.train(source_config)
        target_config = self.config(2, domain={"time_end": .04, "interior_points": 8, "initial_points": 8})
        def model(config):
            return build_model(equation=config.equation, initial=config.initial,
                domain=config.domain, network=config.network, training=config.training, loss=config.loss)
        saved = load_checkpoint(source.path / "checkpoint.pt")
        old, new = model(source_config), model(target_config)
        old.net.load_state_dict(saved["model"])
        new.net.load_state_dict(initial_weights(saved["model"], time_ratio=target_config.domain.time_end / source_config.domain.time_end))
        points = np.array([[-1., 0.], [0., .01], [1., .02]], dtype=np.float32)
        np.testing.assert_allclose(new.predict(points), old.predict(points), atol=3e-7, rtol=0)
        transferred = pinn_gaussian.transfer(source, target_config)
        self.assertEqual(load_checkpoint(transferred.path / "checkpoint.pt")["train_state"]["step"], 2)
        self.assertEqual(transferred.load_metadata()["parent_mode"], "init")

    def test_cli(self):
        with patch.object(sys, "argv", ["scripts/gaussian.py", "solver.time_end=0.01", "recording.interval=0.01", "grid.num_points=16"]):
            runpy.run_path("scripts/gaussian.py", run_name="__main__")
        config_path = self.root / "config.yaml"
        import yaml
        config_path.write_text(yaml.safe_dump(self.config(2).model_dump(mode="json")), encoding="utf-8")
        with patch.object(sys, "argv", ["scripts/pinn_gaussian.py", "train", "--config", str(config_path)]):
            runpy.run_path("scripts/pinn_gaussian.py", run_name="__main__")
        source = runs.list_runs()[-1]
        for command in (["extend", source.id, "--steps", "1"], ["transfer", source.id, "training.iterations=1"]):
            with patch.object(sys, "argv", ["scripts/pinn_gaussian.py", *command]):
                runpy.run_path("scripts/pinn_gaussian.py", run_name="__main__")
        self.assertTrue(all(r.load_metadata()["status"] == "complete" for r in runs.list_runs()))

    def test_checkpoint_with_opaque_parameters(self):
        config = self.config(2)
        model = build_model(equation=config.equation, initial=config.initial,
            domain=config.domain, network=config.network, training=config.training, loss=config.loss)
        parameters = {"description": "Independent training"}
        training = PinnTraining(model, config.training,
            checkpoint_path=self.root / "checkpoint.pt", checkpoint_every_steps=1,
            parameters=parameters)
        training.fit()
        saved = load_checkpoint(self.root / "checkpoint.pt")
        self.assertEqual(saved["config"], parameters)
        self.assertEqual(saved["train_state"]["step"], 2)
        training.checkpoint.restore(saved)
        self.assertTrue(training.checkpoint.complete)

    def test_workflow_rejects_incompatible_operations(self):
        source = pinn_gaussian.train(self.config(3))
        count = len(runs.list_runs())
        with self.assertRaisesRegex(ValueError, "positive"):
            pinn_gaussian.extend(source, 0)
        for updates, message in (
            ({"network": {"hidden_layers": 1, "hidden_width": 9, "spatial_harmonics": 2}}, "network"),
            ({"domain": {"length": 4., "time_end": .02, "interior_points": 8, "initial_points": 8}}, "domain.length"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                pinn_gaussian.transfer(source, self.config(**updates))
        self.assertEqual(len(runs.list_runs()), count)
        scheduled = pinn_gaussian.train(self.config(loss={
            "causal": {"bands": 2, "schedule": {"epsilons": [.1, 1.]}}
        }))
        with self.assertRaisesRegex(ValueError, "schedule"):
            pinn_gaussian.extend(scheduled, 1)
        self.assertEqual(len(runs.list_runs()), count + 1)


if __name__ == "__main__":
    unittest.main()
