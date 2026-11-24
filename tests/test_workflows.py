"""Проверки API: .venv/Scripts/python.exe -m unittest discover -s tests -v."""

import os

os.environ["DDE_BACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lle.cli.main import main
from lle.config import load_config
from lle.numerics.grid import PeriodicGrid
from lle.numerics.integrators import rk4_step
from functools import partial
from lle.numerics.lle import linear_operator, nonlinear_term
from lle.numerics.split_schemes import StrangSplitStep
from lle.numerics.ssfm import solve_strang_rk4
from lle.numerics.steps import FourierStep, IntegratorStep
from lle.runs.checkpoint import load_checkpoint, save_checkpoint
from lle.pinn.training import TrainingCheckpoint
from lle.physics.equation import LLEParameters
from lle.physics.initial import GaussianPulse
from lle.physics.problem import LLEProblem
from lle.pinn.features import PeriodicFeatures
from lle.pinn.model import (
    build_model,
    build_network,
    lle_residual,
    evaluate_residual,
    predict_field,
)
from lle.pinn.parameters import LossParameters, NetworkParameters, TrainingParameters
from lle.pinn.solver import train
from lle.pinn.training import PinnTraining, configure_adam, predict_evolution, train_model
from lle.runs import runs
from lle.storage import load_evolution
from lle.workflows import pinn_gaussian
from lle.workflows.ssfm_gaussian.config import SsfmGaussianConfig
from lle.workflows.ssfm_gaussian.workflow import run as run_ssfm, simulate


def _build_model(config):
    return build_model(
        config.to_problem(), network=NetworkParameters(**config.network.model_dump()), loss=config.loss_parameters(),
        interior_points=config.domain.interior_points, initial_points=config.domain.initial_points,
        seed=config.training.seed,
    )


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
            "domain": {"time_end": 0.02, "interior_points": 8, "initial_points": 8},
            "network": {"hidden_layers": 1, "hidden_width": 8, "spatial_harmonics": 2},
            "training": {"iterations": steps, "resample_every": 2, "log_every": 1},
            "recording": {"interval": 0.01, "num_points": 8},
            "checkpoint": {"every_steps": 3},
        }
        values.update(updates)
        return pinn_gaussian.PinnGaussianConfig.model_validate(values)

    def assert_same(self, a, b):
        left = load_checkpoint(a.path / "checkpoint.pt")
        right = load_checkpoint(b.path / "checkpoint.pt")
        for key in left["model"]:
            torch.testing.assert_close(
                left["model"][key], right["model"][key], rtol=0, atol=0
            )
        np.testing.assert_array_equal(
            load_evolution(a.path).values, load_evolution(b.path).values
        )

    def test_ssfm_and_storage(self):
        config = load_config(
            SsfmGaussianConfig,
            overrides=[
                "grid.num_points=32",
                "solver.time_end=0.02",
                "recording.interval=0.01",
            ],
        )
        field = simulate(config)
        run = run_ssfm(config)
        self.assertEqual(run.load_metadata()["status"], "complete")
        self.assertEqual(runs.list_runs(script="scripts/gaussian.py"), [run])
        self.assertEqual(run.load_config(SsfmGaussianConfig), config)
        np.testing.assert_array_equal(load_evolution(run.path).values, field.values)
        # Однородное поле без накачки: известны затухание и нелинейная фаза.
        grid = PeriodicGrid(16, 2 * np.pi)
        equation = LLEParameters(3.0, 0.0, -0.1)
        scheme = StrangSplitStep(
            FourierStep(partial(linear_operator, equation), grid.wavenumbers),
            IntegratorStep(partial(nonlinear_term, equation), rk4_step),
        )
        start = 0.7 + 0.2j
        actual = scheme.advance(np.full(16, start, dtype=np.complex128), 100, 0.001)
        expected = start * np.exp(
            -0.1 + 1j * (abs(start) ** 2 * (1 - np.exp(-0.2)) / 2 - 0.3)
        )
        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)

    def test_training_in_memory_matches_saved_run(self):
        config = self.config(5)
        model = build_model(
            config.to_problem(), network=NetworkParameters(**config.network.model_dump()), loss=config.loss_parameters(),
            interior_points=config.domain.interior_points, initial_points=config.domain.initial_points,
            seed=config.training.seed,
        )
        snapshots = []
        history = train_model(
            model, TrainingParameters(**config.training.model_dump()), checkpoint_every=2, on_checkpoint=snapshots.append
        )
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertIs(history, model.losshistory)
        self.assertEqual([state["train_state"]["step"] for state in snapshots], [2, 4, 5])
        run = pinn_gaussian.train_run(config)
        saved = load_checkpoint(run.path / "checkpoint.pt")
        for name, weights in snapshots[-1]["model"].items():
            torch.testing.assert_close(weights, saved["model"][name], rtol=0, atol=0)
        np.testing.assert_array_equal(model.losshistory.loss_train, saved["history"]["loss_train"])
        reference = load_evolution(run.path)
        predicted = predict_evolution(model.net, theta=reference.theta, times=reference.times)
        np.testing.assert_array_equal(predicted.values, reference.values)

    def test_high_level_prediction_matches_run(self):
        config = self.config(3)
        from lle.pinn.model import build_periodic_network
        problem = config.to_problem()
        torch.manual_seed(42)
        network = build_periodic_network(PeriodicFeatures(problem.length, problem.time_end, 2),
                                         hidden_layers=1, hidden_width=8)
        solution = train(network, problem.equation, problem.domain, problem.initial, steps=3,
                         interior_points=8, initial_points=8, resample_every=2, log_every=1)
        self.assertIs(solution.model.net, network)
        self.assertEqual(list(self.root.iterdir()), [])
        run = pinn_gaussian.train_run(config)
        field = load_evolution(run.path)
        for times in (field.times, field.times[::-1]):
            predicted = solution.predict(theta=field.theta, times=times)
            indices = np.searchsorted(field.times, times)
            np.testing.assert_array_equal(predicted.values, field.values[indices])
        saved = load_checkpoint(run.path / "checkpoint.pt")
        np.testing.assert_array_equal(solution.history.loss_train, saved["history"]["loss_train"])

    def test_core_continuation_from_memory(self):
        from lle.pinn.solver import extend, resume, transfer

        problem = LLEProblem(LLEParameters(1.5, 1.0, -0.1),
                             lambda theta: np.full_like(theta, 0.5 - 0.5j, dtype=complex),
                             length=3.0, time_end=0.02)
        def inputs():
            network = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
            return network, problem.equation, problem.domain, problem.initial
        options = dict(interior_points=8, initial_points=8, resample_every=2, log_every=1)
        snapshots = []
        full = train(*inputs(), steps=6, on_checkpoint=snapshots.append, checkpoint_every=3, **options)
        source = snapshots[0]
        theta, times = [-0.2, 0.1], [0.0, 0.02]
        expected = full.predict(theta=theta, times=times).values
        for result in (resume(*inputs(), state=source, steps=6, **options),
                       extend(*inputs(), state=source, steps=3, **options)):
            np.testing.assert_array_equal(result.predict(theta=theta, times=times).values, expected)
            np.testing.assert_array_equal(result.history.loss_train, full.history.loss_train)
        finished = resume(*inputs(), state=full.state, steps=6, **options)
        np.testing.assert_array_equal(finished.history.steps, full.history.steps)
        child = transfer(*inputs(), state=full.state, steps=2, **options)
        self.assertEqual(child.state["train_state"]["step"], 2)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_extend_matches_uninterrupted(self):
        full = pinn_gaussian.train(self.config())
        short = pinn_gaussian.train(self.config(3))
        extended = pinn_gaussian.extend(short.id, 3)
        self.assert_same(full, extended)
        self.assertEqual(extended.load_metadata()["parent_run_id"], short.id)
        self.assertEqual(short.load_metadata()["status"], "complete")

    def test_resume_matches_uninterrupted(self):
        for loss in (
            {},
            {"relobralo": {}},
            {"causal": {"bands": 2}},
            {
                "causal": {
                    "bands": 2,
                    "schedule": {"epsilons": [0.1, 1.0], "threshold": 0.999999},
                }
            },
        ):
            with self.subTest(loss=loss):
                config = self.config(loss=loss)
                full = pinn_gaussian.train(config)
                original = TrainingCheckpoint.save

                def interrupt(checkpoint):
                    original(checkpoint)
                    if checkpoint.model.train_state.step == 3:
                        raise RuntimeError("Test interruption")

                with patch.object(TrainingCheckpoint, "save", interrupt):
                    with self.assertRaisesRegex(RuntimeError, "Test interruption"):
                        pinn_gaussian.train(config)
                interrupted = runs.list_runs()[-1]
                self.assertEqual(interrupted.load_metadata()["status"], "incomplete")
                main(["pinn", "resume", interrupted.id])
                resumed = runs.get_run(interrupted.id)
                self.assertEqual(resumed.id, interrupted.id)
                self.assert_same(full, resumed)
                with self.assertRaisesRegex(ValueError, "Only incomplete"):
                    pinn_gaussian.resume(resumed)

    def test_transfer_preserves_complex_prediction(self):
        source_config = self.config(3)
        source = pinn_gaussian.train(source_config)
        target_config = self.config(
            2, domain={"time_end": 0.04, "interior_points": 8, "initial_points": 8}
        )

        def model(config):
            return _build_model(config)

        saved = load_checkpoint(source.path / "checkpoint.pt")
        old, new = model(source_config), model(target_config)
        old.net.load_state_dict(saved["model"])
        new.net.load_state_dict(
            PeriodicFeatures(
                source_config.domain.length,
                source_config.domain.time_end,
                source_config.network.spatial_harmonics,
            ).transfer_weights(
                saved["model"],
                PeriodicFeatures(
                    target_config.domain.length,
                    target_config.domain.time_end,
                    target_config.network.spatial_harmonics,
                ),
            )
        )
        points = np.array([[-1.0, 0.0], [0.0, 0.01], [1.0, 0.02]], dtype=np.float32)
        np.testing.assert_allclose(
            predict_field(new.net, points),
            predict_field(old.net, points),
            atol=3e-7,
            rtol=0,
        )
        transferred = pinn_gaussian.transfer(source, target_config)
        self.assertEqual(
            load_checkpoint(transferred.path / "checkpoint.pt")["train_state"]["step"],
            2,
        )
        self.assertEqual(transferred.load_metadata()["parent_mode"], "init")

    def test_cli(self):
        main(["ssfm", "solve", "solver.time_end=0.01",
              "recording.interval=0.01", "grid.num_points=16"])
        self.assertEqual(runs.list_runs()[-1].load_metadata()["script"], "scripts/gaussian.py")
        config_path = self.root / "config.yaml"
        import yaml

        config_path.write_text(
            yaml.safe_dump(self.config(2).model_dump(mode="json")), encoding="utf-8"
        )
        main(["pinn", "train", "training.seed=17", "--config", str(config_path),
              "training.log_every=2"])
        source = runs.list_runs()[-1]
        self.assertEqual(source.load_metadata()["script"], "scripts/pinn_gaussian.py")
        self.assertEqual(source.load_config(pinn_gaussian.PinnGaussianConfig).training.seed, 17)
        for command in (
            ["extend", source.id, "--steps", "1"],
            ["transfer", source.id, "training.iterations=1"],
        ):
            main(["pinn", *command])
        self.assertTrue(
            all(r.load_metadata()["status"] == "complete" for r in runs.list_runs())
        )

    def test_checkpoint_with_opaque_parameters(self):
        config = self.config(2)
        model = _build_model(config)
        configure_adam(
            model,
            learning_rate=config.training.learning_rate,
            initial_weight=config.training.initial_weight,
        )
        parameters = {"description": "Independent training"}
        training = PinnTraining(
            model,
            iterations=config.training.iterations,
            resample_every=config.training.resample_every,
            log_every=config.training.log_every,
        )
        training.fit()
        save_checkpoint(self.root / "checkpoint.pt", training.state_dict(), parameters)
        saved = load_checkpoint(self.root / "checkpoint.pt")
        self.assertEqual(saved["config"], parameters)
        self.assertEqual(saved["train_state"]["step"], 2)
        training.load_state_dict(saved)
        self.assertTrue(training.complete)

    def test_prediction_without_optimizer(self):
        config = self.config(2)
        with patch(
            "deepxde.data.TimePDE",
            side_effect=AssertionError("No training data expected"),
        ):
            features = PeriodicFeatures(
                config.domain.length,
                config.domain.time_end,
                config.network.spatial_harmonics,
            )
            net = build_network(
                input_size=features.output_size,
                hidden_layers=config.network.hidden_layers,
                hidden_width=config.network.hidden_width,
            )
            net.apply_feature_transform(features)
            points = np.array([[-1.0, 0.0], [0.0, 0.01], [1.0, 0.02]], dtype=np.float32)
            standalone_field = predict_field(net, points)
            residual_fn = lle_residual(LLEParameters(**config.equation.model_dump()))
            standalone_residual = evaluate_residual(net, points, residual_fn)
        model = _build_model(config)
        model.net.load_state_dict(net.state_dict())
        self.assertIsNone(model.opt)
        points = np.array([[-1.0, 0.0], [0.0, 0.01], [1.0, 0.02]], dtype=np.float32)
        field = predict_field(model.net, points)
        residual = model.predict(points, operator=model.data.pde)
        np.testing.assert_array_equal(field, standalone_field)
        for actual, expected in zip(residual, standalone_residual):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertTrue(all(np.isfinite(component).all() for component in residual))
        self.assertIsNone(model.opt)
        configure_adam(
            model,
            learning_rate=config.training.learning_rate,
            initial_weight=config.training.initial_weight,
        )
        prediction = model.predict(points)
        np.testing.assert_array_equal(
            field,
            prediction[:, 0].astype(np.float64)
            + 1j * prediction[:, 1].astype(np.float64),
        )
        with self.assertRaisesRegex(ValueError, "already has an optimizer"):
            configure_adam(model, learning_rate=0.001, initial_weight=10.0)

    def test_plain_network_and_shared_equation(self):
        equation = LLEParameters(detuning=1.0, pump_amplitude=1.0, beta2=-0.1)
        net = build_network(input_size=2, hidden_layers=1, hidden_width=8)
        with torch.no_grad():
            for value in net.parameters():
                value.zero_()
            net.linears[-1].bias[0] = 1.0
        points = np.array([[-1.0, 0.0], [0.0, 0.1], [1.0, 0.2]], dtype=np.float32)
        np.testing.assert_array_equal(predict_field(net, points), np.ones(3))
        for component in evaluate_residual(net, points, lle_residual(equation)):
            np.testing.assert_allclose(component, 0.0, atol=1e-7)
        grid = PeriodicGrid(16)
        scheme = StrangSplitStep(
            FourierStep(partial(linear_operator, equation), grid.wavenumbers),
            IntegratorStep(partial(nonlinear_term, equation), rk4_step),
        )
        actual = scheme.advance(np.ones(16, dtype=np.complex128), 10, 0.0001)
        np.testing.assert_allclose(actual, np.ones(16), atol=1e-10, rtol=0)

    def test_shared_problem_on_reference_axes(self):
        value = 0.5 - 0.5j
        problem = LLEProblem(
            LLEParameters(detuning=1.5, pump_amplitude=1.0, beta2=-0.1),
            lambda theta: np.full(theta.shape, value, dtype=np.complex128),
            length=3.0,
            time_end=0.02,
        )
        reference = solve_strang_rk4(problem, num_points=16, time_step=0.0001, recording_interval=0.01)
        model = build_model(
            problem, network=NetworkParameters(), loss=LossParameters(), seed=42,
            interior_points=8, initial_points=8,
        )
        with torch.no_grad():
            for parameter in model.net.parameters():
                parameter.zero_()
            model.net.linears[-1].bias.copy_(torch.tensor([value.real, value.imag]))
        points = np.array([[0.2, 0.0], [-0.3, 0.01]], dtype=np.float32)
        for condition, expected in zip(model.data.bcs, (value.real, value.imag), strict=True):
            np.testing.assert_allclose(condition.func(points, 0, 2, None).cpu(), expected)
        for residual in evaluate_residual(model.net, points, model.data.pde):
            np.testing.assert_allclose(residual, 0.0, atol=1e-7)
        prediction = predict_evolution(model.net, theta=reference.theta, times=reference.times)
        np.testing.assert_array_equal(prediction.theta, reference.theta)
        np.testing.assert_array_equal(prediction.times, reference.times)
        np.testing.assert_allclose(prediction.values, reference.values, atol=1e-9, rtol=0)

    def test_prediction_on_arbitrary_axes(self):
        net = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            net.weight.copy_(torch.eye(2, dtype=torch.float64))
        theta = np.linspace(-1, 2, 3001)
        times = np.array([0.41, 0.0, 0.12])
        field = predict_evolution(net, theta=theta, times=times)
        np.testing.assert_array_equal(field.values, theta[None, :] + 1j * times[:, None])
        for axis in ([], [[0]], [np.nan]):
            with self.assertRaises(ValueError):
                predict_evolution(net, theta=axis, times=[0])

    def test_shared_initial_and_independent_state(self):
        config = self.config(2)
        pulse = GaussianPulse(**config.initial.model_dump())
        model = _build_model(config)
        points = np.array([[-0.1, 0.0], [0.0, 0.0], [0.1, 0.0]])
        expected = pulse(points[:, :1])
        np.testing.assert_allclose(
            model.data.bcs[0].func(points, 0, 3, None).cpu(), expected.real, rtol=1e-7
        )
        configure_adam(model, learning_rate=0.001, initial_weight=10.0)
        training = PinnTraining(model, iterations=2, resample_every=2, log_every=1)
        saved = training.state_dict()
        original = saved["model"]["linears.0.weight"].clone()
        training.fit()
        torch.testing.assert_close(
            saved["model"]["linears.0.weight"], original, rtol=0, atol=0
        )
        self.assertIn("rng", saved)
        self.assertNotIn("config", saved)

    def test_workflow_rejects_incompatible_operations(self):
        source = pinn_gaussian.train(self.config(3))
        count = len(runs.list_runs())
        with self.assertRaisesRegex(ValueError, "positive"):
            pinn_gaussian.extend(source, 0)
        for updates, message in (
            (
                {
                    "network": {
                        "hidden_layers": 1,
                        "hidden_width": 9,
                        "spatial_harmonics": 2,
                    }
                },
                "network",
            ),
            (
                {
                    "domain": {
                        "length": 4.0,
                        "time_end": 0.02,
                        "interior_points": 8,
                        "initial_points": 8,
                    }
                },
                "domain.length",
            ),
        ):
            with self.assertRaisesRegex(ValueError, message):
                pinn_gaussian.transfer(source, self.config(**updates))
        self.assertEqual(len(runs.list_runs()), count)
        scheduled = pinn_gaussian.train(
            self.config(
                loss={"causal": {"bands": 2, "schedule": {"epsilons": [0.1, 1.0]}}}
            )
        )
        with self.assertRaisesRegex(ValueError, "schedule"):
            pinn_gaussian.extend(scheduled, 1)
        self.assertEqual(len(runs.list_runs()), count + 1)


if __name__ == "__main__":
    unittest.main()
