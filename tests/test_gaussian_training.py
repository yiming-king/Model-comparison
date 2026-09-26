"""Regression checks for the Gaussian training grid and unified networks."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.examples.gaussian.approximators.config import (
    NOTEBOOK_PRESETS,
    TrainingConfig,
    model_path,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "benchmark" / "examples" / "gaussian" / "notebooks"
MODELS = ("m1", "m2", "m3", "m4")
HISTORICAL_PRESETS = ("20d_10n", "40d_10n", "80d_10n", "40d_100n")
PRESETS = (
    "20d_10n",
    "40d_10n",
    "80d_10n",
    "20d_100n",
    "40d_100n",
    "80d_100n",
)


def _notebook_tree(model, preset):
    notebook = json.loads((NOTEBOOKS / f"{model}_s_{preset}.ipynb").read_text())
    return ast.parse(
        "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
    )


def _constant(node, variables):
    """Read notebook constants without executing notebook cells."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return variables[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _constant(node.left, variables) * _constant(node.right, variables)
    raise ValueError(ast.dump(node))


def _notebook_config(model, preset):
    tree = _notebook_tree(model, preset)
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = _constant(node.value, values)
        except (ValueError, KeyError):
            pass
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "DeepSet":
            for keyword in node.keywords:
                values[keyword.arg] = _constant(keyword.value, values)
        elif node.func.attr == "CosineDecay":
            for keyword in node.keywords:
                values[keyword.arg] = _constant(keyword.value, values)
    return values


@pytest.fixture(scope="module")
def indirect():
    pytest.importorskip("bayesflow")
    from benchmark.examples.gaussian.approximators import indirect

    return indirect


@pytest.mark.parametrize("preset", HISTORICAL_PRESETS)
@pytest.mark.parametrize("model", MODELS)
def test_presets_preserve_notebook_dimensions_and_batch_settings(model, preset):
    original = _notebook_config(model, preset)
    config = TrainingConfig.from_preset(preset)

    assert config.num_dims == original["num_dims"]
    assert config.num_obs == original["num_obs"]
    assert config.summary_dim == original["summary_dim"]
    assert config.batch_size == original["batch_size"]
    assert config.num_batches == original["num_batches_per_epoch"]
    assert config.learning_rate == original["initial_learning_rate"]


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_cover_both_observation_counts_with_identical_training_defaults(preset):
    config = TrainingConfig.from_preset(preset)
    summary, observations = preset.split("_")

    assert set(NOTEBOOK_PRESETS) == set(PRESETS)
    assert (config.num_dims, config.num_obs, config.summary_dim) == (
        20,
        int(observations[:-1]),
        int(summary[:-1]),
    )
    assert config.summary_base_distribution is None
    assert [config.epochs_for(model) for model in MODELS] == [100] * 4
    assert config.network_tag == f"{preset}_bf_default"


def test_100_observation_preset_retains_notebook_summary_dimension():
    # The preset tag records the 40-dimensional summary and 100 observations.
    config = TrainingConfig.from_preset("40d_100n")
    assert (config.num_dims, config.num_obs, config.summary_dim) == (20, 100, 40)


def test_explicit_epochs_and_run_suffix_are_respected(tmp_path):
    config = TrainingConfig.from_preset("80d_10n", epochs=3, run_suffix="trial")
    assert [config.epochs_for(model) for model in MODELS] == [3, 3, 3, 3]
    assert (
        model_path("m1", config, tmp_path).name == "m1_s_80d_10n_bf_default_trial.keras"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_dims": 0},
        {"num_obs": -1},
        {"summary_dim": 0},
        {"epochs": 0},
        {"batch_size": 0},
        {"num_batches": 0},
        {"learning_rate": 0},
        {"summary_base_distribution": "unsupported"},
        {"run_suffix": "../escape"},
    ],
)
def test_invalid_training_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        TrainingConfig(**overrides)


@pytest.mark.parametrize("model", MODELS)
def test_simulator_matches_notebook_prior_and_likelihood(indirect, model):
    config = TrainingConfig(num_dims=3, num_obs=4, summary_dim=3, seed=123)
    original = _notebook_config(model, "20d_10n")
    draws = indirect.get_simulator(model, config).sample(1)
    expected_rng = np.random.default_rng(config.seed)
    expected_mu = expected_rng.normal(
        loc=original["mu_prior_mean"],
        scale=original["mu_prior_std"],
        size=config.num_dims,
    )
    expected_x = expected_rng.normal(
        loc=expected_mu,
        scale=original["likelihood_std"],
        size=(config.num_obs, config.num_dims),
    )

    assert draws["mu"].shape == (1, 3)
    assert draws["x"].shape == (1, 4, 3)
    np.testing.assert_allclose(draws["mu"][0], expected_mu)
    np.testing.assert_allclose(draws["x"][0], expected_x)
    adapted = indirect.build_adapter()(draws)
    assert set(adapted) == {"inference_variables", "summary_variables"}
    assert adapted["inference_variables"].dtype == np.float32
    assert adapted["summary_variables"].dtype == np.float32
    np.testing.assert_allclose(adapted["inference_variables"], draws["mu"], rtol=1e-6)
    np.testing.assert_allclose(adapted["summary_variables"], draws["x"], rtol=1e-6)


def test_validation_is_reproducible_and_does_not_advance_training_rng(indirect):
    config = TrainingConfig(num_dims=3, num_obs=4, summary_dim=3)
    training = indirect.get_simulator("m2", config)
    control = indirect.get_simulator("m2", config)
    training.sample(2)
    control.sample(2)

    first = indirect.generate_validation_data("m2", config, 3, 987)
    second = indirect.generate_validation_data("m2", config, 3, 987)
    different = indirect.generate_validation_data("m2", config, 3, 988)
    actual_training = training.sample(2)
    expected_training = control.sample(2)
    for key in ("mu", "x"):
        np.testing.assert_array_equal(first[key], second[key])
        assert not np.array_equal(first[key], different[key])
        np.testing.assert_array_equal(actual_training[key], expected_training[key])


def _without_keras_names(value):
    """Ignore auto-incremented layer names when comparing realized configs."""
    if isinstance(value, dict):
        return {
            key: _without_keras_names(item)
            for key, item in value.items()
            if key != "name"
        }
    if isinstance(value, (list, tuple)):
        return [_without_keras_names(item) for item in value]
    return value


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("model", MODELS)
def test_workflow_uses_bayesflow_defaults_and_requested_optimizer(
    indirect, model, preset
):
    config = TrainingConfig.from_preset(
        preset, epochs=3, num_batches=5, learning_rate=2e-4
    )
    workflow = indirect.build_workflow(model, config)
    import bayesflow as bf

    expected_flow = bf.networks.CouplingFlow().get_config()
    actual_flow = workflow.approximator.inference_network.get_config()
    assert _without_keras_names(actual_flow) == _without_keras_names(expected_flow)
    expected_summary = bf.networks.DeepSet(summary_dim=config.summary_dim).get_config()
    summary = workflow.approximator.summary_network.get_config()
    assert _without_keras_names(summary) == _without_keras_names(expected_summary)
    assert summary["summary_dim"] == config.summary_dim
    assert summary.get("base_distribution") is None
    assert workflow.approximator.standardizer.standardize == "all"
    optimizer = workflow.optimizer
    schedule = optimizer.get_config()["learning_rate"]
    assert type(optimizer).__name__ == "Adam"
    assert schedule["class_name"] == "CosineDecay"
    assert schedule["config"]["initial_learning_rate"] == pytest.approx(2e-4)
    assert schedule["config"]["decay_steps"] == 15
    # BayesFlow builds its default optimizer at fit time; ours must survive that step.
    assert (
        workflow.build_optimizer(epochs=3, num_batches=5, strategy="online")
        is optimizer
    )


def test_summary_mmd_is_opt_in_and_uses_a_separate_archive(indirect, tmp_path):
    config = TrainingConfig(summary_base_distribution="normal")
    workflow = indirect.build_workflow("m1", config)
    assert (
        workflow.approximator.summary_network.get_config()["base_distribution"]
        == "normal"
    )
    assert (
        model_path("m1", config, tmp_path).name == "m1_s_20d_10n_bf_default_mmd.keras"
    )
    assert model_path("m1", config, tmp_path) != model_path(
        "m1", TrainingConfig(), tmp_path
    )
    suffixed = TrainingConfig(summary_base_distribution="normal", run_suffix="trial")
    assert (
        model_path("m1", suffixed, tmp_path).name
        == "m1_s_20d_10n_bf_default_mmd_trial.keras"
    )


def _run_cli(module, *args):
    # The command planner must work even on a machine without ML dependencies.
    script = """
import runpy
import sys
class NoMLImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bayesflow', 'keras', 'tensorflow', 'torch', 'jax'}:
            raise AssertionError('Unexpected ML import: ' + fullname)
sys.meta_path.insert(0, NoMLImports())
module = sys.argv.pop(1)
runpy.run_module(module, run_name='__main__')
"""
    return subprocess.run(
        [sys.executable, "-c", script, module, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


@pytest.mark.parametrize("module", ["indirect", "train_grid"])
def test_cli_help_does_not_import_ml_dependencies(module):
    result = _run_cli(f"benchmark.examples.gaussian.approximators.{module}", "--help")
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout
    assert "--overwrite" in result.stdout


def test_single_preset_dry_run_uses_uniform_epochs_without_writing(tmp_path):
    output = tmp_path / "models"
    result = _run_cli(
        "benchmark.examples.gaussian.approximators.indirect",
        "--dry-run",
        "--preset",
        "40d_10n",
        "--output-dir",
        output,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    for model in MODELS:
        line = next(line for line in lines if f"{model}:" in line)
        assert "D=20, n=10, S=40" in line
        assert "epochs=100," in line
        assert f"{model}_s_40d_10n_bf_default.keras" in line
    assert not output.exists()


def test_grid_dry_run_covers_all_twenty_four_comparison_jobs_without_writing(tmp_path):
    output, logs = tmp_path / "models", tmp_path / "logs"
    result = _run_cli(
        "benchmark.examples.gaussian.approximators.train_grid",
        "--dry-run",
        "--output-dir",
        output,
        "--log-dir",
        logs,
    )
    assert result.returncode == 0, result.stderr
    jobs = [line for line in result.stdout.splitlines() if line.startswith("[train]")]
    commands = [line for line in result.stdout.splitlines() if "KERAS_BACKEND=" in line]
    assert len(jobs) == len(commands) == 24
    expected_paths = {
        str(output / f"{model}_s_{preset}_bf_default.keras")
        for model in MODELS
        for preset in PRESETS
    }
    assert {line.split(" -> ", 1)[1] for line in jobs} == expected_paths
    assert all("--epochs" not in command for command in commands)
    for model in MODELS:
        for preset in PRESETS:
            job = next(line for line in jobs if f"{model} / {preset}," in line)
            assert "epochs=100 ->" in job
    assert not output.exists()
    assert not logs.exists()


def test_grid_can_select_only_the_eight_missing_100_observation_jobs(tmp_path):
    output, logs = tmp_path / "models", tmp_path / "logs"
    result = _run_cli(
        "benchmark.examples.gaussian.approximators.train_grid",
        "--dry-run",
        "--presets",
        "20d_100n",
        "80d_100n",
        "--output-dir",
        output,
        "--log-dir",
        logs,
    )
    assert result.returncode == 0, result.stderr
    jobs = [line for line in result.stdout.splitlines() if line.startswith("[train]")]
    assert len(jobs) == 8
    assert {line.split(" -> ", 1)[1] for line in jobs} == {
        str(output / f"{model}_s_{preset}_bf_default.keras")
        for model in MODELS
        for preset in ("20d_100n", "80d_100n")
    }
    assert not output.exists()
    assert not logs.exists()


@pytest.mark.parametrize("module", ["indirect", "train_grid"])
@pytest.mark.parametrize(
    ("option", "tag"),
    [
        ("--summary-mmd", "20d_10n_bf_default_mmd"),
        ("--no-summary-mmd", "20d_10n_bf_default"),
    ],
)
def test_cli_summary_regularization_selects_distinct_archives(
    module, option, tag, tmp_path
):
    preset_flag = "--preset" if module == "indirect" else "--presets"
    result = _run_cli(
        f"benchmark.examples.gaussian.approximators.{module}",
        "--dry-run",
        "--models",
        "m1",
        preset_flag,
        "20d_10n",
        option,
        "--output-dir",
        tmp_path / "models",
    )
    assert result.returncode == 0, result.stderr
    assert f"m1_s_{tag}.keras" in result.stdout
    if module == "train_grid":
        command = next(
            line for line in result.stdout.splitlines() if "KERAS_BACKEND=" in line
        )
        assert option in command
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("module", ["indirect", "train_grid"])
def test_cli_rejects_conflicting_regularization_flags(module):
    result = _run_cli(
        f"benchmark.examples.gaussian.approximators.{module}",
        "--dry-run",
        "--summary-mmd",
        "--no-summary-mmd",
    )
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr


@pytest.mark.parametrize("module", ["indirect", "train_grid"])
def test_historical_archives_do_not_skip_default_network_training(module, tmp_path):
    historical = tmp_path / "m1_s_20d_10n.keras"
    historical.write_bytes(b"historical custom network")
    preset_flag = "--preset" if module == "indirect" else "--presets"
    result = _run_cli(
        f"benchmark.examples.gaussian.approximators.{module}",
        "--dry-run",
        "--models",
        "m1",
        preset_flag,
        "20d_10n",
        "--output-dir",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "skip" not in result.stdout.lower()
    assert "m1_s_20d_10n_bf_default.keras" in result.stdout
    assert historical.read_bytes() == b"historical custom network"
    assert list(tmp_path.iterdir()) == [historical]


def test_grid_forwards_overrides_and_deduplicates_jobs(tmp_path):
    result = _run_cli(
        "benchmark.examples.gaussian.approximators.train_grid",
        "--dry-run",
        "--models",
        "m1",
        "m3",
        "m1",
        "--presets",
        "20d_10n",
        "80d_10n",
        "20d_10n",
        "--epochs",
        "2",
        "--batch-size",
        "4",
        "--num-batches",
        "1",
        "--run-suffix",
        "trial",
        "--no-summary-mmd",
        "--output-dir",
        tmp_path / "models",
        "--log-dir",
        tmp_path / "logs",
    )
    assert result.returncode == 0, result.stderr
    jobs = [line for line in result.stdout.splitlines() if line.startswith("[train]")]
    commands = [line for line in result.stdout.splitlines() if "KERAS_BACKEND=" in line]
    assert len(jobs) == len(commands) == 4
    assert all("epochs=2 ->" in job and "_trial.keras" in job for job in jobs)
    for command in commands:
        for forwarded in (
            "--epochs 2",
            "--batch-size 4",
            "--num-batches 1",
            "--no-summary-mmd",
            "--run-suffix trial",
        ):
            assert forwarded in command
    assert not list(tmp_path.iterdir())


def _mock_networks(config):
    return {
        "summary_network": SimpleNamespace(
            get_config=lambda: {
                "summary_dim": config.summary_dim,
                "base_distribution": config.summary_base_distribution,
            }
        ),
        "inference_network": SimpleNamespace(
            get_config=lambda: {"test_flow_configuration": "default"}
        ),
    }


def test_training_saves_history_and_reuses_existing_model(
    indirect, tmp_path, monkeypatch
):
    config = TrainingConfig(epochs=2, batch_size=4, num_batches=1, run_suffix="test")
    history = SimpleNamespace(history={"loss": [np.float32(1.5), np.float32(0.5)]})
    fit_calls = []
    historical = tmp_path / "m1_s_20d_10n.keras"
    historical.write_bytes(b"historical custom network")

    class Approximator:
        def __init__(self):
            self.__dict__.update(_mock_networks(config))

        def save(self, path):
            Path(path).write_bytes(b"new model")

    approximator = Approximator()

    def fit_online(**kwargs):
        fit_calls.append(kwargs)
        return history

    workflow = SimpleNamespace(approximator=approximator, fit_online=fit_online)
    monkeypatch.setattr(indirect, "build_workflow", lambda *args: workflow)
    validation = {"mu": np.zeros((2, 20)), "x": np.zeros((2, 10, 20))}
    monkeypatch.setattr(indirect, "generate_validation_data", lambda *args: validation)
    saved, actual_history = indirect.train_approximator(
        "m1",
        config,
        output_dir=tmp_path,
        validation_size=2,
        validation_freq=2,
        verbose=0,
    )
    assert saved is approximator
    assert actual_history is history
    assert len(fit_calls) == 1
    assert fit_calls[0]["epochs"] == 2
    assert fit_calls[0]["batch_size"] == 4
    assert fit_calls[0]["num_batches_per_epoch"] == 1
    assert fit_calls[0]["validation_data"] is validation
    assert fit_calls[0]["validation_freq"] == 2
    assert indirect.load_history("m1", config, output_dir=tmp_path).history == {
        "loss": [1.5, 0.5]
    }
    path = model_path("m1", config, tmp_path)
    assert path.read_bytes() == b"new model"
    assert historical.read_bytes() == b"historical custom network"
    metadata = json.loads(path.with_suffix(".config.json").read_text())
    assert metadata["network_tag"] == "20d_10n_bf_default_test"
    assert metadata["training_config"]["epochs"] == 2
    assert metadata["training_config"]["summary_base_distribution"] is None
    assert metadata["flow"] == "default"
    assert metadata["summary_network"] == approximator.summary_network.get_config()
    assert metadata["inference_network"] == approximator.inference_network.get_config()

    loaded = object()
    monkeypatch.setattr(indirect, "load_approximator", lambda *args, **kwargs: loaded)
    reused, reused_history = indirect.train_approximator(
        "m1", config, output_dir=tmp_path
    )
    assert reused is loaded
    assert reused_history.history == {"loss": [1.5, 0.5]}
    assert len(fit_calls) == 1

    indirect.train_approximator("m1", config, output_dir=tmp_path, overwrite=True)
    assert len(fit_calls) == 2


def test_failed_save_preserves_existing_model_and_history(
    indirect, tmp_path, monkeypatch
):
    config = TrainingConfig(epochs=1)
    path = model_path("m1", config, tmp_path)
    path.write_bytes(b"original model")
    history_path = path.with_suffix(".history.json")
    history_path.write_text('{"loss": [10.0]}')

    def failed_save(path):
        Path(path).write_bytes(b"incomplete model")
        raise RuntimeError("save failed")

    workflow = SimpleNamespace(
        approximator=SimpleNamespace(save=failed_save, **_mock_networks(config)),
        fit_online=lambda **kwargs: SimpleNamespace(history={"loss": [1.0]}),
    )
    monkeypatch.setattr(indirect, "build_workflow", lambda *args: workflow)
    with pytest.raises(RuntimeError, match="save failed"):
        indirect.train_approximator("m1", config, output_dir=tmp_path, overwrite=True)
    assert path.read_bytes() == b"original model"
    assert json.loads(history_path.read_text()) == {"loss": [10.0]}
    assert not list(tmp_path.glob(".gaussian-training-*"))
