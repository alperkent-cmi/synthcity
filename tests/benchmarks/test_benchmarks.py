# stdlib
import hashlib
import json
import os
import platform
import shutil
import time
from copy import copy
from pathlib import Path
from typing import Any, Generator, List
from uuid import uuid4

# third party
import numpy as np
import pandas as pd
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_diabetes, load_iris

# synthcity absolute
from synthcity.benchmark import Benchmarks
from synthcity.benchmark.utils import augment_data, get_json_serializable_kwargs
from synthcity.metrics import Metrics
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    SurvivalAnalysisDataLoader,
    Syn_SeqDataLoader,
)
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from synthcity.plugins.generic.plugin_dummy_sampler import plugin as dummy_sampler_plugin


_WORKSPACE_MARKER = ".synthcity-owned-workspace"
_OWNED_WORKSPACES: dict[Path, str] = {}


def _repository_root() -> Path:
    """Find outer SynthData root, or nearest standalone SynthCity checkout."""
    test_path = Path(__file__).resolve()
    candidates = tuple(test_path.parents)
    synthdata_roots = [
        candidate
        for candidate in candidates
        if (candidate / "synthdata" / "__init__.py").is_file()
        and (candidate / "pyproject.toml").is_file()
    ]
    if synthdata_roots:
        return synthdata_roots[-1]
    checkout_roots = [
        candidate
        for candidate in candidates
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").is_file()
    ]
    if checkout_roots:
        return checkout_roots[0]
    raise RuntimeError(f"Could not identify checkout root for {test_path}")


def _new_owned_workspace() -> Path:
    """Create unique benchmark workspace below checkout-local scratch."""
    scratch_root = _repository_root() / "tmp"
    if scratch_root.is_symlink():
        raise RuntimeError(f"Refusing symlink scratch root: {scratch_root}")
    scratch_root = scratch_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    workspace = scratch_root / f"benchmark-{os.getpid()}-{time.time_ns()}-{uuid4().hex}"
    if workspace.is_symlink():
        raise RuntimeError(f"Refusing symlink workspace: {workspace}")
    workspace = workspace.resolve()
    if not workspace.is_relative_to(scratch_root):
        raise RuntimeError(f"Refusing workspace outside checkout scratch: {workspace}")
    workspace.mkdir(parents=False, exist_ok=False)
    token = uuid4().hex
    marker = workspace / _WORKSPACE_MARKER
    try:
        with marker.open("x", encoding="utf-8") as marker_file:
            marker_file.write(f"{token}\n")
    except OSError as exc:
        if workspace.is_dir() and not workspace.is_symlink() and not any(workspace.iterdir()):
            workspace.rmdir()
        else:
            raise RuntimeError(
                f"Workspace setup failed with ambiguous ownership state: {workspace}"
            ) from exc
        raise RuntimeError(f"Workspace marker setup failed: {workspace}") from exc
    _OWNED_WORKSPACES[workspace] = token
    return workspace


def _remove_owned_workspace(workspace: Path) -> None:
    """Remove only a workspace created by this module after containment checks."""
    if workspace.is_symlink():
        raise RuntimeError(f"Refusing cleanup of symlink workspace: {workspace}")
    scratch_root = _repository_root() / "tmp"
    if scratch_root.is_symlink():
        raise RuntimeError(f"Refusing symlink scratch root: {scratch_root}")
    scratch_root = scratch_root.resolve()
    resolved = workspace.resolve()
    marker = resolved / _WORKSPACE_MARKER
    token = _OWNED_WORKSPACES.get(resolved)
    if not resolved.is_relative_to(scratch_root):
        raise RuntimeError(f"Refusing cleanup outside checkout scratch: {resolved}")
    if (
        token is None
        or not resolved.is_dir()
        or resolved.is_symlink()
        or not marker.is_file()
        or marker.is_symlink()
    ):
        raise RuntimeError(f"Refusing cleanup of unowned workspace: {resolved}")
    if marker.read_text() != f"{token}\n":
        raise RuntimeError(f"Refusing cleanup with invalid ownership marker: {resolved}")
    shutil.rmtree(resolved)
    del _OWNED_WORKSPACES[resolved]


@pytest.fixture
def benchmark_workspace() -> Generator[Path, None, None]:
    """Provide fresh, checkout-local workspace and guarded cleanup."""
    workspace = _new_owned_workspace()
    try:
        yield workspace
    finally:
        _remove_owned_workspace(workspace)


def test_benchmark_fit_on_x_uses_explicit_loader(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    X = GenericDataLoader(data, target_column="target")
    X_test = GenericDataLoader(data.iloc[:2].copy(), target_column="target")
    captured = {}

    class FakeGenerator:
        def fit(self, fit_loader):
            captured["generator_fit_loader"] = fit_loader

        def generate(self, **kwargs):
            return X

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: FakeGenerator())

    def fake_metrics_evaluate(*args, **kwargs):
        captured["metrics_fit_loader"] = args[2]
        return {
            "mean": pd.Series({"sanity.common_rows_proportion": 0.5}),
            "errors": pd.Series({"sanity.common_rows_proportion": 0.0}),
            "durations": pd.Series({"sanity.common_rows_proportion": 0.0}),
            "direction": pd.Series({"sanity.common_rows_proportion": "maximize"}),
        }

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_metrics_evaluate))

    Benchmarks.evaluate(
        [("explicit_fit", "fake", {})],
        X,
        X_test=X_test,
        metrics={"sanity": ["common_rows_proportion"]},
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    assert captured["generator_fit_loader"] is X
    assert captured["metrics_fit_loader"] is X


def test_benchmark_candidate_screen_runs_before_metrics(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    X = GenericDataLoader(data, target_column="target")
    screened = []
    metrics_called = []

    class FakeGenerator:
        def fit(self, fit_loader):
            pass

        def generate(self, **kwargs):
            return X

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: FakeGenerator())
    monkeypatch.setattr(
        Metrics,
        "evaluate",
        staticmethod(
            lambda *args, **kwargs: (
                metrics_called.append(True)
                or {
                    "mean": pd.Series({"sanity.common_rows_proportion": 0.5}),
                    "errors": pd.Series({"sanity.common_rows_proportion": 0.0}),
                    "durations": pd.Series({"sanity.common_rows_proportion": 0.0}),
                    "direction": pd.Series({"sanity.common_rows_proportion": "maximize"}),
                }
            )
        ),
    )

    Benchmarks.evaluate(
        [("screened", "fake", {})],
        X,
        metrics={"sanity": ["common_rows_proportion"]},
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        candidate_screen=lambda candidate: screened.append(candidate.copy()),
    )

    assert len(screened) == 1
    assert metrics_called == [True]


def test_benchmark_forwards_classification_score_policy(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")
    captured = {}

    class FakeGenerator:
        def fit(self, _fit_loader):
            return self

        def generate(self, **_kwargs):
            return loader

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: FakeGenerator())

    def fake_metrics_evaluate(*args, **kwargs):
        captured["classification_score"] = kwargs["classification_score"]
        return {
            "mean": pd.Series({"sanity.common_rows_proportion": 0.5}),
            "errors": pd.Series({"sanity.common_rows_proportion": 0}),
            "durations": pd.Series({"sanity.common_rows_proportion": 0}),
            "direction": pd.Series({"sanity.common_rows_proportion": "maximize"}),
        }

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_metrics_evaluate))

    Benchmarks.evaluate(
        [("score_policy", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        classification_score="macro_f1",
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    assert captured["classification_score"] == "macro_f1"


def test_benchmark_forwards_and_retains_semantic_context(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")
    semantic_context = {
        "schema_version": "semantic-context-v1",
        "task_type": "classification",
    }
    captured = {}

    class FakeGenerator:
        def fit(self, _fit_loader):
            return self

        def generate(self, **_kwargs):
            return loader

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: FakeGenerator())

    def fake_metrics_evaluate(*args, **kwargs):
        del args
        captured["semantic_context"] = kwargs["semantic_context"]
        return {
            "mean": pd.Series({"sanity.common_rows_proportion": 0.5}),
            "errors": pd.Series({"sanity.common_rows_proportion": 0}),
            "durations": pd.Series({"sanity.common_rows_proportion": 0}),
            "direction": pd.Series({"sanity.common_rows_proportion": "maximize"}),
        }

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_metrics_evaluate))

    result = Benchmarks.evaluate(
        [("semantic", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        semantic_context=semantic_context,
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    assert captured["semantic_context"] == semantic_context
    assert result["semantic"].attrs["semantic_context"] == semantic_context


def test_benchmark_repeats_seeded_privacy_metadata(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")
    captured_seeds = []

    class FakeGenerator:
        def fit(self, _fit_loader):
            return self

        def generate(self, **_kwargs):
            return loader

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: FakeGenerator())

    def fake_metrics_evaluate(*args, **kwargs):
        del args
        seed = kwargs["random_state"]
        captured_seeds.append(seed)
        values = {
            "privacy.DomiasMIA_prior.aucroc": 0.2 + 0.1 * seed,
            "privacy.k-anonymization.syn": 2.0 + seed,
            "privacy.identifiability_score.score": 0.3 + 0.05 * seed,
        }
        report = pd.DataFrame(
            {
                "mean": list(values.values()),
                "errors": [0, 0, 0],
                "durations": [0.0, 0.0, 0.0],
                "direction": ["minimize", "maximize", "minimize"],
            },
            index=list(values),
        )
        report.attrs["result_metadata"] = {
            "privacy.DomiasMIA_prior": {
                "result_version": "domias-effective-auc-v2",
                "random_state": seed,
                "domias_protocol": {"auc_chance": 0.5},
            },
            "privacy.k-anonymization": {
                "result_version": "structural-proxy-v2",
                "random_state": seed,
                "proxy_label": "kmeans_partition_screen_not_formal_guarantee",
            },
            "privacy.identifiability_score": {
                "result_version": "identifiability-v2",
                "random_state": seed,
                "calibration_only": True,
            },
        }
        return report

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_metrics_evaluate))

    result = Benchmarks.evaluate(
        [("seeded_privacy", "fake", {})],
        loader,
        metrics={
            "privacy": [
                "DomiasMIA_prior",
                "k-anonymization",
                "identifiability_score",
            ]
        },
        repeats=3,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    report = result["seeded_privacy"]
    domias_metadata = report.attrs["result_metadata"]["privacy.DomiasMIA_prior"]
    assert sorted(captured_seeds) == [0, 1, 2]
    assert report.attrs["metric_metadata"] == report.attrs["result_metadata"]
    assert domias_metadata["random_state"] == 0
    assert report.loc["privacy.DomiasMIA_prior.aucroc", "stddev"] == np.std(
        [0.2, 0.3, 0.4]
    )
    assert domias_metadata["repeated_seed_uncertainty"]["seeds"] == [0, 1, 2]
    assert [
        entry["metadata"]["random_state"]
        for entry in domias_metadata["repeated_seed_metadata"]
    ] == [0, 1, 2]
    assert all(
        entry["metadata"]["domias_protocol"] == {"auc_chance": 0.5}
        for entry in domias_metadata["repeated_seed_metadata"]
    )
    assert report.attrs["result_metadata"]["privacy.k-anonymization"][
        "repeated_seed_metadata"
    ][0]["metadata"]["proxy_label"] == "kmeans_partition_screen_not_formal_guarantee"
    assert report.attrs["result_metadata"]["privacy.identifiability_score"][
        "repeated_seed_metadata"
    ][0]["metadata"]["calibration_only"] is True


def test_benchmark_grouped_unsupported_loader_short_circuits_before_generation(
    monkeypatch, tmp_path
) -> None:
    data = pd.DataFrame({"value": [0, 1, 2, 3], "target": [0, 1, 0, 1]})
    loader = Syn_SeqDataLoader(data, target_column="target", verbose=False)

    def unexpected_plugin_construction(*_args, **_kwargs):
        raise AssertionError("unsupported grouped benchmarks must not construct a plugin")

    monkeypatch.setattr(Plugins, "get", unexpected_plugin_construction)

    result = Benchmarks.evaluate(
        [("blocked_syn_seq", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        group_mode="patient_group",
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
    )

    report = result["blocked_syn_seq"]
    assert report.attrs["testcase"] == "blocked_syn_seq"
    assert report.attrs["group_safety"]["status"] == "group_unsafe"
    assert report.attrs["group_safety"]["group_mode"] == "patient_group"
    assert report.attrs["group_safety"]["loader_types"] == {"X": "syn_seq"}


@pytest.mark.parametrize("missing_loader", ["X", "X_test"])
def test_benchmark_grouped_missing_ids_fails_before_plugin_work(
    monkeypatch, tmp_path, missing_loader
) -> None:
    data = pd.DataFrame(
        {
            "value": [0, 1, 2, 3],
            "target": [0, 1, 0, 1],
        }
    )
    calls = {"construct": 0, "fit": 0, "generate": 0}

    class FakeGenerator:
        def fit(self, _fit_loader):
            calls["fit"] += 1

        def generate(self, **_kwargs):
            calls["generate"] += 1
            return GenericDataLoader(data, target_column="target")

    def fake_get(self, _name, **_kwargs):
        calls["construct"] += 1
        return FakeGenerator()

    monkeypatch.setattr(Plugins, "get", fake_get)
    X = GenericDataLoader(
        data,
        target_column="target",
        group_ids=None if missing_loader == "X" else ["train-a", "train-a", "train-b", "train-b"],
    )
    X_test = GenericDataLoader(
        data,
        target_column="target",
        group_ids=None
        if missing_loader == "X_test"
        else ["test-a", "test-a", "test-b", "test-b"],
    )

    with pytest.raises(ValueError, match=missing_loader):
        Benchmarks.evaluate(
            [("missing_groups", "fake", {})],
            X,
            X_test=X_test,
            metrics={"sanity": ["common_rows_proportion"]},
            group_mode="patient_group",
            repeats=1,
            workspace=tmp_path / "workspace",
            synthetic_cache=False,
            synthetic_reuse_if_exists=False,
            use_metric_cache=False,
            fit_on_X=True,
        )

    assert calls == {"construct": 0, "fit": 0, "generate": 0}


def test_benchmark_grouped_populations_keep_sizes_and_namespaces(
    monkeypatch, tmp_path
) -> None:
    train = pd.DataFrame(
        {
            "value": [0, 1, 2, 3, 4],
            "fairness": ["majority", "majority", "majority", "majority", "minority"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    tuning = pd.DataFrame(
        {
            "value": [5, 6, 7],
            "fairness": ["majority", "minority", "minority"],
            "target": [1, 0, 1],
        }
    )
    train_groups = ["train-a", "train-a", "train-b", "train-b", "train-c"]
    tuning_groups = ["tuning-a", "tuning-b", "tuning-b"]
    generated_lengths = {
        "synthetic": 4,
        "reference_synthetic": 2,
        "augmented": 1,
    }
    generation_namespaces = []
    captured = {}

    class FakeGenerator:
        def fit(self, fit_loader, **_kwargs):
            captured.setdefault("fit_loaders", []).append(fit_loader)

        def generate(self, count=None, **kwargs):
            del count
            namespace = kwargs["_group_namespace"]
            generation_namespaces.append(namespace)
            size = generated_lengths[namespace]
            frame = pd.DataFrame(
                {
                    "value": np.arange(size),
                    "fairness": ["minority"] * size,
                    "target": [index % 2 for index in range(size)],
                }
            )
            return GenericDataLoader(
                frame,
                target_column="target",
                fairness_column="fairness",
                group_ids=[
                    f"__synthcity_generated__{namespace}__{index}"
                    for index in range(size)
                ],
            )

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: FakeGenerator())

    def fake_metrics_evaluate(*args, **kwargs):
        captured["loaders"] = args
        captured["kwargs"] = kwargs
        return pd.DataFrame(
            {
                "mean": [0.5],
                "errors": [0.0],
                "durations": [0.0],
                "direction": ["maximize"],
            },
            index=["performance.fake_augmentation.aug_ood"],
        )

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_metrics_evaluate))

    result = Benchmarks.evaluate(
        [("grouped_sizes", "fake", {})],
        GenericDataLoader(
            train,
            target_column="target",
            fairness_column="fairness",
            group_ids=train_groups,
        ),
        X_test=GenericDataLoader(
            tuning,
            target_column="target",
            fairness_column="fairness",
            group_ids=tuning_groups,
        ),
        metrics={"performance": ["fake_augmentation"]},
        group_mode="patient_group",
        repeats=1,
        synthetic_size=4,
        augmentation_rule="ad-hoc",
        ad_hoc_augment_vals={"majority": 0, "minority": 1},
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        augmented_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    assert result["grouped_sizes"].loc[
        "performance.fake_augmentation.aug_ood", "mean"
    ] == pytest.approx(0.5)
    assert generation_namespaces == ["synthetic", "reference_synthetic", "augmented"]
    loaders = captured["loaders"]
    assert [len(loader) for loader in loaders] == [3, 4, 5, 2, 6]
    assert loaders[0].group_ids.tolist() == tuning_groups
    assert loaders[2].group_ids.tolist() == train_groups
    assert all(
        group_id.startswith("__synthcity_generated__synthetic__")
        for group_id in loaders[1].group_ids
    )
    assert all(
        group_id.startswith("__synthcity_generated__reference_synthetic__")
        for group_id in loaders[3].group_ids
    )
    assert loaders[4].group_ids[: len(train_groups)].tolist() == train_groups
    assert loaders[4].group_ids[len(train_groups) :].tolist() == [
        "__synthcity_generated__augmented__0"
    ]
    assert captured["kwargs"]["group_mode"] == "patient_group"


def test_benchmark_preserves_group_safety_from_metrics(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(
        data,
        target_column="target",
        group_ids=["patient-a", "patient-a", "patient-b", "patient-b", "patient-c", "patient-c"],
    )
    captured = {}

    class FakeGenerator:
        def fit(self, _fit_loader):
            return self

        def generate(self, **_kwargs):
            return loader

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: FakeGenerator())

    def fake_metrics_evaluate(*_args, **kwargs):
        captured["group_mode"] = kwargs["group_mode"]
        report = pd.DataFrame(
            {
                "mean": [float("nan")],
                "errors": [1],
                "durations": [0.0],
                "direction": ["maximize"],
            },
            index=["sanity.common_rows_proportion"],
        )
        report.attrs["group_safety"] = {
            "schema_version": "group-safety-v1",
            "status": "group_unsafe",
            "group_mode": "patient_group",
            "task_type": "classification",
            "reason": "test marker",
            "loader_types": {"X_gt": "generic", "X_syn": "generic"},
            "metrics": ["sanity.common_rows_proportion"],
        }
        return report

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(fake_metrics_evaluate))

    result = Benchmarks.evaluate(
        [("retained_safety", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        group_mode="patient_group",
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    assert captured["group_mode"] == "patient_group"
    assert result["retained_safety"].attrs["group_safety"]["status"] == "group_unsafe"


def test_benchmark_generator_fit_failure_returns_error_report(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")
    semantic_context = {
        "schema_version": "semantic-context-v1",
        "task_type": "classification",
    }

    class FailingGenerator:
        def fit(self, _fit_loader):
            raise RuntimeError("fit exploded")

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: FailingGenerator())

    result = Benchmarks.evaluate(
        [("failed_fit", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        semantic_context=semantic_context,
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    report = result["failed_fit"]
    assert report.loc["failed_fit", "status"] == "failed"
    assert report.loc["failed_fit", "stage"] == "fit"
    assert report.loc["failed_fit", "error_type"] == "RuntimeError"
    assert report.loc["failed_fit", "error"] == "fit exploded"
    assert report.attrs["semantic_context"] == semantic_context


def test_benchmark_reference_generation_failure_returns_error_report(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")
    empty_loader = GenericDataLoader(data.iloc[:0].copy(), target_column="target")
    generate_calls = []

    class GeneratorWithEmptyReference:
        def fit(self, _fit_loader):
            return self

        def generate(self, **_kwargs):
            generate_calls.append(True)
            return loader if len(generate_calls) == 1 else empty_loader

    monkeypatch.setattr(
        Plugins,
        "get",
        lambda self, name, **kwargs: GeneratorWithEmptyReference(),
    )

    def unexpected_metrics_evaluate(*_args, **_kwargs):
        raise AssertionError("metrics must not run after reference generation failure")

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(unexpected_metrics_evaluate))

    result = Benchmarks.evaluate(
        [("failed_reference", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    report = result["failed_reference"]
    assert len(generate_calls) == 2
    assert report.loc["failed_reference", "status"] == "failed"
    assert report.loc["failed_reference", "stage"] == "reference_generation"
    assert report.loc["failed_reference", "error_type"] == "RuntimeError"
    assert report.loc["failed_reference", "error"] == "Plugin failed to generate reference data"


def test_benchmark_synthetic_generation_failure_returns_error_report(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")
    empty_loader = GenericDataLoader(data.iloc[:0].copy(), target_column="target")

    class GeneratorWithEmptySynthetic:
        def fit(self, _fit_loader):
            return self

        def generate(self, **_kwargs):
            return empty_loader

    monkeypatch.setattr(
        Plugins,
        "get",
        lambda self, name, **kwargs: GeneratorWithEmptySynthetic(),
    )

    def unexpected_metrics_evaluate(*_args, **_kwargs):
        raise AssertionError("metrics must not run after synthetic generation failure")

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(unexpected_metrics_evaluate))

    result = Benchmarks.evaluate(
        [("failed_synthetic", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    report = result["failed_synthetic"]
    assert report.loc["failed_synthetic", "status"] == "failed"
    assert report.loc["failed_synthetic", "stage"] == "synthetic_generation"
    assert report.loc["failed_synthetic", "error_type"] == "RuntimeError"
    assert report.loc["failed_synthetic", "error"] == "Plugin failed to generate data"


def test_benchmark_metric_evaluation_failure_returns_error_report(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")

    class WorkingGenerator:
        def fit(self, _fit_loader):
            return self

        def generate(self, **_kwargs):
            return loader

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: WorkingGenerator())

    def failing_metrics_evaluate(*_args, **_kwargs):
        raise ValueError("metric evaluation exploded")

    monkeypatch.setattr(Metrics, "evaluate", staticmethod(failing_metrics_evaluate))

    result = Benchmarks.evaluate(
        [("failed_metrics", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    report = result["failed_metrics"]
    assert report.loc["failed_metrics", "status"] == "failed"
    assert report.loc["failed_metrics", "stage"] == "metric_evaluation"
    assert report.loc["failed_metrics", "error_type"] == "ValueError"
    assert report.loc["failed_metrics", "error"] == "metric evaluation exploded"


def test_benchmark_empty_metric_evaluation_returns_error_report(monkeypatch, tmp_path) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")

    class WorkingGenerator:
        def fit(self, _fit_loader):
            return self

        def generate(self, **_kwargs):
            return loader

    monkeypatch.setattr(Plugins, "get", lambda self, name, **kwargs: WorkingGenerator())
    monkeypatch.setattr(
        Metrics,
        "evaluate",
        staticmethod(lambda *_args, **_kwargs: pd.DataFrame()),
    )

    result = Benchmarks.evaluate(
        [("empty_metrics", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    report = result["empty_metrics"]
    assert report.loc["empty_metrics", "status"] == "failed"
    assert report.loc["empty_metrics", "stage"] == "metric_evaluation"
    assert report.loc["empty_metrics", "error_type"] == "RuntimeError"
    assert report.loc["empty_metrics", "error"] == "Metrics.evaluate returned no metric rows"


def test_benchmark_generator_construction_failure_returns_error_report(
    monkeypatch, tmp_path
) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")

    def fail_construction(self, name, **kwargs):
        del self, name, kwargs
        raise ValueError("construction exploded")

    monkeypatch.setattr(Plugins, "get", fail_construction)

    result = Benchmarks.evaluate(
        [("failed_construction", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    report = result["failed_construction"]
    assert report.loc["failed_construction", "status"] == "failed"
    assert report.loc["failed_construction", "stage"] == "construction"
    assert report.loc["failed_construction", "error_type"] == "ValueError"
    assert report.loc["failed_construction", "error"] == "construction exploded"


def test_benchmark_report_retains_generator_accounting_metadata(
    monkeypatch, tmp_path
) -> None:
    data = load_diabetes(as_frame=True).frame.head(6).copy()
    data["target"] = data.pop("target")
    loader = GenericDataLoader(data, target_column="target")

    class GeneratorWithAccounting:
        def fit(self, _fit_loader):
            return self

        def get_accounting_metadata(self):
            return {"effective_epsilon": 1.25}

        def fqdn(self):
            return "fake"

        def generate(self, **_kwargs):
            return loader

    monkeypatch.setattr(
        Plugins, "get", lambda self, name, **kwargs: GeneratorWithAccounting()
    )
    monkeypatch.setattr(
        Metrics,
        "evaluate",
        staticmethod(
            lambda *args, **kwargs: {
                "mean": pd.Series({"sanity.common_rows_proportion": 0.5}),
                "errors": pd.Series({"sanity.common_rows_proportion": 0}),
                "durations": pd.Series({"sanity.common_rows_proportion": 0}),
                "direction": pd.Series({"sanity.common_rows_proportion": "maximize"}),
            }
        ),
    )

    result = Benchmarks.evaluate(
        [("accounted", "fake", {})],
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        repeats=1,
        workspace=tmp_path / "workspace",
        synthetic_cache=False,
        synthetic_reuse_if_exists=False,
        use_metric_cache=False,
        fit_on_X=True,
    )

    metadata = result["accounted"].attrs["metric_metadata"]["generator.fake"]
    assert metadata["schema_version"] == "generator-runtime-metadata-v2"
    assert metadata["plugin_fqdn"] == "fake"
    assert metadata["n_samples"] == len(data)
    assert metadata["random_state"] == 0
    assert metadata["privacy_claim_type"] == "none"
    assert metadata["accounting"] == {"effective_epsilon": 1.25}


def test_benchmark_sanity(benchmark_workspace: Path) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    scores = Benchmarks.evaluate(
        [
            ("test1", "marginal_distributions", {}),
            ("test2", "dummy_sampler", {}),
        ],
        GenericDataLoader(X, sensitive_columns=["sex"]),
        metrics={"sanity": ["common_rows_proportion", "data_mismatch_score"]},
        workspace=benchmark_workspace,
    )

    Benchmarks.print(scores)


def test_benchmark_augmentation(benchmark_workspace: Path) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    scores = Benchmarks.evaluate(
        [
            ("test1", "marginal_distributions", {}),
            ("test2", "dummy_sampler", {}),
        ],
        GenericDataLoader(X, fairness_column="sex"),
        metrics={
            "performance": [
                "linear_model_augmentation",
                "mlp_augmentation",
                "xgb_augmentation",
            ]
        },
        workspace=benchmark_workspace,
    )

    Benchmarks.print(scores)


def test_augment_data_extends_group_ids_for_generated_rows() -> None:
    frame = pd.DataFrame(
        {
            "value": [0, 1, 2],
            "fairness": ["majority", "majority", "minority"],
            "target": [0, 1, 0],
        }
    )
    training_groups = ["patient-a", "patient-a", "patient-b"]
    loader = GenericDataLoader(
        frame,
        target_column="target",
        fairness_column="fairness",
        group_ids=training_groups,
    )
    generator = dummy_sampler_plugin()
    generator.fit(loader, cond=loader["fairness"])

    augmented = augment_data(
        loader,
        generator,
        rule="ad-hoc",
        ad_hoc_augment_vals={"majority": 0, "minority": 1},
    )

    assert len(augmented) == len(frame) + 1
    assert len(augmented.group_ids) == len(augmented)
    assert augmented.group_ids[: len(frame)].tolist() == training_groups
    assert augmented.group_ids[-1].startswith("__synthcity_generated__augmented__")


def test_benchmark_invalid_plugin(benchmark_workspace: Path) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                ("test1", "invalid", {}),
                ("test2", "dummy_sampler", {}),
            ],
            GenericDataLoader(X, sensitive_columns=["sex"]),
            metrics={"sanity": ["common_rows_proportion", "data_mismatch_score"]},
            workspace=benchmark_workspace,
        )


def test_benchmark_invalid_metric(benchmark_workspace: Path) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    score = Benchmarks.evaluate(
        [
            ("test2", "uniform_sampler", {}),
        ],
        GenericDataLoader(X, sensitive_columns=["sex"]),
        metrics={"sanity": ["invalid"]},
        workspace=benchmark_workspace,
    )
    assert len(score["test2"]) == 0


def test_benchmark_custom_target(benchmark_workspace: Path) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    Benchmarks.evaluate(
        [
            ("test2", "ctgan", {}),
        ],
        GenericDataLoader(X, target_column="target"),
        metrics={
            "performance": [
                "linear_model",
            ]
        },
        task_type="regression",
        workspace=benchmark_workspace,
    )


def test_benchmark_survival_analysis(benchmark_workspace: Path) -> None:
    df = load_rossi()

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                ("test2", "uniform_sampler", {}),
            ],
            SurvivalAnalysisDataLoader(
                df, target_column=None, time_to_event_column="week", time_horizons=[30]
            ),
            task_type="survival_analysis",
            metrics={
                "performance": [
                    "linear_model",
                ]
            },
            workspace=benchmark_workspace,
        )

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                ("test2", "uniform_sampler", {}),
            ],
            SurvivalAnalysisDataLoader(
                df,
                target_column="arrest",
                time_to_event_column=None,
                time_horizons=[30],
            ),
            task_type="survival_analysis",
            metrics={
                "performance": [
                    "linear_model",
                ]
            },
            workspace=benchmark_workspace,
        )

    with pytest.raises(ValueError):
        Benchmarks.evaluate(
            [
                ("test1", "uniform_sampler", {}),
            ],
            SurvivalAnalysisDataLoader(
                df,
                target_column="arrest",
                time_to_event_column="week",
                time_horizons=None,
            ),
            task_type="survival_analysis",
            metrics={
                "performance": [
                    "linear_model",
                ]
            },
            workspace=benchmark_workspace,
        )

    score = Benchmarks.evaluate(
        [
            ("test1", "uniform_sampler", {}),
        ],
        SurvivalAnalysisDataLoader(
            df, target_column="arrest", time_to_event_column="week", time_horizons=[30]
        ),
        task_type="survival_analysis",
        metrics={
            "performance": [
                "linear_model",
            ]
        },
        workspace=benchmark_workspace,
    )
    Benchmarks.print(score)

    score = Benchmarks.evaluate(
        [
            ("test1", "marginal_distributions", {}),
            ("test2", "dummy_sampler", {}),
        ],
        SurvivalAnalysisDataLoader(
            df,
            target_column="arrest",
            fairness_column="age",
            time_to_event_column="week",
            time_horizons=[30],
        ),
        task_type="survival_analysis",
        metrics={
            "performance": [
                "linear_model",
                "linear_model_augmentation",
            ]
        },
        workspace=benchmark_workspace,
    )
    Benchmarks.print(score)


def test_benchmark_workspace_cache(benchmark_workspace: Path) -> None:
    df = load_rossi()

    workspace = benchmark_workspace

    X = SurvivalAnalysisDataLoader(
        df,
        target_column="arrest",
        fairness_column="age",
        time_to_event_column="week",
        time_horizons=[30],
    )

    testcase = "test1"
    plugin = "uniform_sampler"
    kwargs = {"workspace": workspace}

    kwargs_hash = ""
    if len(kwargs) > 0:
        serializable_kwargs = get_json_serializable_kwargs(kwargs)
        kwargs_hash_raw = json.dumps(serializable_kwargs, sort_keys=True).encode()
        hash_object = hashlib.sha256(kwargs_hash_raw)
        kwargs_hash = hash_object.hexdigest()

    augmentation_arguments = {
        "augmentation_rule": "equal",
        "strict_augmentation": False,
        "ad_hoc_augment_vals": None,
    }
    augmentation_arguments_hash_raw = json.dumps(
        copy(augmentation_arguments), sort_keys=True
    ).encode()
    augmentation_hash_object = hashlib.sha256(augmentation_arguments_hash_raw)
    augmentation_hash = augmentation_hash_object.hexdigest()

    experiment_name = X.hash()
    repeats = 3

    Benchmarks.evaluate(
        [
            (testcase, plugin, kwargs),
        ],
        X,
        task_type="survival_analysis",
        metrics={
            "performance": [
                "linear_model_augmentation",
            ]
        },
        repeats=repeats,
        workspace=workspace,
        augmented_reuse_if_exists=False,
        synthetic_reuse_if_exists=False,
    )

    assert workspace.exists()

    for repeat in range(repeats):
        X_syn_cache_file = (
            workspace
            / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
        )
        generator_file = (
            workspace
            / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
        )
        X_augment_cache_file = (
            workspace
            / f"{experiment_name}_{testcase}_{plugin}_augmentation_{augmentation_hash}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
        )

        augment_generator_file = (
            workspace
            / f"{experiment_name}_{testcase}_{plugin}_augmentation_{augmentation_hash}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
        )

        assert X_syn_cache_file.exists()
        assert generator_file.exists()

        assert X_augment_cache_file.exists()
        assert augment_generator_file.exists()


def test_benchmark_added_plugin(benchmark_workspace: Path) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    class DummyCopyDataPlugin(Plugin):
        """Dummy plugin for debugging."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

        @staticmethod
        def name() -> str:
            return "copy_data"

        @staticmethod
        def type() -> str:
            return "debug"

        @staticmethod
        def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
            return []

        def _fit(
            self, X: DataLoader, *args: Any, **kwargs: Any
        ) -> "DummyCopyDataPlugin":
            self.features_count = X.shape[1]
            self.X = X
            return self

        def _generate(
            self, count: int, syn_schema: Schema, **kwargs: Any
        ) -> DataLoader:
            return self.X.sample(count)

    generators = Plugins()
    # Add the new plugin to the collection
    generators.add("copy_data", DummyCopyDataPlugin)

    score = Benchmarks.evaluate(
        [
            ("copy_data", "copy_data", {}),
        ],
        GenericDataLoader(X, target_column="target"),
        metrics={
            "performance": [
                "linear_model",
            ]
        },
        workspace=benchmark_workspace,
    )
    assert "copy_data" in score
