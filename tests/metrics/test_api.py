# third party
import logging
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
import torch
from torch.utils.data import TensorDataset

# synthcity absolute
from synthcity.benchmark.utils import augment_data
from synthcity.metrics import Metrics, WeightedMetrics
from synthcity.metrics import eval as eval_module
from synthcity.metrics.eval_detection import _detection_cv_splits
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    ImageDataLoader,
    Syn_SeqDataLoader,
    TimeSeriesDataLoader,
    TimeSeriesSurvivalDataLoader,
)


@pytest.mark.parametrize(
    ("phase", "failure_point"),
    [
        ("group_task_checks", "group_task_checks"),
        ("loader_validation_conversion", "loader_conversion"),
        ("group_loader_metadata_resolution", "group_loader_metadata_resolution"),
        ("encoder_fit_transform", "encoder_fit_transform"),
        ("evaluator_construction_queue", "evaluator_construction_queue"),
        ("score_compute_entry", "score_compute_entry"),
    ],
)
def test_metrics_pre_callback_failures_log_safe_phase(
    monkeypatch, caplog, tmp_path, phase: str, failure_point: str
) -> None:
    from synthcity.metrics.eval_sanity import CommonRowsProportion
    from synthcity.plugins.core.dataloader import GenericDataLoader

    sensitive_literal = "patient/HMAC_private-marker"
    exception_message = f"unknown category {sensitive_literal}; /private/traceback.py:7"

    def fail_with_sensitive_detail(*_args, **_kwargs):
        raise ValueError(exception_message)

    evidence = pd.DataFrame({"feature": [0, 1], "target": [0, 1]})
    synthetic = evidence.copy()
    train = evidence.copy()
    metric_kwargs = {}
    if failure_point == "group_task_checks":
        metric_kwargs["task_type"] = f"unsupported-{sensitive_literal}"
    elif failure_point == "loader_conversion":
        monkeypatch.setattr(GenericDataLoader, "__init__", fail_with_sensitive_detail)
    elif failure_point == "group_loader_metadata_resolution":
        evidence = GenericDataLoader(evidence)
        synthetic = GenericDataLoader(synthetic)
        train = GenericDataLoader(train)
        monkeypatch.setattr(
            GenericDataLoader,
            "group_ids",
            property(fail_with_sensitive_detail),
            raising=False,
        )
    elif failure_point == "encoder_fit_transform":
        evidence = pd.DataFrame(
            {"category": [sensitive_literal, "known"], "target": [0, 1]}
        )
        synthetic = pd.DataFrame({"category": ["known", "known"], "target": [0, 1]})
        train = synthetic.copy()
    elif failure_point == "evaluator_construction_queue":
        monkeypatch.setattr(
            CommonRowsProportion, "__init__", fail_with_sensitive_detail
        )
    else:
        monkeypatch.setattr(
            eval_module.ScoreEvaluator, "compute", fail_with_sensitive_detail
        )

    callbacks = []
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ValueError):
        Metrics.evaluate(
            evidence,
            synthetic,
            train,
            metrics={"sanity": ["common_rows_proportion"]},
            workspace=tmp_path,
            use_cache=False,
            progress_model="model-debug-test",
            progress_role="tuning",
            progress_callback=callbacks.append,
            **metric_kwargs,
        )

    output = caplog.text
    assert f"phase start model=model-debug-test role=tuning phase={phase}" in output
    assert f"phase failed model=model-debug-test role=tuning phase={phase}" in output
    start_markers = [
        line
        for line in output.splitlines()
        if "phase start model=model-debug-test role=tuning phase=" in line
    ]
    assert start_markers[-1].split("phase=")[-1].split()[0] == phase
    assert "exception_type=ValueError reason_code=phase_failed" in output
    assert exception_message not in output
    assert sensitive_literal not in output
    assert "/private/traceback.py" not in output
    assert callbacks == []


@pytest.mark.parametrize("test_plugin", ["dummy_sampler", "marginal_distributions"])
def test_basic(test_plugin: str) -> None:
    model = Plugins().get(test_plugin)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    Xraw = GenericDataLoader(X, target_column="target")

    model.fit(Xraw)
    X_gen = model.generate(100)

    out = Metrics.evaluate(
        Xraw,
        X_gen,
        metrics={"sanity": ["common_rows_proportion"]},
    )

    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(
        [
            "mean",
            "min",
            "max",
            "median",
            "iqr",
            "stddev",
            "rounds",
            "durations",
            "errors",
            "error_types",
            "error_messages",
            "direction",
        ]
    )


def test_grouped_detection_cv_never_splits_a_group() -> None:
    real = pd.DataFrame(
        {"value": np.arange(8), "target": [0, 0, 1, 1, 0, 0, 1, 1]}
    )
    synthetic = pd.DataFrame(
        {"value": np.arange(8, 16), "target": [0, 0, 1, 1, 0, 0, 1, 1]}
    )
    real_groups = ["real-1", "real-1", "real-2", "real-2", "real-3", "real-3", "real-4", "real-4"]
    synthetic_groups = [
        ("synthetic", index) for index in range(len(synthetic))
    ]
    real_loader = GenericDataLoader(real, target_column="target", group_ids=real_groups)
    synthetic_loader = GenericDataLoader(
        synthetic, target_column="target", group_ids=synthetic_groups
    )
    data = np.concatenate([real_loader.numpy(), synthetic_loader.numpy()])
    labels = np.concatenate(
        [np.zeros(len(real_loader), dtype=int), np.ones(len(synthetic_loader), dtype=int)]
    )
    groups = real_groups + synthetic_groups

    for train_idx, test_idx in _detection_cv_splits(
        real_loader, synthetic_loader, data, labels, n_folds=2, random_state=7
    ):
        assert set(groups[index] for index in train_idx).isdisjoint(
            groups[index] for index in test_idx
        )


def test_grouped_non_tabular_evaluation_materializes_group_unsafe_status() -> None:
    frame = pd.DataFrame({"value": np.arange(4), "target": [0, 1, 0, 1]})
    groups = ["patient-a", "patient-a", "patient-b", "patient-b"]
    loader = GenericDataLoader(frame, target_column="target", group_ids=groups)

    report = Metrics.evaluate(
        loader,
        loader,
        task_type="time_series",
        group_mode="patient_group",
        metrics={"sanity": ["common_rows_proportion"]},
        use_cache=False,
    )

    assert list(report.index) == ["sanity.common_rows_proportion"]
    assert report.loc["sanity.common_rows_proportion", "error_types"] == "GroupUnsafe"
    assert report.attrs["group_safety"]["status"] == "group_unsafe"


def test_grouped_syn_seq_evaluation_blocks_before_loader_reconstruction() -> None:
    frame = pd.DataFrame({"value": np.arange(4), "target": [0, 1, 0, 1]})
    loader = Syn_SeqDataLoader(frame, target_column="target", verbose=False)

    report = Metrics.evaluate(
        loader,
        loader,
        task_type="classification",
        group_mode="patient_group",
        metrics={"sanity": ["common_rows_proportion"]},
        use_cache=False,
    )

    assert list(report.index) == ["sanity.common_rows_proportion"]
    assert report.loc["sanity.common_rows_proportion", "error_types"] == "GroupUnsafe"
    assert report.attrs["group_safety"] == {
        "schema_version": "group-safety-v1",
        "status": "group_unsafe",
        "group_mode": "patient_group",
        "task_type": "classification",
        "reason": (
            "Patient-group SynthCity evaluation is not supported for loader type(s) "
            "['syn_seq']; metric internals require a group-aware implementation"
        ),
        "loader_types": {"X_gt": "syn_seq", "X_syn": "syn_seq"},
        "metrics": ["sanity.common_rows_proportion"],
    }


def test_patient_group_evaluation_without_group_ids_is_blocked() -> None:
    frame = pd.DataFrame({"value": np.arange(4), "target": [0, 1, 0, 1]})
    loader = GenericDataLoader(frame, target_column="target")

    report = Metrics.evaluate(
        loader,
        loader,
        group_mode="patient_group",
        metrics={"sanity": ["common_rows_proportion"]},
        use_cache=False,
    )

    safety = report.attrs["group_safety"]
    assert safety["status"] == "group_unsafe"
    assert "aligned group IDs" in safety["reason"]
    assert safety["loader_types"] == {"X_gt": "generic", "X_syn": "generic"}


@pytest.mark.parametrize(
    "loader_type",
    ["syn_seq", "images", "time_series", "time_series_survival"],
)
def test_grouped_unsupported_modalities_materialize_group_unsafe_status(loader_type: str) -> None:
    if loader_type == "syn_seq":
        frame = pd.DataFrame({"value": np.arange(4), "target": [0, 1, 0, 1]})
        loader = Syn_SeqDataLoader(frame, target_column="target", verbose=False)
    elif loader_type == "images":
        loader = ImageDataLoader(
            TensorDataset(torch.zeros((4, 1, 4, 4)), torch.tensor([0, 1, 0, 1]))
        )
    elif loader_type == "time_series":
        loader = TimeSeriesDataLoader(
            temporal_data=[pd.DataFrame({"value": [0.0, 1.0]}) for _ in range(4)],
            observation_times=[[0.0, 1.0] for _ in range(4)],
        )
    else:
        loader = TimeSeriesSurvivalDataLoader(
            temporal_data=[pd.DataFrame({"value": [0.0, 1.0]}) for _ in range(4)],
            observation_times=[[0.0, 1.0] for _ in range(4)],
            T=np.array([1.0, 2.0, 3.0, 4.0]),
            E=np.array([0, 1, 0, 1]),
        )

    report = Metrics.evaluate(
        loader,
        loader,
        group_mode="patient_group",
        metrics={"sanity": ["common_rows_proportion"]},
        use_cache=False,
    )

    assert report.attrs["group_safety"]["status"] == "group_unsafe"
    assert report.attrs["group_safety"]["group_mode"] == "patient_group"
    assert report.attrs["group_safety"]["loader_types"] == {
        "X_gt": loader_type,
        "X_syn": loader_type,
    }
    assert "group-aware implementation" in report.attrs["group_safety"]["reason"]


def test_metrics_preserves_semantic_context_on_reports() -> None:
    frame = pd.DataFrame({"value": np.arange(4), "target": [0, 1, 0, 1]})
    loader = GenericDataLoader(frame, target_column="target")
    semantic_context = {
        "schema_version": "semantic-context-v1",
        "task_type": "classification",
    }

    report = Metrics.evaluate(
        loader,
        loader,
        metrics={"sanity": ["common_rows_proportion"]},
        semantic_context=semantic_context,
        use_cache=False,
    )

    assert report.attrs["semantic_context"] == semantic_context


def test_metrics_fit_encoders_on_real_fit_loader_only(tmp_path) -> None:
    train = pd.DataFrame(
        {"category": ["known", "known", "known", "known"], "target": [0, 1, 0, 1]}
    )
    evidence = pd.DataFrame(
        {"category": ["known", "known", "known", "known"], "target": [0, 1, 0, 1]}
    )
    synthetic = pd.DataFrame(
        {"category": ["known", "known", "known", "known"], "target": [0, 1, 0, 1]}
    )
    train_loader = GenericDataLoader(train, target_column="target")
    evidence_loader = GenericDataLoader(evidence, target_column="target")
    synthetic_loader = GenericDataLoader(synthetic, target_column="target")

    out = Metrics.evaluate(
        evidence_loader,
        synthetic_loader,
        train_loader,
        metrics={"sanity": ["common_rows_proportion"]},
        workspace=tmp_path,
        use_cache=False,
    )

    assert not out.empty


def test_metrics_reject_unseen_evidence_category_after_train_fit(tmp_path) -> None:
    train = pd.DataFrame(
        {"category": ["known", "known", "known", "known"], "target": [0, 1, 0, 1]}
    )
    evidence = pd.DataFrame(
        {"category": ["unseen", "known", "known", "known"], "target": [0, 1, 0, 1]}
    )
    synthetic = pd.DataFrame(
        {"category": ["known", "known", "known", "known"], "target": [0, 1, 0, 1]}
    )

    with pytest.raises(ValueError, match="previously unseen labels"):
        Metrics.evaluate(
            GenericDataLoader(evidence, target_column="target"),
            GenericDataLoader(synthetic, target_column="target"),
            GenericDataLoader(train, target_column="target"),
            metrics={"sanity": ["common_rows_proportion"]},
            workspace=tmp_path,
            use_cache=False,
        )


def test_list() -> None:
    assert set(Metrics.list().keys()) == set(
        [
            "privacy",
            "stats",
            "sanity",
            "detection",
            "performance",
            "attack",
        ]
    )
    assert set(Metrics.list()["attack"]) == {
        "data_leakage_mlp",
        "data_leakage_xgb",
        "data_leakage_linear",
    }
    assert "detection_gmm" not in set(Metrics.list()["detection"])


def test_attack_metric_filter(tmp_path) -> None:
    X = pd.DataFrame(
        {
            "feature": list(range(20)),
            "sex": [0, 1] * 10,
            "target": [0, 1] * 10,
        }
    )

    Xraw = GenericDataLoader(X, target_column="target", sensitive_features=["sex"])

    out = Metrics.evaluate(
        Xraw,
        Xraw,
        metrics={"attack": ["data_leakage_linear"]},
        sensitive_target_types={"sex": "categorical"},
        workspace=tmp_path,
        use_cache=False,
    )

    assert any(index.startswith("attack.data_leakage_linear") for index in out.index)


def test_structural_privacy_metadata_survives_metrics_table() -> None:
    frame = pd.DataFrame(
        {
            "quasi_id": list(range(10)),
            "secret": [0, 1] * 5,
        }
    )
    loader = GenericDataLoader(frame, sensitive_features=["secret"])

    out = Metrics.evaluate(
        loader,
        loader,
        metrics={"privacy": ["k-anonymization"]},
        structural_n_clusters=[2],
        structural_min_rows_per_cluster=2,
        use_cache=False,
    )

    metadata = out.attrs["metric_metadata"]["privacy.k-anonymization"]
    assert metadata["result_version"] == "structural-proxy-v2"
    assert metadata["feature_selection"]["real_columns"] == ["quasi_id"]
    assert metadata["kmeans"]["n_clusters"] == [2]


@pytest.mark.parametrize(
    "metric_filter",
    [
        {"sanity": ["data_mismatch", "common_rows_proportion"]},
    ],
)
def test_metric_filter(metric_filter: dict) -> None:
    model = Plugins().get("marginal_distributions")

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    X["target"] = X["target"].astype(str)

    Xraw = GenericDataLoader(X, target_column="target")

    model.fit(Xraw)
    X_gen = model.generate(100)
    assert not X_gen.dataframe().empty
    print(X_gen)

    # Add debugging here
    print(f"Metrics to evaluate: {metric_filter}")
    print(
        f"Xraw shape: {Xraw.dataframe().shape}, X_gen shape: {X_gen.dataframe().shape}"
    )

    out = Metrics.evaluate(
        Xraw,
        X_gen,
        metrics=metric_filter,
    )

    print(f"Output of Metrics.evaluate: {out}")

    expected_index = [
        f"{category}.{metric}.score"
        for category in metric_filter
        for metric in metric_filter[category]
    ]
    assert set(list(out.index)) == set(expected_index)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == set(
        [
            "mean",
            "min",
            "max",
            "median",
            "iqr",
            "stddev",
            "rounds",
            "durations",
            "errors",
            "error_types",
            "error_messages",
            "direction",
        ]
    )


@pytest.mark.parametrize(
    "target",
    [
        None,
        "target",
        "sepal width (cm)",
    ],
)
def test_custom_label(target: str) -> None:
    model = Plugins().get("marginal_distributions")

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    X["target"] = X["target"].astype(str)

    Xraw = GenericDataLoader(X, target_column="target")

    model.fit(Xraw)
    X_gen = model.generate(100)

    out = Metrics.evaluate(Xraw, X_gen, metrics={"performance": "linear_model"})

    assert "performance.linear_model.syn_id" in out.index
    assert "performance.linear_model.syn_ood" in out.index


@pytest.mark.parametrize("test_plugin", ["dummy_sampler"])
def test_weighted_metric(test_plugin: str) -> None:
    model = Plugins().get(test_plugin)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    Xraw = GenericDataLoader(X, target_column="target")

    model.fit(Xraw)
    X_gen = model.generate(100)

    score = WeightedMetrics(
        [
            ("sanity", "common_rows_proportion"),
            ("sanity", "data_mismatch"),
        ],
        weights=[0.5, 0.5],
    ).evaluate(
        Xraw,
        X_gen,
    )

    assert isinstance(score, (float, int))

    # invalid metric
    with pytest.raises(ValueError):
        WeightedMetrics([("sanity", "fake")], [1])

    # invalid weights
    with pytest.raises(ValueError):
        WeightedMetrics([("sanity", "common_rows_proportion")], [0.5, 0.5])

    # different direction
    with pytest.raises(ValueError):
        WeightedMetrics(
            [("sanity", "common_rows_proportion"), ("performance", "xgb")], [0.5, 0.5]
        )


@pytest.mark.parametrize(
    "fairness_column, rule, strict, ad_hoc_vals",
    [
        ("sepal length (cm)", "equal", True, {}),
        ("sepal length (cm)", "equal", False, {}),
        ("sepal length (cm)", "log", True, {}),
        ("sepal length (cm)", "log", False, {}),
        (
            "sepal length (cm)",
            "ad-hoc",
            True,
            {k: 10 for k in [4.6, 5.0, 5.4, 4.4, 4.8, 7.4, 7.9]},
        ),
        (
            "sepal length (cm)",
            "ad-hoc",
            False,
            {k: 10 for k in [4.6, 5.0, 5.4, 4.4, 4.8, 7.4, 7.9]},
        ),
        pytest.param(
            "sepal length (cm)",
            "ad-hoc",
            True,
            {k: 10 for k in [-1, 10000]},
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_augmentation(
    fairness_column: str, rule: str, strict: bool, ad_hoc_vals: dict
) -> None:
    augment_generator = Plugins().get("marginal_distributions")

    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    X["target"] = X["target"].astype(str)

    Xraw = GenericDataLoader(X, target_column="target", fairness_column=fairness_column)

    augment_generator.fit(Xraw, cond=Xraw[fairness_column])

    X_augmented = augment_data(
        Xraw,
        augment_generator,
        rule=rule,
        strict=strict,
        ad_hoc_augment_vals=ad_hoc_vals,
    )
    assert len(Xraw) < len(X_augmented)

    X_gen = augment_generator.generate(100)

    out = Metrics.evaluate(
        Xraw, X_gen, X_augmented, metrics={"performance": "linear_model_augmentation"}
    )
    assert "performance.linear_model_augmentation.aug_ood" in out.index
