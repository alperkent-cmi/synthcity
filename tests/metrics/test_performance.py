# stdlib
from pathlib import Path
import sys
from typing import Optional, Type

# third party
import numpy as np
import pandas as pd
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_diabetes, load_iris
from torchvision import datasets

# synthcity absolute
from synthcity.metrics.eval_performance import (
    FeatureImportanceRankDistance,
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    ImageDataLoader,
    SurvivalAnalysisDataLoader,
    TimeSeriesDataLoader,
    TimeSeriesSurvivalDataLoader,
    create_from_info,
)
from synthcity.plugins.core.models.survival_analysis import (
    benchmarks as survival_benchmarks,
)
from synthcity.plugins.core.models.time_series_survival.benchmarks import (
    evaluate_ts_classification,
    evaluate_ts_survival_model,
)
from synthcity.plugins.core.models.ts_model import TimeSeriesModel
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
from synthcity.utils.datasets.time_series.pbc import PBCDataloader
from synthcity.utils.evaluation import cross_validation_splits


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_classifier(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X, target_column="target")
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t(use_cache=False)
    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    assert good_score["gt"] > 0
    assert good_score["syn_id"] > 0
    assert good_score["syn_ood"] > 0

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        Xloader,
        GenericDataLoader(X_rnd),
    )

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] < good_score["syn_id"]
    assert score["syn_ood"] < good_score["syn_ood"]

    assert evaluator.type() == "performance"
    assert evaluator.direction() == "maximize"

    def_score = evaluator.evaluate_default(Xloader, GenericDataLoader(X_rnd))

    assert def_score == score["syn_id"]


@pytest.mark.parametrize("distance", ["kendall", "spearman"])
@pytest.mark.parametrize(
    "test_plugin",
    [
        Plugins().get("ctgan"),
    ],
)
@pytest.mark.xfail
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
@pytest.mark.slow_1
@pytest.mark.slow
def test_evaluate_feature_importance_rank_dist_clf(
    distance: str, test_plugin: Plugin
) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X, target_column="target")
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(len(X))

    evaluator = FeatureImportanceRankDistance(
        distance=distance,
        use_cache=False,
    )
    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "corr" in good_score
    assert "pvalue" in good_score

    assert good_score["corr"] > 0
    assert good_score["pvalue"] > 0


def test_feature_importance_rank_distance_orients_correlation_as_maximize() -> None:
    evaluator = FeatureImportanceRankDistance(use_cache=False)

    result = evaluator._summarize_distance(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 3.0]),
    )

    assert evaluator.direction() == "maximize"
    assert result["corr"] == pytest.approx(1.0)
    assert "pvalue" in result
    metadata = evaluator.result_metadata()
    assert metadata["default_key"] == "corr"
    assert metadata["correlation_key"] == "corr"
    assert metadata["pvalue_key"] == "pvalue"
    assert metadata["pvalue_role"] == "audit_only"


def test_feature_importance_rank_distance_cache_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingClassifier:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit(self, data: pd.DataFrame, labels: pd.Series) -> "RecordingClassifier":
            del data, labels
            return self

    class ControlledExplainer:
        def __init__(self, model: RecordingClassifier) -> None:
            del model

        def shap_values(self, data: pd.DataFrame) -> np.ndarray:
            return np.tile(np.asarray([[1.0, 2.0, 3.0]]), (len(data), 1))

    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.XGBClassifier",
        RecordingClassifier,
    )
    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.shap.TreeExplainer",
        ControlledExplainer,
    )
    frame = pd.DataFrame(
        {
            "first": np.arange(12, dtype=float),
            "second": np.arange(12, dtype=float) + 1.0,
            "third": np.arange(12, dtype=float) + 2.0,
            "target": [0, 1] * 6,
        }
    )
    real = GenericDataLoader(frame, target_column="target", train_size=0.75)
    synthetic = GenericDataLoader(frame.copy(), target_column="target")

    first = FeatureImportanceRankDistance(
        distance="spearman",
        task_type="classification",
        use_cache=True,
        workspace=tmp_path,
    )
    first_result = first.evaluate(real, synthetic)

    second = FeatureImportanceRankDistance(
        distance="spearman",
        task_type="classification",
        use_cache=True,
        workspace=tmp_path,
    )
    second_result = second.evaluate(real, synthetic)

    assert second_result == first_result
    assert second.evaluate_default(real, synthetic) == pytest.approx(first_result["corr"])
    assert second.result_metadata() == first.result_metadata()
    assert second.result_metadata()["schema_version"] == "rank-v3"


def test_feature_importance_rank_distance_cache_identity_includes_distance(
    tmp_path: Path,
) -> None:
    kendall = FeatureImportanceRankDistance(
        distance="kendall",
        task_type="classification",
        workspace=tmp_path,
    )
    spearman = FeatureImportanceRankDistance(
        distance="spearman",
        task_type="classification",
        workspace=tmp_path,
    )

    assert kendall.result_metadata()["distance"] == "kendall"
    assert spearman.result_metadata()["distance"] == "spearman"
    assert kendall._distance != spearman._distance


def test_feature_importance_rank_distance_rejects_constant_rankings() -> None:
    evaluator = FeatureImportanceRankDistance(use_cache=False)

    with pytest.raises(RuntimeError, match="correlation is non-finite"):
        evaluator._summarize_distance(
            np.asarray([1.0, 1.0, 1.0]),
            np.asarray([1.0, 1.0, 1.0]),
        )


@pytest.mark.parametrize("distance", ["kendall", "spearman"])
def test_feature_importance_rank_distance_detects_disagreement(distance: str) -> None:
    evaluator = FeatureImportanceRankDistance(distance=distance, use_cache=False)

    result = evaluator._summarize_distance(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([3.0, 2.0, 1.0]),
    )

    assert result["corr"] == pytest.approx(-1.0)
    assert np.isfinite(result["pvalue"])


@pytest.mark.parametrize(
    ("task_type", "shap_values", "expected"),
    [
        (
            "classification",
            [
                np.asarray([[1.0, -2.0, 3.0], [1.0, -2.0, 3.0]]),
                np.asarray([[2.0, -1.0, 4.0], [2.0, -1.0, 4.0]]),
            ],
            np.asarray([1.5, 1.5, 3.5]),
        ),
        (
            "classification",
            np.asarray(
                [
                    [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
                    [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
                ]
            ),
            np.asarray([1.5, 3.0, 4.5]),
        ),
        (
            "regression",
            np.asarray([1.0, -2.0, 3.0]),
            np.asarray([1.0, 2.0, 3.0]),
        ),
        (
            "survival_analysis",
            np.asarray([[1.0, -2.0, 3.0], [2.0, -4.0, 6.0]]),
            np.asarray([1.5, 3.0, 4.5]),
        ),
    ],
    ids=["classification-list", "classification-array", "regression", "survival"],
)
def test_feature_importance_rank_distance_aggregates_supported_shap_shapes(
    task_type: str,
    shap_values: object,
    expected: np.ndarray,
) -> None:
    importance = FeatureImportanceRankDistance._aggregate_shap_importance(
        shap_values,
        n_features=3,
        task_type=task_type,
    )

    assert importance == pytest.approx(expected)


def test_feature_importance_rank_distance_reports_contextual_shap_shape_failure() -> None:
    evaluator = FeatureImportanceRankDistance(task_type="classification", use_cache=False)

    with pytest.raises(
        RuntimeError,
        match="synthetic classification data.*does not preserve the feature axis",
    ):
        evaluator._aggregate_shap_for_side(
            np.zeros((2, 2)),
            n_features=3,
            side="synthetic",
        )


def test_feature_importance_rank_distance_evaluates_classification_shap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingClassifier:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.is_synthetic = False

        def fit(self, data: pd.DataFrame, labels: pd.Series) -> "RecordingClassifier":
            del labels
            self.is_synthetic = float(data.iloc[:, 0].mean()) > 10.0
            return self

    class ControlledExplainer:
        def __init__(self, model: RecordingClassifier) -> None:
            self.model = model

        def shap_values(self, data: pd.DataFrame) -> np.ndarray:
            del data
            if self.model.is_synthetic:
                return np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            return np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])

    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.XGBClassifier",
        RecordingClassifier,
    )
    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.shap.TreeExplainer",
        ControlledExplainer,
    )
    real = GenericDataLoader(
        pd.DataFrame(
            {
                "first": np.arange(12, dtype=float),
                "second": np.arange(12, dtype=float) + 1.0,
                "third": np.arange(12, dtype=float) + 2.0,
                "target": [0, 1] * 6,
            }
        ),
        target_column="target",
        train_size=0.75,
        random_state=0,
    )
    synthetic = GenericDataLoader(
        pd.DataFrame(
            {
                "first": np.arange(12, dtype=float) + 20.0,
                "second": np.arange(12, dtype=float) + 21.0,
                "third": np.arange(12, dtype=float) + 22.0,
                "target": [0, 1] * 6,
            }
        ),
        target_column="target",
    )

    result = FeatureImportanceRankDistance(
        task_type="classification",
        use_cache=False,
        workspace=tmp_path,
    ).evaluate(real, synthetic)

    assert result["corr"] == pytest.approx(-1.0)
    assert np.isfinite(result["pvalue"])


def test_feature_importance_rank_distance_evaluates_regression_shap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingRegressor:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit(self, data: pd.DataFrame, labels: pd.Series) -> "RecordingRegressor":
            del data, labels
            return self

    class ControlledExplainer:
        def __init__(self, model: RecordingRegressor) -> None:
            del model

        def shap_values(self, data: pd.DataFrame) -> np.ndarray:
            return np.tile(np.asarray([[1.0, 2.0, 3.0]]), (len(data), 1))

    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.XGBRegressor",
        RecordingRegressor,
    )
    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.shap.TreeExplainer",
        ControlledExplainer,
    )
    real = GenericDataLoader(
        pd.DataFrame(
            {
                "first": np.arange(12, dtype=float),
                "second": np.arange(12, dtype=float) + 1.0,
                "third": np.arange(12, dtype=float) + 2.0,
                "target": np.arange(12, dtype=float) + 3.0,
            }
        ),
        target_column="target",
        train_size=0.75,
        random_state=0,
    )
    synthetic = GenericDataLoader(
        real.dataframe().copy(),
        target_column="target",
    )

    result = FeatureImportanceRankDistance(
        task_type="regression",
        use_cache=False,
        workspace=tmp_path,
    ).evaluate(real, synthetic)

    assert result["corr"] == pytest.approx(1.0)
    assert np.isfinite(result["pvalue"])


def test_feature_importance_rank_distance_reports_synthetic_fit_failure(
    monkeypatch,
) -> None:
    class FailingClassifier:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, *args, **kwargs):
            del args, kwargs
            raise ValueError("synthetic fixture failure")

    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.XGBClassifier",
        FailingClassifier,
    )
    data = pd.DataFrame(
        {
            "feature": [0.0, 1.0] * 5,
            "target": [0, 1] * 5,
        }
    )
    loader = GenericDataLoader(data, target_column="target")
    evaluator = FeatureImportanceRankDistance(task_type="classification", use_cache=False)

    with pytest.raises(RuntimeError, match="synthetic classification data"):
        evaluator.evaluate(loader, loader)


def test_feature_importance_rank_distance_evaluates_survival_shap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingSurvivalModel:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit(
            self,
            data: pd.DataFrame,
            time_to_event: pd.Series,
            event: pd.Series,
        ) -> "RecordingSurvivalModel":
            del data, time_to_event, event
            return self

        def explain(self, data: pd.DataFrame) -> np.ndarray:
            return np.tile(np.asarray([[[1.0], [2.0], [3.0]]]), (len(data), 1, 1))

    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.XGBSurvivalAnalysis",
        RecordingSurvivalModel,
    )
    frame = pd.DataFrame(
        {
            "first": np.arange(12, dtype=float),
            "second": np.arange(12, dtype=float) + 1.0,
            "third": np.arange(12, dtype=float) + 2.0,
            "time": np.arange(1, 13, dtype=float),
            "event": [0, 1] * 6,
        }
    )
    real = SurvivalAnalysisDataLoader(
        frame.copy(),
        time_to_event_column="time",
        target_column="event",
        train_size=0.75,
        random_state=0,
    )
    synthetic = SurvivalAnalysisDataLoader(
        frame.copy(),
        time_to_event_column="time",
        target_column="event",
    )

    result = FeatureImportanceRankDistance(
        task_type="survival_analysis",
        use_cache=False,
        workspace=tmp_path,
    ).evaluate(real, synthetic)

    assert result["corr"] == pytest.approx(1.0)
    assert np.isfinite(result["pvalue"])


def test_feature_importance_rank_distance_reports_real_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingRealClassifier:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit(self, data: pd.DataFrame, labels: pd.Series) -> "FailingRealClassifier":
            del labels
            if float(data.iloc[:, 0].mean()) < 10.0:
                raise ValueError("real fixture failure")
            return self

    class ControlledExplainer:
        def __init__(self, model: FailingRealClassifier) -> None:
            del model

        def shap_values(self, data: pd.DataFrame) -> np.ndarray:
            return np.tile(np.asarray([[1.0, 2.0, 3.0]]), (len(data), 1))

    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.XGBClassifier",
        FailingRealClassifier,
    )
    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.shap.TreeExplainer",
        ControlledExplainer,
    )
    real_data = pd.DataFrame(
        {
            "first": np.arange(12, dtype=float),
            "second": np.arange(12, dtype=float) + 1.0,
            "third": np.arange(12, dtype=float) + 2.0,
            "target": [0, 1] * 6,
        }
    )
    synthetic_data = real_data.copy()
    synthetic_data.iloc[:, 0] += 20.0

    real = GenericDataLoader(real_data, target_column="target", train_size=0.75)
    synthetic = GenericDataLoader(synthetic_data, target_column="target")

    with pytest.raises(RuntimeError, match="real classification data"):
        FeatureImportanceRankDistance(
            task_type="classification",
            use_cache=False,
            workspace=tmp_path,
        ).evaluate(real, synthetic)


def test_feature_importance_rank_distance_reports_shap_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RecordingClassifier:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit(self, data: pd.DataFrame, labels: pd.Series) -> "RecordingClassifier":
            del data, labels
            return self

    class FailingExplainer:
        def __init__(self, model: RecordingClassifier) -> None:
            del model

        def shap_values(self, data: pd.DataFrame) -> np.ndarray:
            del data
            raise ValueError("SHAP fixture failure")

    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.XGBClassifier",
        RecordingClassifier,
    )
    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.shap.TreeExplainer",
        FailingExplainer,
    )
    frame = pd.DataFrame(
        {
            "first": np.arange(12, dtype=float),
            "second": np.arange(12, dtype=float) + 1.0,
            "third": np.arange(12, dtype=float) + 2.0,
            "target": [0, 1] * 6,
        }
    )
    loader = GenericDataLoader(frame, target_column="target", train_size=0.75)

    with pytest.raises(RuntimeError, match="synthetic classification data"):
        FeatureImportanceRankDistance(
            task_type="classification",
            use_cache=False,
            workspace=tmp_path,
        ).evaluate(loader, loader)


def test_survival_benchmark_uses_group_disjoint_cv(monkeypatch) -> None:
    class RecordingEstimator:
        fitted_groups: list[tuple[int, ...]] = []

        def fit(self, X_train, T_train, Y_train, groups=None):
            del T_train, Y_train
            fitted_groups = tuple(sorted(set(X_train["group"].astype(int))))
            type(self).fitted_groups.append(fitted_groups)
            assert groups is not None
            assert len(groups) == len(X_train)
            return self

        def predict(self, X_test, time_horizons):
            return pd.DataFrame(
                np.zeros((len(X_test), len(time_horizons))),
                index=X_test.index,
            )

    monkeypatch.setattr(survival_benchmarks, "evaluate_c_index", lambda *args: 0.0)
    monkeypatch.setattr(
        survival_benchmarks, "evaluate_brier_score", lambda *args: 0.0
    )

    groups = np.repeat(["p1", "p2", "p3", "p4"], 2)
    X = pd.DataFrame({"group": np.repeat(np.arange(4), 2)})
    T = pd.Series(np.arange(1, 9, dtype=float))
    Y = pd.Series(np.repeat([0, 1, 0, 1], 2))

    survival_benchmarks.evaluate_survival_model(
        RecordingEstimator(),
        X,
        T,
        Y,
        time_horizons=[2],
        n_folds=2,
        metrics=[],
        random_state=17,
        groups=groups,
    )

    expected_splits = cross_validation_splits(
        len(X),
        y=Y,
        n_folds=2,
        seed=17,
        groups=groups,
        stratified=True,
    )
    expected_train_groups = [
        tuple(sorted(set(X.iloc[train_idx]["group"].astype(int))))
        for train_idx, _ in expected_splits
    ]

    assert sorted(RecordingEstimator.fitted_groups) == sorted(expected_train_groups)


def test_time_series_model_fit_keeps_groups_aligned_by_window(monkeypatch) -> None:
    model = TimeSeriesModel(
        task_type="regression",
        n_static_units_in=1,
        n_temporal_units_in=1,
        n_temporal_window=3,
        output_shape=[1],
        n_iter=0,
        train_ratio=0.5,
        device="cpu",
    )
    static_data = np.arange(4, dtype=float).reshape(4, 1)
    temporal_data = np.empty(4, dtype=object)
    observation_times = np.empty(4, dtype=object)
    for index, window_length in enumerate([2, 3, 2, 3]):
        temporal_data[index] = np.zeros((window_length, 1))
        observation_times[index] = np.arange(window_length, dtype=float)
    outcome = np.arange(4, dtype=float).reshape(4, 1)
    groups = ["p1", "p2", "p3", "p4"]
    recorded_groups = []
    original_dataloader = model.dataloader

    def recording_dataloader(*args, groups=None, **kwargs):
        recorded_groups.append(None if groups is None else tuple(groups))
        return original_dataloader(*args, groups=groups, **kwargs)

    monkeypatch.setattr(model, "dataloader", recording_dataloader)
    model.fit(static_data, temporal_data, observation_times, outcome, groups=groups)

    assert sorted(recorded_groups) == [("p1", "p3"), ("p2", "p4")]


def test_time_series_survival_benchmark_passes_grouped_fit_inputs(monkeypatch) -> None:
    class RecordingEstimator:
        fitted_groups: list[tuple[str, ...]] = []

        def fit(
            self,
            static_train,
            temporal_train,
            observation_times_train,
            T_train,
            Y_train,
            groups=None,
        ):
            del temporal_train, observation_times_train, T_train, Y_train
            assert groups is not None
            assert len(groups) == len(static_train)
            type(self).fitted_groups.append(tuple(sorted(set(groups))))
            return self

        def predict(self, static_test, temporal_test, observation_times_test, time_horizons):
            del temporal_test, observation_times_test
            return pd.DataFrame(
                np.zeros((len(static_test), len(time_horizons))),
                index=np.arange(len(static_test)),
            )

    monkeypatch.setattr(
        "synthcity.plugins.core.models.time_series_survival.benchmarks.evaluate_c_index",
        lambda *args: 0.0,
    )
    monkeypatch.setattr(
        "synthcity.plugins.core.models.time_series_survival.benchmarks.evaluate_brier_score",
        lambda *args: 0.0,
    )
    static = np.arange(8, dtype=float).reshape(8, 1)
    temporal = np.zeros((8, 2, 1))
    observation_times = np.zeros((8, 2))
    T = np.arange(1, 9, dtype=float)
    Y = np.repeat([0, 1, 0, 1], 2)
    groups = np.repeat(["p1", "p2", "p3", "p4"], 2)

    evaluate_ts_survival_model(
        RecordingEstimator(),
        static,
        temporal,
        observation_times,
        T,
        Y,
        time_horizons=[2],
        n_folds=2,
        metrics=[],
        random_state=17,
        groups=groups,
    )

    expected_splits = cross_validation_splits(
        len(static),
        y=Y,
        n_folds=2,
        seed=17,
        groups=groups,
        stratified=True,
    )
    expected_train_groups = [
        tuple(sorted(set(groups[train_idx]))) for train_idx, _ in expected_splits
    ]
    assert sorted(RecordingEstimator.fitted_groups) == sorted(expected_train_groups)


def test_standard_performance_mlp_forwards_grouped_fit_ids(monkeypatch) -> None:
    class RecordingMLP:
        fitted_groups = []

        def __init__(self, **kwargs):
            del kwargs

        def fit(self, X, y, groups=None):
            del y
            assert groups is not None
            assert len(groups) == len(X)
            type(self).fitted_groups.append(tuple(groups))
            return self

        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

    monkeypatch.setattr(
        "synthcity.metrics.eval_performance.MLP",
        RecordingMLP,
    )
    real = GenericDataLoader(
        pd.DataFrame(
            {
                "feature": np.arange(20, dtype=float),
                "target": np.tile([0, 1], 10),
            }
        ),
        target_column="target",
        group_ids=np.repeat([f"real-{index}" for index in range(10)], 2),
    )
    synthetic = GenericDataLoader(
        pd.DataFrame(
            {
                "feature": np.arange(20, dtype=float) + 0.1,
                "target": np.tile([0, 1], 10),
            }
        ),
        target_column="target",
        group_ids=[f"synthetic-{index}" for index in range(20)],
    )

    result = PerformanceEvaluatorMLP(
        n_folds=2,
        use_cache=False,
        workspace=Path("tmp"),
    ).evaluate(real, synthetic)

    assert result["gt"] == pytest.approx(0.5)
    assert len(RecordingMLP.fitted_groups) == 6
    assert all(groups for groups in RecordingMLP.fitted_groups)


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_standard_performance_model_failure_is_not_a_numeric_sentinel(
    monkeypatch, task_type: str
) -> None:
    class FailingEstimator:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, *args, **kwargs):
            del args, kwargs
            raise ValueError("standard performance fixture failure")

    data = pd.DataFrame(
        {
            "feature": np.arange(12, dtype=float),
            "target": ([0, 1] * 6 if task_type == "classification" else np.arange(12)),
        }
    )
    loader = GenericDataLoader(data, target_column="target")
    evaluator = (
        PerformanceEvaluatorLinear(task_type=task_type, use_cache=False)
        if task_type == "classification"
        else PerformanceEvaluatorLinear(task_type=task_type, use_cache=False)
    )
    target = (
        "synthcity.metrics.eval_performance.LogisticRegression"
        if task_type == "classification"
        else "synthcity.metrics.eval_performance.LinearRegression"
    )
    monkeypatch.setattr(target, FailingEstimator)

    with pytest.raises(RuntimeError, match="performance evaluation failed"):
        evaluator.evaluate(loader, loader)


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_evaluate_performance_regression(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    Xloader = GenericDataLoader(X, target_column="target")

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    evaluator = evaluator_t(
        task_type="regression",
        use_cache=False,
    )
    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    sz = 1000
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        Xloader,
        GenericDataLoader(X_rnd),
    )

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] <= good_score["syn_id"]
    assert score["syn_ood"] <= good_score["syn_ood"]

    def_score = evaluator.evaluate_default(Xloader, GenericDataLoader(X_rnd))

    assert def_score == score["syn_id"]


@pytest.mark.parametrize("distance", ["kendall", "spearman"])
@pytest.mark.parametrize(
    "test_plugin",
    [
        Plugins().get("ctgan"),
    ],
)
@pytest.mark.xfail
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
@pytest.mark.slow_1
@pytest.mark.slow
def test_evaluate_feature_importance_rank_dist_reg(
    distance: str, test_plugin: Plugin
) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X, target_column="target")

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(len(X))

    evaluator = FeatureImportanceRankDistance(
        distance=distance,
        task_type="regression",
        use_cache=False,
    )
    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "corr" in score
    assert "pvalue" in score

    assert score["corr"] > 0
    assert score["pvalue"] > 0


@pytest.mark.slow_1
@pytest.mark.slow
@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
def test_evaluate_performance_survival_analysis(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    X = load_rossi()
    T = X["week"]
    time_horizons = np.linspace(T.min(), T.max(), num=4)[1:-1].tolist()

    Xloader = SurvivalAnalysisDataLoader(
        X,
        target_column="arrest",
        time_to_event_column="week",
        time_horizons=time_horizons,
    )
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t(
        task_type="survival_analysis",
        use_cache=False,
    )
    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "gt.c_index" in good_score
    assert "gt.brier_score" in good_score
    assert "syn_id.c_index" in good_score
    assert "syn_id.brier_score" in good_score
    assert "syn_ood.c_index" in good_score
    assert "syn_ood.brier_score" in good_score

    sz = 100
    X_rnd = pd.DataFrame(
        np.random.randn(sz, len(X.columns) - 1),
        columns=[col for col in X.columns if col != "week"],
    )
    X_rnd.insert(loc=0, column="week", value=np.random.randint(1, 52, size=sz))
    X_rnd["arrest"] = 1
    score = evaluator.evaluate(
        Xloader,
        create_from_info(X_rnd, Xloader.info()),
    )

    assert "gt.c_index" in score
    assert "gt.brier_score" in score
    assert "syn_id.c_index" in score
    assert "syn_id.brier_score" in score
    assert "syn_ood.c_index" in score
    assert "syn_ood.brier_score" in score

    assert score["syn_id.c_index"] < 1
    assert score["syn_id.brier_score"] < 1
    assert score["syn_ood.c_index"] < 1
    assert score["syn_ood.brier_score"] < 1
    assert good_score["gt.c_index"] < 1
    assert good_score["gt.brier_score"] < 1

    def_score = evaluator.evaluate_default(
        Xloader, create_from_info(X_rnd, Xloader.info())
    )

    assert def_score == score["syn_id.c_index"] - score["syn_id.brier_score"]


@pytest.mark.parametrize("distance", ["kendall", "spearman"])
@pytest.mark.parametrize(
    "test_plugin",
    [
        Plugins().get("ctgan"),
    ],
)
@pytest.mark.xfail
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
@pytest.mark.slow_1
@pytest.mark.slow
def test_evaluate_feature_importance_rank_dist_surv(
    distance: str, test_plugin: Plugin
) -> None:
    X = load_rossi()
    T = X["week"]
    time_horizons = np.linspace(T.min(), T.max(), num=4).tolist()

    Xloader = SurvivalAnalysisDataLoader(
        X,
        target_column="arrest",
        time_to_event_column="week",
        time_horizons=time_horizons,
    )

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(len(X))

    evaluator = FeatureImportanceRankDistance(
        distance=distance,
        task_type="survival_analysis",
        use_cache=False,
    )
    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "corr" in good_score
    assert "pvalue" in good_score

    assert good_score["corr"] > 0
    assert good_score["pvalue"] > 0


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorXGB,
    ],
)
@pytest.mark.parametrize("target", [None, "target", "sepal width (cm)"])
def test_evaluate_performance_custom_labels(
    test_plugin: Plugin, evaluator_t: Type, target: Optional[str]
) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X, target_column="target")

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t(use_cache=False)

    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score


@pytest.mark.slow_1
@pytest.mark.slow
@pytest.mark.parametrize("test_plugin", [Plugins().get("timegan")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorMLP,
    ],
)
def test_evaluate_performance_time_series(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    (
        static_data,
        temporal_data,
        observation_times,
        outcome,
    ) = GoogleStocksDataloader().load()
    data = TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        outcome=outcome,
    )

    test_plugin.fit(data)
    data_gen = test_plugin.generate(100)

    evaluator = evaluator_t(
        task_type="time_series",
        use_cache=False,
    )
    good_score = evaluator.evaluate(
        data,
        data_gen,
    )

    assert "gt" in good_score
    assert "syn_id" in good_score
    assert "syn_ood" in good_score

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(data.columns)), columns=data.columns)
    X_rnd["arrest"] = 1
    score = evaluator.evaluate(
        data,
        create_from_info(X_rnd, data.info()),
    )

    assert "gt" in score
    assert "syn_id" in score
    assert "syn_ood" in score

    assert score["syn_id"] < 1
    assert score["syn_ood"] < 1
    assert good_score["gt"] < 1
    assert good_score["syn_id"] > score["syn_id"]
    assert good_score["syn_ood"] > score["syn_ood"]


def test_time_series_classification_uses_group_disjoint_cv() -> None:
    class RecordingEstimator:
        fitted_groups: list[tuple[int, ...]] = []

        def fit(
            self,
            static_train,
            temporal_train,
            observation_times_train,
            y_train,
            groups=None,
        ):
            del temporal_train, observation_times_train, y_train
            assert groups is not None
            assert len(groups) == len(static_train)
            fitted_groups = tuple(sorted(set(static_train[:, 0].astype(int))))
            type(self).fitted_groups.append(fitted_groups)
            return self

        def predict(self, static_test, temporal_test, observation_times_test):
            del temporal_test, observation_times_test
            return np.zeros(len(static_test))

    static = np.repeat(np.arange(4), 2).reshape(-1, 1)
    temporal = np.zeros((8, 2, 1))
    observation_times = np.zeros((8, 2))
    labels = np.repeat([0, 1, 0, 1], 2)
    groups = np.repeat(["p1", "p2", "p3", "p4"], 2)

    evaluate_ts_classification(
        RecordingEstimator(),
        static,
        temporal,
        observation_times,
        labels,
        n_folds=2,
        random_state=17,
        groups=groups,
    )

    expected_splits = cross_validation_splits(
        len(static),
        y=labels,
        n_folds=2,
        seed=17,
        groups=groups,
        stratified=True,
    )
    expected_train_groups = [
        tuple(sorted(set(static[train_idx, 0].astype(int))))
        for train_idx, _ in expected_splits
    ]

    assert sorted(RecordingEstimator.fitted_groups) == sorted(expected_train_groups)


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        PerformanceEvaluatorLinear,
        PerformanceEvaluatorMLP,
        PerformanceEvaluatorXGB,
    ],
)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_evaluate_performance_time_series_survival(
    test_plugin: Plugin, evaluator_t: Type
) -> None:
    static_data, temporal_data, observation_times, outcome = PBCDataloader().load()

    T, E = outcome

    data = TimeSeriesSurvivalDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        static_data=static_data,
        T=T,
        E=E,
    )

    test_plugin.fit(data)
    data_gen = test_plugin.generate(len(temporal_data))

    evaluator = evaluator_t(
        task_type="time_series_survival",
    )

    good_score = evaluator.evaluate(
        data,
        data_gen,
    )
    assert "gt.c_index" in good_score
    assert "gt.brier_score" in good_score
    assert "syn_id.c_index" in good_score
    assert "syn_id.brier_score" in good_score
    assert "syn_ood.c_index" in good_score
    assert "syn_ood.brier_score" in good_score

    assert good_score["syn_id.c_index"] < 1
    assert good_score["syn_ood.c_index"] < 1

    def_score = evaluator.evaluate_default(data, data_gen)

    assert def_score == good_score["syn_id.c_index"] - good_score["syn_id.brier_score"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.slow_1
@pytest.mark.slow
def test_image_support_perf() -> None:
    dataset = datasets.MNIST(".", download=True)

    X1 = ImageDataLoader(dataset).sample(100)
    X2 = ImageDataLoader(dataset).sample(100)

    for evaluator in [
        PerformanceEvaluatorMLP,
    ]:
        score = evaluator().evaluate(X1, X2)
        assert isinstance(score, dict)
        for k in score:
            assert score[k] >= 0
            assert not np.isnan(score[k])
