# stdlib
import sys
from pathlib import Path
from typing import Type

# third party
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_iris
from torchvision import datasets

# synthcity absolute
from synthcity.metrics import Metrics
from synthcity.metrics.eval_detection import (
    SyntheticDetectionGMM,
    SyntheticDetectionLinear,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import (
    GenericDataLoader,
    ImageDataLoader,
    TimeSeriesDataLoader,
)
from synthcity.utils.datasets.time_series import google_stocks as google_stocks_dataset
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


@pytest.fixture(autouse=True)
def _redirect_dataset_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(google_stocks_dataset, "df_path", tmp_path / "goog.csv")


@pytest.mark.parametrize("reduction", ["mean", "max", "min"])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        SyntheticDetectionXGB,
    ],
)
def test_detect_reduction(reduction: str, evaluator_t: Type) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin = Plugins().get("marginal_distributions")
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t(
        reduction=reduction,
        use_cache=False,
    )

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert reduction in score
    assert "mean" in score
    assert score["raw_auc"] == pytest.approx(score[reduction])
    assert score["effective_auc_v2"] == pytest.approx(
        max(score[reduction], 1.0 - score[reduction])
    )
    assert evaluator.result_metadata()["raw_auc_key"] == "raw_auc"
    assert evaluator.result_metadata()["reducer"] == reduction
    assert evaluator.result_metadata()["default_key"] == "effective_auc_v2"

    def_score = evaluator.evaluate_default(Xloader, X_gen)

    assert def_score == score["effective_auc_v2"]


@pytest.mark.parametrize(
    ("raw_auc", "expected"),
    [(0.2, 0.8), (0.5, 0.5), (0.9, 0.9)],
)
def test_effective_auc_is_inversion_aware(raw_auc: float, expected: float) -> None:
    assert SyntheticDetectionLinear.effective_auc(raw_auc) == pytest.approx(expected)


@pytest.mark.parametrize("raw_auc", [-0.01, 1.01, float("nan")])
def test_effective_auc_rejects_invalid_values(raw_auc: float) -> None:
    with pytest.raises(ValueError, match="AUC must be finite"):
        SyntheticDetectionLinear.effective_auc(raw_auc)


def test_identical_tabular_distributions_are_chance(tmp_path) -> None:
    frame = pd.DataFrame({"feature": np.zeros(20)})
    loader = GenericDataLoader(frame)
    evaluator = SyntheticDetectionLinear(
        n_folds=5,
        random_state=0,
        use_cache=False,
        workspace=tmp_path,
    )

    result = evaluator.evaluate(loader, loader)

    assert result["raw_auc"] == pytest.approx(0.5)
    assert result["effective_auc_v2"] == pytest.approx(0.5)
    assert evaluator.evaluate_default(loader, loader) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("evaluator_t", "model_attribute"),
    [
        (SyntheticDetectionXGB, "XGBClassifier"),
        (SyntheticDetectionMLP, "MLP"),
        (SyntheticDetectionLinear, "LogisticRegression"),
    ],
)
@pytest.mark.parametrize(
    ("mode", "real_marker", "synthetic_marker", "expected_raw_auc", "expected_effective_auc"),
    [
        ("chance", 0.0, 0.0, 0.5, 0.5),
        ("separable", 0.0, 1.0, 1.0, 1.0),
        ("inverted", 0.0, 1.0, 0.0, 1.0),
    ],
)
def test_controlled_detector_auc_matrix(
    monkeypatch,
    evaluator_t: Type,
    model_attribute: str,
    mode: str,
    real_marker: float,
    synthetic_marker: float,
    expected_raw_auc: float,
    expected_effective_auc: float,
) -> None:
    class ControlledDetector:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, data, labels):
            del data, labels
            return self

        def predict_proba(self, data):
            if mode == "chance":
                positive_probability = np.full(len(data), 0.5)
            elif mode == "separable":
                positive_probability = data[:, 0]
            else:
                positive_probability = 1.0 - data[:, 0]
            return np.column_stack((1.0 - positive_probability, positive_probability))

    monkeypatch.setattr(
        f"synthcity.metrics.eval_detection.{model_attribute}",
        ControlledDetector,
    )
    real = GenericDataLoader(
        pd.DataFrame(
            {
                "marker": [real_marker] * 20,
                "target": [0, 1] * 10,
            }
        ),
        target_column="target",
    )
    synthetic = GenericDataLoader(
        pd.DataFrame(
            {
                "marker": [synthetic_marker] * 20,
                "target": [0, 1] * 10,
            }
        ),
        target_column="target",
    )

    evaluator = evaluator_t(n_folds=5, random_state=0, use_cache=False)
    result = evaluator.evaluate(real, synthetic)

    assert result["mean"] == pytest.approx(expected_raw_auc)
    assert result["raw_auc"] == pytest.approx(expected_raw_auc)
    assert result["effective_auc_v2"] == pytest.approx(expected_effective_auc)
    assert evaluator.evaluate_default(real, synthetic) == pytest.approx(
        expected_effective_auc
    )


@pytest.mark.parametrize(
    ("evaluator_t", "patch_target"),
    [
        (SyntheticDetectionLinear, "LogisticRegression"),
        (SyntheticDetectionXGB, "XGBClassifier"),
        (SyntheticDetectionMLP, "suggest_image_classifier_arch"),
    ],
)
@pytest.mark.parametrize(
    ("mode", "expected_raw_auc", "expected_effective_auc"),
    [
        ("chance", 0.5, 0.5),
        ("separable", 1.0, 1.0),
        ("inverted", 0.0, 1.0),
    ],
)
def test_controlled_image_detector_auc_matrix(
    monkeypatch: pytest.MonkeyPatch,
    evaluator_t: Type,
    patch_target: str,
    mode: str,
    expected_raw_auc: float,
    expected_effective_auc: float,
) -> None:
    class ControlledDetector:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit(
            self,
            data: object,
            labels: object = None,
            groups: object = None,
        ) -> "ControlledDetector":
            del data, labels, groups
            return self

        def predict_proba(self, data: object) -> np.ndarray | torch.Tensor:
            if isinstance(data, torch.Tensor):
                marker = data[:, 0, 0, 0].detach().cpu().numpy()
            else:
                marker = np.asarray(data)[:, 0]

            if mode == "chance":
                positive_probability = np.full(len(marker), 0.5)
            else:
                positive_probability = (marker > 0.0).astype(float)
                if mode == "inverted":
                    positive_probability = 1.0 - positive_probability
            probabilities = np.column_stack(
                (1.0 - positive_probability, positive_probability)
            )
            if isinstance(data, torch.Tensor):
                return torch.from_numpy(probabilities)
            return probabilities

    from synthcity.metrics import eval_detection

    if patch_target == "suggest_image_classifier_arch":
        monkeypatch.setattr(
            eval_detection,
            patch_target,
            lambda **kwargs: ControlledDetector(**kwargs),
        )
    else:
        monkeypatch.setattr(eval_detection, patch_target, ControlledDetector)

    labels = torch.tensor([0, 1] * 10)
    real = ImageDataLoader(
        (torch.zeros((20, 1, 2, 2)), labels),
        height=2,
        width=2,
    )
    synthetic = ImageDataLoader(
        (torch.ones((20, 1, 2, 2)), labels),
        height=2,
        width=2,
    )

    result = evaluator_t(n_folds=5, random_state=0, use_cache=False).evaluate(
        real,
        synthetic,
    )

    assert result["mean"] == pytest.approx(expected_raw_auc)
    assert result["raw_auc"] == pytest.approx(expected_raw_auc)
    assert result["effective_auc_v2"] == pytest.approx(expected_effective_auc)


def test_detection_mlp_forwards_grouped_fit_ids(monkeypatch) -> None:
    class RecordingMLP:
        fitted_groups = []

        def __init__(self, **kwargs):
            del kwargs

        def fit(self, X, y, groups=None):
            del X, y
            assert groups is not None
            type(self).fitted_groups.append(tuple(groups))
            return self

        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

    monkeypatch.setattr(
        "synthcity.metrics.eval_detection.MLP",
        RecordingMLP,
    )
    real = GenericDataLoader(
        pd.DataFrame({"feature": np.arange(8, dtype=float)}),
        group_ids=np.repeat(["real-0", "real-1", "real-2", "real-3"], 2),
    )
    synthetic = GenericDataLoader(
        pd.DataFrame({"feature": np.arange(8, dtype=float) + 0.1}),
        group_ids=[f"synthetic-{index}" for index in range(8)],
    )

    result = SyntheticDetectionMLP(
        n_folds=2,
        use_cache=False,
    ).evaluate(real, synthetic)

    assert result["mean"] == pytest.approx(0.5)
    assert len(RecordingMLP.fitted_groups) == 2
    assert all(groups for groups in RecordingMLP.fitted_groups)


@pytest.mark.parametrize("test_plugin", [Plugins().get("marginal_distributions")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        SyntheticDetectionXGB,
        SyntheticDetectionMLP,
        SyntheticDetectionLinear,
    ],
)
def test_detect_synth_generic(test_plugin: Plugin, evaluator_t: Type) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y

    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(100)

    evaluator = evaluator_t()

    good_score = evaluator.evaluate(
        Xloader,
        X_gen,
    )["mean"]

    assert good_score > 0
    assert good_score <= 1

    sz = 100
    X_rnd = pd.DataFrame(np.random.randn(sz, len(X.columns)), columns=X.columns)
    score = evaluator.evaluate(
        Xloader,
        GenericDataLoader(X_rnd),
    )["mean"]

    assert score > 0
    assert score <= 1
    assert good_score < score

    assert evaluator.type() == "detection"
    assert evaluator.direction() == "minimize"


def test_gmm_detection_is_explicitly_audit_only() -> None:
    assert "detection_gmm" not in set(Metrics.list()["detection"])

    frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    loader = GenericDataLoader(frame)

    with pytest.raises(RuntimeError, match="GMM is audit-only"):
        SyntheticDetectionGMM().evaluate(loader, loader)


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        SyntheticDetectionXGB,
    ],
)
def test_detect_synth_timeseries(test_plugin: Plugin, evaluator_t: Type) -> None:
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
    data_gen = test_plugin.generate(200)

    evaluator = evaluator_t()

    good_score = evaluator.evaluate(
        data,
        data_gen,
    )["mean"]

    assert good_score > 0
    assert good_score <= 1

    sz = 200
    data_rnd = TimeSeriesDataLoader.from_info(
        pd.DataFrame(np.random.randn(sz, len(data.columns)), columns=data.columns),
        data.info(),
    )

    score = evaluator.evaluate(
        data,
        data_rnd,
    )["mean"]

    assert score > 0
    assert score <= 1
    assert good_score < score

    assert evaluator.type() == "detection"
    assert evaluator.direction() == "minimize"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
@pytest.mark.slow_1
@pytest.mark.slow
def test_image_support_detection(tmp_path: Path) -> None:

    dataset = datasets.MNIST(tmp_path, download=True)

    X1 = ImageDataLoader(dataset).sample(100)
    X2 = ImageDataLoader(dataset).sample(100)

    for evaluator in [
        SyntheticDetectionLinear,
        SyntheticDetectionXGB,
        SyntheticDetectionMLP,
    ]:
        score = evaluator().evaluate(X1, X2)
        assert isinstance(score, dict)
        for k in score:
            assert score[k] >= 0
            assert not np.isnan(score[k])
