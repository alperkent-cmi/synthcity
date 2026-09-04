import numpy as np
import pandas as pd
import pytest
import torch

from synthcity.plugins.core.models.time_to_event import benchmarks
from synthcity.plugins.core.models.time_to_event import tte_deephit
from synthcity.plugins.core.models.time_to_event.tte_date import TimeEventGAN
from synthcity.plugins.core.models.time_to_event.tte_deephit import (
    DeephitTimeToEvent,
)
from synthcity.plugins.core.models.time_to_event.tte_tenn import TimeEventNN
from synthcity.utils.evaluation import cross_validation_splits


def _feature_groups(loader) -> set[int]:
    features = loader.dataset.tensors[0].cpu().numpy()
    return {int(feature_row[0]) // 2 for feature_row in features}


@pytest.mark.parametrize("model_type", [TimeEventGAN, TimeEventNN])
def test_neural_time_to_event_validation_is_group_disjoint(model_type) -> None:
    if model_type is TimeEventGAN:
        model = model_type(
            n_features=1,
            n_units_latent=1,
            generator_n_layers_hidden=1,
            generator_n_units_hidden=2,
            discriminator_n_layers_hidden=1,
            discriminator_n_units_hidden=2,
        )
    else:
        model = model_type(
            n_features=1,
            n_layers_hidden=1,
            n_units_hidden=2,
        )

    features = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    times = torch.arange(1, 13, dtype=torch.float32)
    events = torch.tensor([0, 1] * 6)
    groups = np.repeat([f"p{index}" for index in range(6)], 2)

    train_loader, validation_loader = model.dataloader(
        features,
        times,
        events,
        groups=groups,
    )

    train_groups = _feature_groups(train_loader)
    validation_groups = _feature_groups(validation_loader)

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups | validation_groups == set(range(6))


def test_deephit_time_to_event_validation_uses_groups(monkeypatch) -> None:
    fitted_inputs = {}

    class FakeLabTrans:
        cuts = np.array([0.0, 1.0])
        out_features = 2

        def fit_transform(self, durations, events):
            return np.asarray(durations), np.asarray(events)

        def transform(self, durations, events):
            return np.asarray(durations), np.asarray(events)

    class FakeOptimizer:
        def set_lr(self, learning_rate):
            fitted_inputs["learning_rate"] = learning_rate

    class FakeDeepHitSingle:
        @classmethod
        def label_transform(cls, num_durations):
            del num_durations
            return FakeLabTrans()

        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.optimizer = FakeOptimizer()

        def fit(
            self, features, targets, batch_size, epochs, callbacks, val_data, verbose
        ):
            del targets, batch_size, epochs, callbacks, verbose
            fitted_inputs["train"] = features
            fitted_inputs["validation"] = val_data[0]

    monkeypatch.setattr(tte_deephit, "DeepHitSingle", FakeDeepHitSingle)

    model = DeephitTimeToEvent(
        num_durations=2,
        dim_hidden=2,
        epochs=1,
        device="cpu",
    )
    monkeypatch.setattr(model, "_fit_censoring_model", lambda *args: model)

    features = pd.DataFrame({"feature": np.arange(12, dtype=float)})
    times = pd.Series(np.arange(1, 13, dtype=int))
    events = pd.Series([0, 1] * 6)
    groups = np.repeat([f"p{index}" for index in range(6)], 2)

    model.fit(features, times, events, groups=groups)

    train_groups = {int(row[0]) // 2 for row in fitted_inputs["train"]}
    validation_groups = {int(row[0]) // 2 for row in fitted_inputs["validation"]}
    assert train_groups.isdisjoint(validation_groups)


def test_time_to_event_benchmark_passes_fold_groups(monkeypatch) -> None:
    class RecordingModel:
        fitted_groups: list[tuple[str, ...]] = []

        def __init__(self, **kwargs):
            del kwargs

        def fit(self, X_train, T_train, E_train, groups=None):
            del T_train, E_train
            assert groups is not None
            assert len(groups) == len(X_train)
            type(self).fitted_groups.append(tuple(sorted(set(groups))))
            return self

        def predict(self, X_test):
            return pd.Series(np.zeros(len(X_test)), index=X_test.index)

    monkeypatch.setattr(benchmarks, "get_model_template", lambda name: RecordingModel)
    monkeypatch.setattr(benchmarks, "expected_time_error", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(benchmarks, "c_index", lambda *args, **kwargs: 0.0)

    features = pd.DataFrame({"feature": np.arange(12, dtype=float)})
    times = pd.Series(np.arange(1, 13, dtype=float))
    events = pd.Series([0, 1] * 6)
    groups = np.repeat([f"p{index}" for index in range(6)], 2)

    benchmarks.evaluate_model(
        "recording",
        {},
        features,
        times,
        events,
        n_folds=2,
        random_state=17,
        groups=groups,
    )

    expected_splits = cross_validation_splits(
        len(features),
        y=events,
        n_folds=2,
        seed=17,
        groups=groups,
        stratified=True,
    )
    expected_train_groups = [
        tuple(sorted(set(groups[train_idx]))) for train_idx, _ in expected_splits
    ]

    assert sorted(RecordingModel.fitted_groups) == sorted(expected_train_groups)
