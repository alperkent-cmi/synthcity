import numpy as np
import pandas as pd
import pytest
import torch
from synthcity.plugins.core.dataset import FlexibleDataset
from synthcity.plugins.core.models.gan import GAN
from synthcity.plugins.core.models.image_gan import ImageGAN
from synthcity.plugins.core.models.ts_gan import TimeSeriesGAN
from synthcity.plugins.core.models.ts_vae import TimeSeriesVAE
from synthcity.plugins.core.models.vae import VAE
from synthcity.utils.callbacks import ValidationMixin


def _group_set(rows: torch.Tensor) -> set[int]:
    return {int(row[0]) // 2 for row in rows}


@pytest.mark.parametrize("model_type", ["gan", "vae"])
def test_tabular_neural_validation_is_group_disjoint(model_type: str) -> None:
    if model_type == "gan":
        model = GAN(
            n_features=1,
            n_units_latent=2,
            generator_n_iter=1,
        )
        model.patience_metric = object()
    else:
        model = VAE(
            n_features=1,
            n_units_embedding=2,
            n_iter=1,
        )

    features = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
    groups = np.repeat([f"p{index}" for index in range(6)], 2)

    train, validation, _, _ = model._train_test_split(
        features,
        None,
        groups=groups,
    )

    assert _group_set(train).isdisjoint(_group_set(validation))
    assert _group_set(train) | _group_set(validation) == set(range(6))


def test_image_gan_validation_is_group_disjoint() -> None:
    images = torch.arange(12, dtype=torch.float32).reshape(12, 1, 1, 1)
    labels = torch.zeros(12, dtype=torch.long)
    dataset = FlexibleDataset(torch.utils.data.TensorDataset(images, labels))
    model = ImageGAN(
        image_generator=torch.nn.Identity(),
        image_discriminator=torch.nn.Identity(),
        n_units_latent=1,
        n_channels=1,
    )
    model.patience_metric = object()
    groups = np.repeat([f"p{index}" for index in range(6)], 2)

    train, _, validation, _ = model._train_test_split(dataset, groups=groups)

    train_groups = set(groups[train.indices])
    validation_groups = set(groups[validation.indices])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups | validation_groups == set(groups)


class _ValidationProbe(ValidationMixin):
    def generate(self, count: int, cond: object = None) -> pd.DataFrame:
        return pd.DataFrame()


def test_callback_validation_is_group_disjoint() -> None:
    data = pd.DataFrame({"row": np.arange(12)})
    groups = np.repeat([f"p{index}" for index in range(6)], 2)
    callback = _ValidationProbe(valid_metric=object(), valid_size=0.5)

    train = callback._set_val_data(data, groups=groups)

    train_groups = set(groups[train.index])
    validation_groups = set(groups[callback.valid_set.index])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups | validation_groups == set(groups)


def test_image_gan_patience_metric_preserves_validation_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingMetric:
        def __init__(self) -> None:
            self.real_groups = None
            self.synthetic_groups = None

        def direction(self) -> str:
            return "minimize"

        def evaluate(self, real: object, synthetic: object) -> float:
            self.real_groups = real.group_ids
            self.synthetic_groups = synthetic.group_ids
            return 0.0

    metric = RecordingMetric()
    model = ImageGAN(
        image_generator=torch.nn.Identity(),
        image_discriminator=torch.nn.Identity(),
        n_units_latent=1,
        n_channels=1,
    )
    model.patience_metric = metric
    groups = np.array(["p0", "p0", "p1", "p1"], dtype=object)
    images = torch.zeros(4, 1, 2, 2)
    monkeypatch.setattr(
        model,
        "generate",
        lambda count, cond=None: torch.zeros(count, 1, 2, 2),
    )

    model._evaluate_patience_metric(images, None, np.inf, 0, groups=groups)

    assert np.array_equal(metric.real_groups, groups)
    assert metric.synthetic_groups is None


def test_time_series_gan_forwards_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TimeSeriesGAN(
        n_static_units=1,
        n_static_units_latent=1,
        n_temporal_units=1,
        n_temporal_window=2,
        n_temporal_units_latent=1,
    )
    groups = ["p0", "p1", "p2"]
    received = {}

    def record_train(
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        cond: object = None,
        groups: object = None,
    ) -> TimeSeriesGAN:
        received["groups"] = groups
        return model

    monkeypatch.setattr(model, "_train", record_train)
    model.fit(
        np.zeros((3, 1)),
        np.zeros((3, 2, 1)),
        np.zeros((3, 2)),
        groups=groups,
    )

    assert received["groups"] == groups


def test_time_series_vae_forwards_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    model = TimeSeriesVAE(
        n_static_units=1,
        n_static_units_embedding=1,
        n_temporal_units=1,
        n_temporal_window=2,
        n_temporal_units_embedding=1,
        n_iter=1,
    )
    groups = ["p0", "p1", "p2"]
    received = {}

    def record_train(
        static: torch.Tensor,
        temporal: torch.Tensor,
        observation_times: torch.Tensor,
        groups: object = None,
    ) -> TimeSeriesVAE:
        received["groups"] = groups
        return model

    monkeypatch.setattr(model, "_train", record_train)
    model.fit(
        np.zeros((3, 1)),
        np.zeros((3, 2, 1)),
        np.zeros((3, 2)),
        groups=groups,
    )

    assert received["groups"] == groups