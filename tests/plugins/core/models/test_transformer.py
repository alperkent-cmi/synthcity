# third party
import pytest
import torch

# synthcity absolute
from synthcity.plugins.core.models.transformer import TransformerModel
from synthcity.utils.constants import DEVICE
from synthcity.utils.datasets.time_series import google_stocks
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


def test_sanity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(google_stocks, "df_path", tmp_path / "goog.csv")
    _, temporal, _, _ = GoogleStocksDataloader(as_numpy=True).load()

    model = TransformerModel(n_units_in=temporal[0].shape[-1], n_units_hidden=10)
    temporal = torch.from_numpy(temporal).to(DEVICE)
    out = model.forward(temporal)

    assert out.shape == (len(temporal), temporal[0].shape[0], 10)
