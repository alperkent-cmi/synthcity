# stdlib
from pathlib import Path

# third party
import pytest

# synthcity absolute
from synthcity.utils.datasets.time_series import google_stocks
from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader


@pytest.fixture(autouse=True)
def redirect_dataset_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(google_stocks, "df_path", tmp_path / "goog.csv")


def test_dataloader() -> None:
    loader = GoogleStocksDataloader(seq_len=20)

    _, temporal_data, observation_times, outcome = loader.load()

    assert outcome.shape == (len(temporal_data), 1)
    assert len(temporal_data) == 40
    assert len(observation_times) == 40
    for idx, item in enumerate(temporal_data):
        assert item.shape == (20, 5)
        assert len(observation_times[idx]) == 20
