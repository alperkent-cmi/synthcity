import os
from pathlib import Path

import numpy as np
import pandas as pd
import synthcity.metrics.eval_detection as detection
from synthcity.utils.optimizer import search_parameters


def test_search_parameters_uses_group_disjoint_hpo_data(monkeypatch) -> None:
    fitted_groups = []

    class RecordingModel:
        def __init__(self, **kwargs):
            del kwargs

        @staticmethod
        def name() -> str:
            return "recording"

        @staticmethod
        def hyperparameter_space() -> list:
            return []

        def fit(self, data, groups=None):
            assert groups is not None
            fitted_groups.append((set(data.index), tuple(groups)))
            return self

        def generate(self, count: int) -> pd.DataFrame:
            return pd.DataFrame(np.zeros((count, 1)), columns=["feature"])

    class RecordingMetric:
        def evaluate(self, real, synthetic):
            assert real.group_ids is not None
            assert synthetic.group_ids is not None
            assert len(synthetic.group_ids) == len(synthetic)
            return {"mean": 0.5}

    monkeypatch.setattr(
        "synthcity.utils.optimizer.create_study",
        lambda **kwargs: (None, None),
    )
    monkeypatch.setattr(
        detection,
        "SyntheticDetectionMLP",
        lambda: RecordingMetric(),
    )

    data = pd.DataFrame({"feature": np.arange(12, dtype=float)})
    groups = np.repeat([f"p{index}" for index in range(6)], 2)

    workspace = Path("tmp") / f"test_optimizer_grouped_{os.getpid()}"
    workspace.mkdir(parents=True, exist_ok=True)
    result = search_parameters(
        RecordingModel,
        data,
        n_trials=0,
        workspace=workspace,
        groups=(group for group in groups),
    )

    assert result == {}
    assert len(fitted_groups) == 1
    _, train_groups = fitted_groups[0]
    assert all(train_groups.count(group) == 2 for group in set(train_groups))