import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from synthcity.utils.evaluation import evaluate_classifier, evaluate_regression


def test_evaluate_classifier_accepts_group_disjoint_splits() -> None:
    frame = pd.DataFrame(
        {
            "feature": np.arange(12, dtype=float),
            "target": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    groups = np.repeat(["p1", "p2", "p3", "p4", "p5", "p6"], 2)
    seen = []

    class RecordingClassifier(LogisticRegression):
        def fit(self, X, y):
            self.train_indices = set(X.index)
            return super().fit(X, y)

        def predict(self, X):
            seen.append((self.train_indices, set(X.index)))
            return np.zeros(len(X), dtype=int)

    evaluate_classifier(
        RecordingClassifier(),
        frame[["feature"]],
        frame["target"],
        n_folds=3,
        groups=groups,
    )

    group_by_index = dict(enumerate(groups))
    for train_indices, test_indices in seen:
        assert {
            group_by_index[index] for index in train_indices
        }.isdisjoint({group_by_index[index] for index in test_indices})


def test_evaluate_regression_rejects_misaligned_groups() -> None:
    frame = pd.DataFrame({"feature": np.arange(6, dtype=float)})
    target = pd.DataFrame({"target": np.arange(6, dtype=float)})

    with pytest.raises(ValueError, match="groups length 2 does not match data length 6"):
        evaluate_regression(
            LinearRegression(),
            frame,
            target,
            n_folds=2,
            groups=["p1", "p2"],
        )