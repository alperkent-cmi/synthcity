# stdlib
import copy
from typing import Any, Dict, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def _validated_groups(groups: Any, n_samples: int) -> np.ndarray:
    values = list(groups)
    if len(values) != n_samples:
        raise ValueError(
            f"groups length {len(values)} does not match data length {n_samples}"
        )
    validated = np.empty(len(values), dtype=object)
    validated[:] = values
    return validated


def cross_validation_splits(
    n_samples: int,
    y: Any = None,
    n_folds: int = 3,
    seed: int = 0,
    groups: Any = None,
    stratified: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic row- or group-disjoint cross-validation splits."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    if n_samples < n_folds:
        raise ValueError(
            f"n_samples must be at least n_folds, got {n_samples} and {n_folds}"
        )
    if stratified and y is None:
        raise ValueError("stratified splits require y")

    indices = np.arange(n_samples)
    if groups is None:
        if stratified:
            splitter = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=seed
            )
            return list(splitter.split(indices, np.asarray(y)))
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(indices))

    validated_groups = _validated_groups(groups, n_samples)
    if stratified:
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=seed
        )
        return list(splitter.split(indices, np.asarray(y), validated_groups))
    splitter = GroupKFold(n_splits=n_folds)
    return list(splitter.split(indices, groups=validated_groups))


def train_test_indices(
    n_samples: int,
    train_size: float,
    seed: int = 0,
    y: Any = None,
    groups: Any = None,
    stratified: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic row- or group-disjoint train/test indices."""
    if stratified and y is None:
        raise ValueError("stratified split requires y")

    indices = np.arange(n_samples)
    if groups is None:
        stratify = np.asarray(y) if stratified else None
        return train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            stratify=stratify,
        )

    validated_groups = _validated_groups(groups, n_samples)
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=seed,
    )
    return next(splitter.split(indices, groups=validated_groups))


def evaluate_classifier(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.Series,
    n_folds: int = 3,
    seed: int = 0,
    groups: Any = None,
) -> Dict:
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    metric = "aucroc"
    metric_ = np.zeros(n_folds)

    cv_splits = cross_validation_splits(
        len(X),
        y=Y,
        n_folds=n_folds,
        seed=seed,
        groups=groups,
        stratified=True,
    )
    for indx, (train_index, test_index) in enumerate(cv_splits):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        metric_[indx] = roc_auc_score(Y_test, preds)

        indx += 1

    output_clf = generate_score(metric_)

    return {
        "clf": {
            metric: output_clf,
        },
        "str": {
            metric: print_score(output_clf),
        },
    }


def evaluate_regression(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_folds: int = 3,
    seed: int = 0,
    groups: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    """Helper for evaluating regression tasks.
    Args:
        estimator:
            The regressor to evaluate
        X:
            covariates
        Y:
            outcomes
        n_folds: int
            Number of cross-validation folds
        metric: str
            r2
        seed: int
            Random seed
    """
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    metric = "r2"
    metric_ = np.zeros(n_folds)

    cv_splits = cross_validation_splits(
        len(X),
        n_folds=n_folds,
        seed=seed,
        groups=groups,
        stratified=False,
    )
    for indx, (train_index, test_index) in enumerate(cv_splits):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        metric_[indx] = r2_score(Y_test, preds)

        indx += 1

    output_clf = generate_score(metric_)

    return {
        "clf": {
            metric: output_clf,
        },
        "str": {
            metric: print_score(output_clf),
        },
    }


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 3)) + " +/- " + str(round(score[1], 3))
