# stdlib
import copy
from typing import Any, Callable, Dict, List

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# synthcity absolute
from synthcity.plugins.core.models.survival_analysis.loader import (
    generate_dataset_for_horizon,
)
from synthcity.plugins.core.models.survival_analysis.metrics import (
    evaluate_brier_score,
    evaluate_c_index,
    generate_score,
    print_score,
)
from synthcity.utils.dataframe import constant_columns
from synthcity.utils.evaluation import cross_validation_splits, train_test_indices


def evaluate_survival_model(
    estimator: Any,
    X: pd.DataFrame,
    T: pd.DataFrame,
    Y: pd.DataFrame,
    time_horizons: List,
    n_folds: int = 3,
    metrics: List[str] = ["c_index", "brier_score", "aucroc"],
    random_state: int = 0,
    pretrained: bool = False,
    groups: Any = None,
) -> Dict:
    """Helper for evaluating survival analysis tasks.

    Args:
        model_name: str
            The model to evaluate
        model_args: dict
            The model args to use
        X: DataFrame
            The covariates
        T: Series
            time to event
        Y: Series
            event or censored
        time_horizons: list
            Horizons where to evaluate the performance.
        n_folds: int
            Number of folds for cross validation
        metrics: list
            Available metrics: "c_index", "brier_score", "aucroc"
        random_state: int
            Random seed
        pretrained: bool
            If the estimator was trained or not
    """

    supported_metrics = ["c_index", "brier_score", "aucroc"]
    results = {}

    for metric in metrics:
        if metric not in supported_metrics:
            raise ValueError(f"Metric {metric} not supported")

        results[metric] = np.zeros(n_folds)

    def _get_surv_metrics(
        cv_idx: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        T_train: pd.DataFrame,
        T_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        time_horizons: list,
        groups_train: Any = None,
    ) -> tuple:
        train_max = T_train.max()
        T_test[T_test > train_max] = train_max

        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)

            constant_cols = constant_columns(X_train)
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols)

            if groups_train is None:
                model.fit(X_train, T_train, Y_train)
            else:
                model.fit(X_train, T_train, Y_train, groups=groups_train)

        pred = model.predict(X_test, time_horizons).to_numpy()

        c_index = 0.0
        brier_score = 0.0

        for k in range(len(time_horizons)):
            eval_horizon = min(time_horizons[k], np.max(T_test) - 1)

            def get_score(fn: Callable) -> float:
                return fn(
                    T_train,
                    Y_train,
                    pred[:, k],
                    T_test,
                    Y_test,
                    eval_horizon,
                ) / (len(time_horizons))

            c_index += get_score(evaluate_c_index)
            brier_score += get_score(evaluate_brier_score)

        return c_index, brier_score

    def _get_clf_metrics(
        cv_idx: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        T_train: pd.DataFrame,
        T_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        time_horizons: list,
        groups_train: Any = None,
    ) -> float:
        cv_idx = 0

        train_max = T_train.max()
        T_test[T_test > train_max] = train_max

        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)

            constant_cols = constant_columns(X_train)
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols)

            if groups_train is None:
                model.fit(X_train, T_train, Y_train)
            else:
                model.fit(X_train, T_train, Y_train, groups=groups_train)

        pred = model.predict(X_test, time_horizons).to_numpy()

        local_preds = pd.DataFrame(pred[:, k]).squeeze()

        return roc_auc_score(Y_test, local_preds) / (len(time_horizons))

    def horizon_groups(horizon_days: int) -> Any:
        if groups is None:
            return None
        group_values = list(groups)
        event_horizon = ((np.asarray(Y) == 1) & (np.asarray(T) <= horizon_days)) | (
            (np.asarray(Y) == 0) & (np.asarray(T) > horizon_days)
        )
        censored_event_horizon = (np.asarray(Y) == 1) & (
            np.asarray(T) > horizon_days
        )
        selected = np.concatenate(
            [np.flatnonzero(event_horizon), np.flatnonzero(censored_event_horizon)]
        )
        if len(group_values) != len(X):
            raise ValueError(
                f"groups length {len(group_values)} does not match data length {len(X)}"
            )
        return [group_values[index] for index in selected]

    group_values = None
    if groups is not None:
        group_list = list(groups)
        if len(group_list) != len(X):
            raise ValueError(
                f"groups length {len(group_list)} does not match data length {len(X)}"
            )
        group_values = np.empty(len(group_list), dtype=object)
        group_values[:] = group_list

    if n_folds == 1:
        cv_idx = 0
        train_index, test_index = train_test_indices(
            len(X),
            train_size=0.75,
            seed=random_state,
            y=Y,
            groups=groups,
            stratified=groups is None,
        )
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        T_train, T_test = T.iloc[train_index], T.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

        c_index, brier_score = _get_surv_metrics(
            cv_idx,
            X_train,
            X_test,
            T_train,
            T_test,
            Y_train,
            Y_test,
            local_time_horizons,
            groups_train=None if group_values is None else group_values[train_index],
        )
        for metric in metrics:
            if metric == "c_index":
                results[metric][cv_idx] = c_index
            elif metric == "brier_score":
                results[metric][cv_idx] = brier_score

        if "aucroc" in metrics:
            for k in range(len(time_horizons)):
                cv_idx = 0

                X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
                    X, T, Y, time_horizons[k]
                )
                horizon_group_values = horizon_groups(time_horizons[k])
                train_index, test_index = train_test_indices(
                    len(X_horizon),
                    train_size=0.75,
                    seed=random_state,
                    y=Y_horizon,
                    groups=horizon_group_values,
                    stratified=horizon_group_values is None,
                )
                X_train = X_horizon.iloc[train_index]
                X_test = X_horizon.iloc[test_index]
                T_train = T_horizon.iloc[train_index]
                T_test = T_horizon.iloc[test_index]
                Y_train = Y_horizon.iloc[train_index]
                Y_test = Y_horizon.iloc[test_index]

                metric = "aucroc"

                results[metric][cv_idx] += _get_clf_metrics(
                    cv_idx,
                    X_train,
                    X_test,
                    T_train,
                    T_test,
                    Y_train,
                    Y_test,
                    local_time_horizons,
                    groups_train=(
                        None
                        if horizon_group_values is None
                        else [horizon_group_values[index] for index in train_index]
                    ),
                )

    else:
        cv_splits = cross_validation_splits(
            len(X),
            y=Y,
            n_folds=n_folds,
            seed=random_state,
            groups=groups,
            stratified=True,
        )

        cv_idx = 0
        for train_index, test_index in cv_splits:

            X_train = X.iloc[train_index]
            Y_train = Y.iloc[train_index]
            T_train = T.iloc[train_index]
            X_test = X.iloc[test_index]
            Y_test = Y.iloc[test_index]
            T_test = T.iloc[test_index]

            local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

            c_index, brier_score = _get_surv_metrics(
                cv_idx,
                X_train,
                X_test,
                T_train,
                T_test,
                Y_train,
                Y_test,
                local_time_horizons,
                groups_train=None if group_values is None else group_values[train_index],
            )
            for metric in metrics:
                if metric == "c_index":
                    results[metric][cv_idx] = c_index
                elif metric == "brier_score":
                    results[metric][cv_idx] = brier_score

            cv_idx += 1

        if "aucroc" in metrics:
            for k in range(len(time_horizons)):
                cv_idx = 0

                X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
                    X, T, Y, time_horizons[k]
                )
                horizon_group_values = horizon_groups(time_horizons[k])
                horizon_splits = cross_validation_splits(
                    len(X_horizon),
                    y=Y_horizon,
                    n_folds=n_folds,
                    seed=random_state,
                    groups=horizon_group_values,
                    stratified=True,
                )
                for train_index, test_index in horizon_splits:

                    X_train = X_horizon.iloc[train_index]
                    Y_train = Y_horizon.iloc[train_index]
                    T_train = T_horizon.iloc[train_index]
                    X_test = X_horizon.iloc[test_index]
                    Y_test = Y_horizon.iloc[test_index]
                    T_test = T_horizon.iloc[test_index]

                    metric = "aucroc"

                    results[metric][cv_idx] += _get_clf_metrics(
                        cv_idx,
                        X_train,
                        X_test,
                        T_train,
                        T_test,
                        Y_train,
                        Y_test,
                        local_time_horizons,
                        groups_train=(
                            None
                            if horizon_group_values is None
                            else [horizon_group_values[index] for index in train_index]
                        ),
                    )

                    cv_idx += 1

    output: dict = {
        "clf": {},
        "str": {},
    }

    for metric in metrics:
        output["clf"][metric] = generate_score(results[metric])
        output["str"][metric] = print_score(output["clf"][metric])

    return output
