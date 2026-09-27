# stdlib
import hashlib
import json
import platform
from typing import Any, Dict, Mapping

# third party
import numpy as np
import pandas as pd
from pydantic import validate_call
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

# synthcity absolute
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.mlp import MLP
from synthcity.utils.serialization import load_from_file, save_to_file

ATTACK_RESULT_VERSION = "attribute-inference-v4"
ATTACK_UNCERTAINTY_METHOD = "paired_bootstrap_standard_error_v2"
ATTACK_UNCERTAINTY_ROUNDS = 200


class AttackEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.AttackEvaluator
        :parts: 1

    Evaluating the risk of attribute inference attack.

    This class evaluates the risk of a type of privacy attack, known as attribute inference attack.
    In this setting, the attacker has access to the synthetic dataset as well as partial information about the real data
    (quasi-identifiers). The attacker seeks to uncover the sensitive attributes of the real data using these two pieces
    of information.
    """

    def __init__(
        self,
        quasi_identifier_columns: list[str] | None = None,
        sensitive_target_types: Mapping[str, str] | None = None,
        classification_score: str = "balanced_accuracy",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._quasi_identifier_columns = tuple(quasi_identifier_columns or ())
        self._sensitive_target_types = dict(sensitive_target_types or {})
        if classification_score not in {"balanced_accuracy", "macro_f1"}:
            raise ValueError(
                "classification_score must be 'balanced_accuracy' or 'macro_f1', "
                f"got {classification_score!r}"
            )
        self._classification_score = classification_score
        self._resolved_quasi_identifier_columns: tuple[str, ...] = ()
        self._protocol_metadata: dict[str, Any] = {}
        invalid_types = set(self._sensitive_target_types.values()) - {
            "categorical",
            "continuous",
        }
        if invalid_types:
            raise ValueError(f"Invalid sensitive target type(s): {sorted(invalid_types)}")

    @staticmethod
    def _qualified_name(value: Any) -> str:
        return (
            f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        )

    @staticmethod
    def _stable_parameters(parameters: Mapping[str, Any]) -> dict[str, str]:
        return {str(key): repr(parameters[key]) for key in sorted(parameters)}

    def _record_worst_target_selection(self, risk_scores: Mapping[str, float]) -> None:
        if not risk_scores:
            return
        selected_target = max(
            risk_scores,
            key=lambda target: (float(risk_scores[target]), target),
        )
        self._protocol_metadata["worst_target_selection"] = {
            "metric": "baseline_adjusted_advantage_v2",
            "direction": "maximize",
            "selected_target": selected_target,
            "target_risks": {
                target: float(risk_scores[target])
                for target in sorted(risk_scores)
            },
            "tie_break": "lexicographically_largest_target_name",
        }

    def _bootstrap_risk_uncertainty(
        self,
        target_name: str,
        target_type: str,
        test_target: np.ndarray,
        predictions: np.ndarray,
        baseline: float,
        groups: np.ndarray | None = None,
    ) -> float:
        """Estimate risk uncertainty without changing the policy score."""
        sample_count = len(test_target)
        if sample_count < 2:
            return 0.0
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != sample_count:
                raise ValueError(
                    "Attack uncertainty groups must align with evaluation rows: "
                    f"groups={len(groups)}, rows={sample_count}"
                )
            group_codes, unique_groups = pd.factorize(groups, sort=False)
            if len(unique_groups) < 2:
                return 0.0
        seed_material = f"{self._random_state}:{target_name}".encode()
        seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        bootstrap_values = []
        for _ in range(ATTACK_UNCERTAINTY_ROUNDS):
            if groups is None:
                indices = rng.integers(0, sample_count, size=sample_count)
            else:
                sampled_groups = rng.integers(
                    0,
                    len(unique_groups),
                    size=len(unique_groups),
                )
                indices = np.concatenate(
                    [np.flatnonzero(group_codes == group) for group in sampled_groups]
                )
            sampled_target = test_target[indices]
            sampled_predictions = predictions[indices]
            if target_type == "categorical":
                classification_value = (
                    float(
                        balanced_accuracy_score(
                            sampled_target,
                            sampled_predictions,
                        )
                    )
                    if self._classification_score == "balanced_accuracy"
                    else float(
                        f1_score(
                            sampled_target,
                            sampled_predictions,
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
                if baseline < 1.0:
                    bootstrap_values.append(
                        float(np.clip((classification_value - baseline) / (1.0 - baseline), 0.0, 1.0))
                    )
                else:
                    bootstrap_values.append(0.0)
            else:
                sampled_error = float(
                    np.mean(
                        np.abs(
                            sampled_predictions.astype(float)
                            - sampled_target.astype(float)
                        )
                    )
                )
                bootstrap_values.append(
                    float(np.clip(1.0 - sampled_error / baseline, 0.0, 1.0))
                    if baseline > 0
                    else 0.0
                )
        return float(np.std(bootstrap_values, ddof=1))

    def _resolve_quasi_identifiers(self, X_gt: DataLoader, X_syn: DataLoader) -> list[str]:
        columns = list(self._quasi_identifier_columns or X_gt.important_features)
        if not columns:
            raise ValueError(
                f"{self.name()} requires explicit quasi_identifier_columns or "
                "DataLoader.important_features; using every remaining feature is disabled"
            )
        if len(columns) != len(set(columns)):
            raise ValueError(f"Quasi-identifier columns must be unique: {columns}")
        sensitive_overlap = sorted(set(columns) & set(X_gt.sensitive_features))
        if sensitive_overlap:
            raise ValueError(
                "Quasi-identifier columns must exclude sensitive target columns: "
                f"{sensitive_overlap}"
            )
        missing = sorted(set(columns) - set(X_gt.columns) - set(X_syn.columns))
        if missing:
            raise ValueError(f"Quasi-identifier columns are missing from both inputs: {missing}")
        missing_real = sorted(set(columns) - set(X_gt.columns))
        missing_synthetic = sorted(set(columns) - set(X_syn.columns))
        if missing_real or missing_synthetic:
            raise ValueError(
                "Quasi-identifier columns must be present in both inputs; "
                f"missing_real={missing_real}, missing_synthetic={missing_synthetic}"
            )
        self._resolved_quasi_identifier_columns = tuple(columns)
        return columns

    def _resolve_target_type(self, column: str) -> str:
        if column in self._sensitive_target_types:
            return self._sensitive_target_types[column]
        raise ValueError(
            f"{self.name()} requires schema-defined sensitive_target_types for target {column!r}"
        )

    def result_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ATTACK_RESULT_VERSION,
            "quasi_identifier_columns": list(self._resolved_quasi_identifier_columns),
            "quasi_identifier_hash": self._protocol_metadata.get("quasi_identifier_hash"),
            "sensitive_target_types": self._protocol_metadata.get(
                "sensitive_target_types", dict(self._sensitive_target_types)
            ),
            "target_protocol": self._protocol_metadata.get("target_protocol", {}),
            "classification_score": self._classification_score,
            "random_state": self._random_state,
            "population_roles": self._protocol_metadata.get("population_roles", {}),
            "population_role_metadata": self._protocol_metadata.get(
                "population_role_metadata", {}
            ),
            "preprocessing": self._protocol_metadata.get("preprocessing", {}),
            "attacker_configuration": self._protocol_metadata.get(
                "attacker_configuration", {}
            ),
            "uncertainty": self._protocol_metadata.get("uncertainty", {}),
            "protocol_digest": self._protocol_metadata.get("protocol_digest"),
            "worst_target_selection": self._protocol_metadata.get(
                "worst_target_selection", {}
            ),
        }

    @staticmethod
    def type() -> str:
        return "attack"

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def _evaluate_leakage(
        self,
        classifier_template: Any,
        classifier_args: Dict,
        regressor_template: Any,
        regressor_args: Dict,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if len(X_gt.sensitive_features) == 0:
            return {}

        quasi_identifiers = self._resolve_quasi_identifiers(X_gt, X_syn)
        resolved_target_types = {}
        for column in X_gt.sensitive_features:
            if column not in X_syn.columns:
                raise ValueError(f"Sensitive target column {column!r} is missing from synthetic data")
            resolved_target_types[column] = self._resolve_target_type(column)
        target_protocol = {
            column: {
                "type": target_type,
                "cardinality": (
                    int(X_syn[column].nunique(dropna=False))
                    if target_type == "categorical"
                    else None
                ),
            }
            for column, target_type in resolved_target_types.items()
        }
        quasi_identifier_hash = hashlib.sha256(
            json.dumps(list(quasi_identifiers), separators=(",", ":")).encode()
        ).hexdigest()
        attacker_configuration = {
            "classifier": {
                "class": self._qualified_name(classifier_template),
                "parameters": self._stable_parameters(classifier_args),
            },
            "regressor": {
                "class": self._qualified_name(regressor_template),
                "parameters": self._stable_parameters(regressor_args),
            },
        }
        self._protocol_metadata = {
            "quasi_identifier_hash": quasi_identifier_hash,
            "sensitive_target_types": dict(resolved_target_types),
            "target_protocol": target_protocol,
            "population_roles": {
                "reference": X_gt.hash(),
                "synthetic": X_syn.hash(),
            },
            "population_role_metadata": {
                "reference": {
                    "role": "reference",
                    "hash": X_gt.hash(),
                    "group_ids_present": X_gt.group_ids is not None,
                },
                "synthetic": {
                    "role": "synthetic",
                    "hash": X_syn.hash(),
                    "group_ids_present": X_syn.group_ids is not None,
                },
            },
            "preprocessing": {
                "feature_representation": "native_dataloader_values",
                "categorical_target": "label_encoder_v1",
                "continuous_target": "numeric_passthrough_v1",
                "fit_role": "synthetic",
                "evaluation_role": "reference",
            },
            "attacker_configuration": attacker_configuration,
            "uncertainty": {
                "method": ATTACK_UNCERTAINTY_METHOD,
                "rounds": ATTACK_UNCERTAINTY_ROUNDS,
                "unit": (
                    "evaluation groups when group_ids are present"
                    if X_gt.group_ids is not None
                    else "evaluation rows"
                ),
                "baseline": "fixed_full_evaluation_protocol_baseline_v1",
            },
        }
        if X_gt.group_ids is not None:
            self._protocol_metadata["uncertainty"].update(
                {
                    "resampling_unit": "evaluation_groups",
                    "group_count": int(pd.Series(list(X_gt.group_ids)).nunique()),
                }
            )
        protocol_digest = hashlib.sha256(
            json.dumps(
                {
                    "quasi_identifier_hash": quasi_identifier_hash,
                    "target_protocol": target_protocol,
                    "classification_score": self._classification_score,
                    "reduction": self._reduction,
                    "random_state": self._random_state,
                    "attacker_configuration": attacker_configuration,
                    "uncertainty": self._protocol_metadata["uncertainty"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]
        self._protocol_metadata["protocol_digest"] = protocol_digest
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{ATTACK_RESULT_VERSION}_{X_gt.hash()}_{X_syn.hash()}_{protocol_digest}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            cached_results = load_from_file(cache_file)
            cached_risks = {
                column: cached_results[f"baseline_adjusted_advantage_v2.{column}"]
                for column in resolved_target_types
                if f"baseline_adjusted_advantage_v2.{column}" in cached_results
            }
            self._record_worst_target_selection(cached_risks)
            return cached_results

        legacy_scores = []
        risk_scores: dict[str, float] = {}
        results: dict[str, float] = {}
        for col in X_gt.sensitive_features:
            if col not in X_syn.columns:
                raise ValueError(f"Sensitive target column {col!r} is missing from synthetic data")

            target = X_syn[col]
            keys_data = X_syn[quasi_identifiers]
            test_keys_data = X_gt[quasi_identifiers]
            task_type = resolved_target_types[col]
            evaluation_groups = (
                None
                if X_gt.group_ids is None
                else np.asarray(list(X_gt.group_ids), dtype=object)
            )

            if task_type == "categorical":
                encoder = LabelEncoder()
                target = encoder.fit_transform(target)
                target_args = dict(classifier_args)
                if "n_units_in" in target_args:
                    target_args["n_units_in"] = len(quasi_identifiers)
                if "n_units_out" in target_args:
                    target_args["n_units_out"] = len(np.unique(target))
                model = classifier_template(**target_args)
                try:
                    test_target = encoder.transform(X_gt[col])
                except ValueError as exc:
                    raise ValueError(
                        f"Sensitive target {col!r} contains categories absent from synthetic training data"
                    ) from exc
            else:
                target = pd.to_numeric(target, errors="raise").to_numpy(dtype=float)
                target_args = dict(regressor_args)
                if "n_units_in" in target_args:
                    target_args["n_units_in"] = len(quasi_identifiers)
                model = regressor_template(**target_args)
                test_target = pd.to_numeric(X_gt[col], errors="raise").to_numpy(dtype=float)

            try:
                model.fit(keys_data.values, np.asarray(target))
                preds = model.predict(test_keys_data.values)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.name()} attacker failed for sensitive target {col!r} "
                    f"using task_type={task_type!r} and quasi_identifiers={quasi_identifiers!r}"
                ) from exc

            if task_type == "categorical":
                preds = np.asarray(preds)
                raw_accuracy = float(np.mean(preds == np.asarray(test_target)))
                majority_baseline = float(pd.Series(target).value_counts(normalize=True).max())
                chance_baseline = 1.0 / len(encoder.classes_)
                balanced_accuracy = float(
                    balanced_accuracy_score(np.asarray(test_target), preds)
                )
                macro_f1 = float(
                    f1_score(
                        np.asarray(test_target),
                        preds,
                        average="macro",
                        zero_division=0,
                    )
                )
                majority_class = pd.Series(target).mode().iloc[0]
                majority_prediction = np.full(
                    len(test_target),
                    fill_value=majority_class,
                    dtype=np.asarray(test_target).dtype,
                )
                classification_baseline = (
                    float(
                        balanced_accuracy_score(
                            np.asarray(test_target), majority_prediction
                        )
                    )
                    if self._classification_score == "balanced_accuracy"
                    else float(
                        f1_score(
                            np.asarray(test_target),
                            majority_prediction,
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
                classification_value = (
                    balanced_accuracy
                    if self._classification_score == "balanced_accuracy"
                    else macro_f1
                )
                uncertainty = self._bootstrap_risk_uncertainty(
                    col,
                    task_type,
                    np.asarray(test_target),
                    preds,
                    classification_baseline,
                    groups=evaluation_groups,
                )
                advantage = float(
                    np.clip(
                        (classification_value - classification_baseline)
                        / (1.0 - classification_baseline),
                        0.0,
                        1.0,
                    )
                )
                results[f"raw_accuracy.{col}"] = raw_accuracy
                results[f"majority_baseline.{col}"] = majority_baseline
                results[f"chance_baseline.{col}"] = chance_baseline
                results[f"balanced_accuracy.{col}"] = balanced_accuracy
                results[f"macro_f1.{col}"] = macro_f1
                results[f"classification_score.{col}"] = classification_value
                results[f"classification_baseline.{col}"] = classification_baseline
                results[f"baseline_adjusted_advantage_v2.{col}"] = advantage
                results[f"uncertainty_v2.{col}"] = uncertainty
                legacy_scores.append(raw_accuracy)
                risk_scores[col] = advantage
            else:
                preds = np.asarray(preds, dtype=float)
                absolute_error = np.abs(preds - np.asarray(test_target, dtype=float))
                attacker_mae = float(np.mean(absolute_error))
                baseline_prediction = float(np.median(target))
                baseline_mae = float(
                    np.mean(np.abs(np.asarray(test_target, dtype=float) - baseline_prediction))
                )
                normalized_mae = attacker_mae / baseline_mae if baseline_mae > 0 else 0.0
                disclosure_risk = float(np.clip(1.0 - normalized_mae, 0.0, 1.0))
                uncertainty = self._bootstrap_risk_uncertainty(
                    col,
                    task_type,
                    np.asarray(test_target, dtype=float),
                    preds,
                    baseline_mae,
                    groups=evaluation_groups,
                )
                legacy_scores.append(float(np.mean(preds == np.asarray(test_target))))
                risk_scores[col] = disclosure_risk
                results[f"normalized_mae_v2.{col}"] = normalized_mae
                results[f"disclosure_risk_v2.{col}"] = disclosure_risk
                results[f"baseline_mae.{col}"] = baseline_mae
                results[f"uncertainty_v2.{col}"] = uncertainty

            results[f"n_eval.{col}"] = float(len(test_target))

        self._record_worst_target_selection(risk_scores)
        selected_target = self._protocol_metadata["worst_target_selection"]["selected_target"]
        results["baseline_adjusted_advantage_v2"] = risk_scores[selected_target]
        results["legacy_accuracy"] = float(self.reduction()(legacy_scores))
        results[self._reduction] = results["legacy_accuracy"]

        save_to_file(cache_file, results)

        return results

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._reduction]


class DataLeakageMLP(AttackEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageMLP
        :parts: 1

    Data leakage test using a neural net.
    """

    @staticmethod
    def name() -> str:
        return "data_leakage_mlp"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            MLP,
            {
                "task_type": "classification",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 0,
                "random_state": self._random_state,
            },
            MLP,
            {
                "task_type": "regression",
                "n_units_in": X_gt.shape[1] - 1,
                "n_units_out": 1,
                "random_state": self._random_state,
            },
            X_gt,
            X_syn,
        )


class DataLeakageXGB(AttackEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageXGB
        :parts: 1

    Data leakage test using XGBoost
    """

    @staticmethod
    def name() -> str:
        return "data_leakage_xgb"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            XGBClassifier,
            {
                "n_jobs": -1,
                "eval_metric": "logloss",
            },
            XGBRegressor,
            {"n_jobs": -1},
            X_gt,
            X_syn,
        )


class DataLeakageLinear(AttackEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_attacks.DataLeakageLinear
        :parts: 1


    Data leakage test using a linear model
    """

    @staticmethod
    def name() -> str:
        return "data_leakage_linear"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        return self._evaluate_leakage(
            LogisticRegression,
            {"random_state": self._random_state},
            LinearRegression,
            {},
            X_gt,
            X_syn,
        )
