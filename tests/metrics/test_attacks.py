# stdlib
from typing import Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

# synthcity absolute
from synthcity.metrics.eval_attacks import (
    DataLeakageLinear,
    DataLeakageMLP,
    DataLeakageXGB,
)
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader


def _install_controlled_attack_models(monkeypatch):
    from synthcity.metrics import eval_attacks

    class ControlledClassifier:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, data, target):
            del data, target
            return self

        def predict(self, data):
            return (np.asarray(data)[:, 0] >= 4).astype(int)

    class ControlledRegressor:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, data, target):
            del data, target
            return self

        def predict(self, data):
            return np.asarray(data)[:, 0] * 10 + 10

    class ControlledMLP:
        def __init__(self, task_type="classification", **kwargs):
            del kwargs
            self.task_type = task_type

        def fit(self, data, target):
            del data, target
            return self

        def predict(self, data):
            values = np.asarray(data)
            if self.task_type == "regression":
                return values[:, 0] * 10 + 10
            return (values[:, 0] >= 4).astype(int)

    monkeypatch.setattr(eval_attacks, "LogisticRegression", ControlledClassifier)
    monkeypatch.setattr(eval_attacks, "XGBClassifier", ControlledClassifier)
    monkeypatch.setattr(eval_attacks, "LinearRegression", ControlledRegressor)
    monkeypatch.setattr(eval_attacks, "XGBRegressor", ControlledRegressor)
    monkeypatch.setattr(eval_attacks, "MLP", ControlledMLP)


@pytest.mark.parametrize("reduction", ["mean", "max", "min"])
@pytest.mark.parametrize(
    "evaluator_t",
    [
        DataLeakageLinear,
        DataLeakageXGB,
        DataLeakageMLP,
    ],
)
def test_reduction(reduction: str, evaluator_t: Type) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    # Sampler
    test_plugin = Plugins().get("dummy_sampler")
    Xloader = GenericDataLoader(X, sensitive_features=["sex"])

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(10)

    evaluator = evaluator_t(
        reduction=reduction,
        use_cache=False,
        quasi_identifier_columns=["age", "bmi", "bp"],
        sensitive_target_types={"sex": "continuous"},
    )

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )

    assert reduction in score

    def_score = evaluator.evaluate_default(Xloader, X_gen)

    assert def_score == score[reduction]


@pytest.mark.parametrize(
    "evaluator_t",
    [
        DataLeakageLinear,
        DataLeakageXGB,
        DataLeakageMLP,
    ],
)
def test_evaluate_sensitive_data_leakage(evaluator_t: Type) -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    # Sampler
    test_plugin = Plugins().get("dummy_sampler")
    Xloader = GenericDataLoader(X, sensitive_features=["sex"])

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    evaluator = evaluator_t(
        quasi_identifier_columns=["age", "bmi", "bp"],
        sensitive_target_types={"sex": "continuous"},
    )

    result = evaluator.evaluate(
        Xloader,
        X_gen,
    )
    assert 0 <= result["baseline_adjusted_advantage_v2"] <= 1
    assert result["normalized_mae_v2.sex"] >= 0

    # Random noise
    test_plugin = Plugins().get("uniform_sampler")
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    result = evaluator.evaluate(
        Xloader,
        X_gen,
    )
    assert 0 <= result["baseline_adjusted_advantage_v2"] <= 1
    assert result["normalized_mae_v2.sex"] >= 0

    assert evaluator.type() == "attack"
    assert evaluator.direction() == "minimize"


def test_categorical_attack_emits_baseline_adjusted_per_target_outputs() -> None:
    frame = pd.DataFrame(
        {
            "quasi_id": [0, 1, 0, 1, 0, 1, 0, 1],
            "secret": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    loader = GenericDataLoader(frame, sensitive_features=["secret"])
    evaluator = DataLeakageLinear(
        use_cache=False,
        quasi_identifier_columns=["quasi_id"],
        sensitive_target_types={"secret": "categorical"},
    )

    result = evaluator.evaluate(loader, loader)

    assert result["raw_accuracy.secret"] == pytest.approx(1.0)
    assert result["majority_baseline.secret"] == pytest.approx(0.5)
    assert result["balanced_accuracy.secret"] == pytest.approx(1.0)
    assert result["baseline_adjusted_advantage_v2"] == pytest.approx(1.0)
    assert result["uncertainty_v2.secret"] >= 0
    metadata = evaluator.result_metadata()
    assert metadata["quasi_identifier_hash"]
    assert metadata["target_protocol"]["secret"] == {
        "type": "categorical",
        "cardinality": 2,
    }
    assert metadata["population_roles"]["reference"] == loader.hash()
    assert metadata["population_role_metadata"]["reference"] == {
        "role": "reference",
        "hash": loader.hash(),
        "group_ids_present": False,
    }
    assert metadata["attacker_configuration"]["classifier"]["class"].endswith(
        "LogisticRegression"
    )
    assert metadata["uncertainty"] == {
        "method": "paired_bootstrap_standard_error_v2",
        "rounds": 200,
        "unit": "evaluation rows",
        "baseline": "fixed_full_evaluation_protocol_baseline_v1",
    }
    assert metadata["protocol_digest"]
    assert metadata["worst_target_selection"] == {
        "metric": "baseline_adjusted_advantage_v2",
        "direction": "maximize",
        "selected_target": "secret",
        "target_risks": {"secret": 1.0},
        "tie_break": "lexicographically_largest_target_name",
    }


@pytest.mark.parametrize(
    "evaluator_t",
    [
        DataLeakageLinear,
        DataLeakageXGB,
        DataLeakageMLP,
    ],
)
@pytest.mark.parametrize(
    ("classification_score", "expected_score", "expected_baseline"),
    [
        ("balanced_accuracy", 0.9, 0.5),
        ("macro_f1", 7 / 9, 5 / 11),
    ],
)
def test_categorical_attack_family_and_score_policy_matrix(
    monkeypatch,
    evaluator_t: Type,
    classification_score: str,
    expected_score: float,
    expected_baseline: float,
) -> None:
    _install_controlled_attack_models(monkeypatch)
    frame = pd.DataFrame(
        {
            "quasi_id": list(range(6)),
            "secret": [0, 0, 0, 0, 0, 1],
        }
    )
    loader = GenericDataLoader(frame, sensitive_features=["secret"])
    evaluator = evaluator_t(
        use_cache=False,
        random_state=19,
        quasi_identifier_columns=["quasi_id"],
        sensitive_target_types={"secret": "categorical"},
        classification_score=classification_score,
    )

    result = evaluator.evaluate(loader, loader)
    metadata = evaluator.result_metadata()

    assert result["raw_accuracy.secret"] == pytest.approx(5 / 6)
    assert result["majority_baseline.secret"] == pytest.approx(5 / 6)
    assert result["chance_baseline.secret"] == pytest.approx(0.5)
    assert result["classification_score.secret"] == pytest.approx(expected_score)
    assert result["classification_baseline.secret"] == pytest.approx(expected_baseline)
    assert result["baseline_adjusted_advantage_v2"] == pytest.approx(
        (expected_score - expected_baseline) / (1 - expected_baseline)
    )
    assert result["n_eval.secret"] == pytest.approx(6)
    assert result["uncertainty_v2.secret"] >= 0
    assert metadata["classification_score"] == classification_score
    assert metadata["random_state"] == 19
    assert metadata["target_protocol"]["secret"] == {
        "type": "categorical",
        "cardinality": 2,
    }
    assert metadata["worst_target_selection"]["selected_target"] == "secret"

    repeated_evaluator = evaluator_t(
        use_cache=False,
        random_state=19,
        quasi_identifier_columns=["quasi_id"],
        sensitive_target_types={"secret": "categorical"},
        classification_score=classification_score,
    )
    assert repeated_evaluator.evaluate(loader, loader) == result


@pytest.mark.parametrize(
    "evaluator_t",
    [
        DataLeakageLinear,
        DataLeakageXGB,
        DataLeakageMLP,
    ],
)
def test_mixed_target_attack_matrix_preserves_typed_outputs_and_worst_target(
    monkeypatch,
    evaluator_t: Type,
) -> None:
    _install_controlled_attack_models(monkeypatch)
    frame = pd.DataFrame(
        {
            "quasi_id": list(range(6)),
            "secret": [0, 0, 0, 0, 0, 1],
            "income": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    loader = GenericDataLoader(
        frame,
        sensitive_features=["secret", "income"],
    )
    evaluator = evaluator_t(
        use_cache=False,
        random_state=23,
        quasi_identifier_columns=["quasi_id"],
        sensitive_target_types={
            "secret": "categorical",
            "income": "continuous",
        },
    )

    result = evaluator.evaluate(loader, loader)
    metadata = evaluator.result_metadata()

    assert "classification_score.secret" in result
    assert "disclosure_risk_v2.income" in result
    assert "disclosure_risk_v2.secret" not in result
    assert "classification_score.income" not in result
    assert result["baseline_adjusted_advantage_v2.secret"] == pytest.approx(0.8)
    assert result["normalized_mae_v2.income"] == pytest.approx(0.0)
    assert result["disclosure_risk_v2.income"] == pytest.approx(1.0)
    assert result["baseline_mae.income"] == pytest.approx(15.0)
    assert result["baseline_adjusted_advantage_v2"] == pytest.approx(1.0)
    assert result["uncertainty_v2.secret"] >= 0
    assert result["uncertainty_v2.income"] >= 0
    assert metadata["target_protocol"] == {
        "secret": {"type": "categorical", "cardinality": 2},
        "income": {"type": "continuous", "cardinality": None},
    }
    assert metadata["quasi_identifier_columns"] == ["quasi_id"]
    assert metadata["worst_target_selection"]["selected_target"] == "income"
    assert metadata["worst_target_selection"]["target_risks"]["income"] == pytest.approx(1.0)


def test_attack_uncertainty_resamples_whole_groups() -> None:
    evaluator = DataLeakageLinear(use_cache=False, random_state=23)
    target = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    predictions = np.asarray([0, 1, 1, 0, 0, 1, 0, 1])
    groups = np.asarray(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"])

    row_uncertainty = evaluator._bootstrap_risk_uncertainty(
        "secret",
        "categorical",
        target,
        predictions,
        0.5,
    )
    group_uncertainty = evaluator._bootstrap_risk_uncertainty(
        "secret",
        "categorical",
        target,
        predictions,
        0.5,
        groups=groups,
    )

    assert group_uncertainty != pytest.approx(row_uncertainty)
    assert group_uncertainty >= 0


def test_attack_protocol_records_group_uncertainty_unit() -> None:
    frame = pd.DataFrame(
        {
            "quasi_id": [0, 1] * 4,
            "secret": [0, 1] * 4,
        }
    )
    loader = GenericDataLoader(
        frame,
        sensitive_features=["secret"],
        group_ids=["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"],
    )
    evaluator = DataLeakageLinear(
        use_cache=False,
        random_state=23,
        quasi_identifier_columns=["quasi_id"],
        sensitive_target_types={"secret": "categorical"},
    )

    evaluator.evaluate(loader, loader)

    uncertainty = evaluator.result_metadata()["uncertainty"]
    assert uncertainty["method"] == "paired_bootstrap_standard_error_v2"
    assert uncertainty["unit"] == "evaluation groups when group_ids are present"
    assert uncertainty["resampling_unit"] == "evaluation_groups"
    assert uncertainty["group_count"] == 4


def test_categorical_attack_uses_configured_macro_f1_score() -> None:
    frame = pd.DataFrame(
        {
            "quasi_id": [0, 1, 2, 3, 4, 5],
            "secret": [0, 0, 0, 0, 0, 1],
        }
    )
    loader = GenericDataLoader(frame, sensitive_features=["secret"])
    evaluator = DataLeakageLinear(
        use_cache=False,
        quasi_identifier_columns=["quasi_id"],
        sensitive_target_types={"secret": "categorical"},
        classification_score="macro_f1",
    )

    result = evaluator.evaluate(loader, loader)

    assert result["classification_score.secret"] == pytest.approx(
        result["macro_f1.secret"]
    )
    assert result["classification_baseline.secret"] == pytest.approx(
        result["macro_f1.secret"]
    )
    assert evaluator.result_metadata()["classification_score"] == "macro_f1"


def test_attack_records_deterministic_worst_target_selection() -> None:
    frame = pd.DataFrame(
        {
            "quasi_id": list(range(8)),
            "strong_secret": [0, 0, 0, 0, 1, 1, 1, 1],
            "weak_secret": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    loader = GenericDataLoader(
        frame,
        sensitive_features=["strong_secret", "weak_secret"],
    )
    evaluator = DataLeakageLinear(
        use_cache=False,
        random_state=17,
        quasi_identifier_columns=["quasi_id"],
        sensitive_target_types={
            "strong_secret": "categorical",
            "weak_secret": "categorical",
        },
    )

    result = evaluator.evaluate(loader, loader)
    metadata = evaluator.result_metadata()

    assert result["baseline_adjusted_advantage_v2"] == pytest.approx(
        result["baseline_adjusted_advantage_v2.strong_secret"]
    )
    assert metadata["worst_target_selection"]["selected_target"] == "strong_secret"
    assert set(metadata["worst_target_selection"]["target_risks"]) == {
        "strong_secret",
        "weak_secret",
    }
    assert result["uncertainty_v2.strong_secret"] >= 0
    assert result["uncertainty_v2.weak_secret"] >= 0

    repeated_evaluator = DataLeakageLinear(
        use_cache=False,
        random_state=17,
        quasi_identifier_columns=["quasi_id"],
        sensitive_target_types={
            "strong_secret": "categorical",
            "weak_secret": "categorical",
        },
    )
    repeated_result = repeated_evaluator.evaluate(loader, loader)
    assert repeated_result["uncertainty_v2.strong_secret"] == pytest.approx(
        result["uncertainty_v2.strong_secret"]
    )
    assert repeated_evaluator.result_metadata()["protocol_digest"] == metadata[
        "protocol_digest"
    ]


def test_attack_requires_explicit_quasi_identifiers() -> None:
    frame = pd.DataFrame({"feature": [0.0, 1.0] * 5, "secret": [0, 1] * 5})
    loader = GenericDataLoader(frame, sensitive_features=["secret"])

    with pytest.raises(ValueError, match="explicit quasi_identifier_columns"):
        DataLeakageLinear(use_cache=False).evaluate(loader, loader)


def test_attack_rejects_sensitive_target_as_quasi_identifier() -> None:
    frame = pd.DataFrame({"feature": [0.0, 1.0] * 5, "secret": [0, 1] * 5})
    loader = GenericDataLoader(frame, sensitive_features=["secret"])
    evaluator = DataLeakageLinear(
        use_cache=False,
        quasi_identifier_columns=["secret"],
        sensitive_target_types={"secret": "categorical"},
    )

    with pytest.raises(ValueError, match="exclude sensitive target columns"):
        evaluator.evaluate(loader, loader)
