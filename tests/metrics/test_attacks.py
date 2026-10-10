# stdlib
from typing import Type

# third party
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

    evaluator = evaluator_t()

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )["mean"]
    assert score > 0.5
    # A row-copying sampler can be predicted perfectly now that the share
    # divides by the number of rows (it divided by rows + 1).
    assert score <= 1

    # Random noise
    test_plugin = Plugins().get("uniform_sampler")
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    score = evaluator.evaluate(
        Xloader,
        X_gen,
    )["mean"]
    assert score < 1

    assert evaluator.type() == "attack"
    assert evaluator.direction() == "minimize"


def test_data_leakage_skips_continuous_and_tolerates_unseen_categories() -> None:
    # third party
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 400
    cat = rng.integers(0, 3, n)
    X = pd.DataFrame(
        {
            "a": cat + rng.normal(0, 0.1, n),
            "b": rng.normal(0, 1, n),
            "secret_cat": cat,
            "secret_num": rng.normal(0, 1, n),
        }
    )
    real = GenericDataLoader(X, sensitive_features=["secret_cat", "secret_num"])
    evaluator = DataLeakageXGB()

    # A continuous secret is skipped, not scored by exact equality (~0), so
    # the mean is the categorical secret's accuracy alone.
    copy_score = evaluator.evaluate(real, real)["mean"]
    cat_only = evaluator.evaluate(
        GenericDataLoader(X, sensitive_features=["secret_cat"]),
        GenericDataLoader(X, sensitive_features=["secret_cat"]),
    )["mean"]
    assert copy_score == pytest.approx(cat_only)
    assert copy_score > 0.9

    # Synthetic data that never produces category 2: the real rows with
    # category 2 count as wrong instead of making the metric raise.
    syn = X[X["secret_cat"] != 2].reset_index(drop=True)
    score = evaluator.evaluate(
        real, GenericDataLoader(syn, sensitive_features=["secret_cat", "secret_num"])
    )["mean"]
    assert score <= 1 - (X["secret_cat"] == 2).mean() + 1e-9
