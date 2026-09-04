import numpy as np
import pytest

from synthcity.metrics.scores import ScoreEvaluator


def test_failed_metric_without_submetrics_is_materialized():
    scores = ScoreEvaluator()

    scores.add_multiple(
        "stats.broken_metric",
        {},
        failed=1,
        duration=0.25,
        direction="minimize",
        error="invalid input shape",
        error_type="ValueError",
    )

    report = scores.to_dataframe()

    assert list(report.index) == ["stats.broken_metric"]
    assert np.isnan(report.loc["stats.broken_metric", "mean"])
    assert report.loc["stats.broken_metric", "errors"] == 1
    assert report.loc["stats.broken_metric", "error_types"] == "ValueError"
    assert report.loc["stats.broken_metric", "error_messages"] == "invalid input shape"
    assert report.loc["stats.broken_metric", "direction"] == "minimize"
    assert report.attrs["metric_metadata"] == report.attrs["result_metadata"] == {}


@pytest.mark.parametrize("result", [np.nan, np.inf, -np.inf])
def test_non_finite_metric_result_is_materialized_as_typed_failure(result):
    scores = ScoreEvaluator()

    scores.add(
        "stats.non_finite_metric",
        result,
        failed=0,
        duration=0.25,
        direction="minimize",
    )

    report = scores.to_dataframe()

    assert np.isnan(report.loc["stats.non_finite_metric", "mean"])
    assert report.loc["stats.non_finite_metric", "errors"] == 1
    assert (
        report.loc["stats.non_finite_metric", "error_types"] == "NonFiniteMetricResult"
    )
    assert report.loc["stats.non_finite_metric", "error_messages"] == (
        f"Metric evaluator returned a non-finite result: {result!r}"
    )


def test_metric_exception_is_materialized_as_typed_failure():
    class FailingMetric:
        @staticmethod
        def fqdn():
            return "stats.failing_metric"

        @staticmethod
        def direction():
            return "minimize"

        @staticmethod
        def evaluate():
            raise RuntimeError("metric exploded")

    scores = ScoreEvaluator()
    scores.queue(FailingMetric())
    scores.compute()

    report = scores.to_dataframe()

    assert np.isnan(report.loc["stats.failing_metric", "mean"])
    assert report.loc["stats.failing_metric", "errors"] == 1
    assert report.loc["stats.failing_metric", "error_types"] == "RuntimeError"
    assert report.loc["stats.failing_metric", "error_messages"] == "metric exploded"


def test_repeated_seed_metadata_preserves_protocols_and_stddev():
    scores = ScoreEvaluator()
    values = [0.2, 0.4, 0.8]

    for repeat, value in enumerate(values):
        scores.add(
            "privacy.DomiasMIA_prior.aucroc",
            value,
            failed=0,
            duration=0.1,
            direction="minimize",
        )
        scores.add_result_metadata(
            {
                "privacy.DomiasMIA_prior": {
                    "result_version": "domias-effective-auc-v2",
                    "random_state": repeat,
                    "domias_protocol": {"auc_chance": 0.5},
                }
            },
            repeat=repeat,
        )

    report = scores.to_dataframe()
    metadata = report.attrs["result_metadata"]["privacy.DomiasMIA_prior"]

    assert report.loc["privacy.DomiasMIA_prior.aucroc", "mean"] == np.mean(values)
    assert report.loc["privacy.DomiasMIA_prior.aucroc", "stddev"] == np.std(values)
    assert metadata["result_version"] == "domias-effective-auc-v2"
    assert metadata["repeated_seed_uncertainty"] == {
        "schema_version": "repeated-seed-uncertainty-v1",
        "method": "standard_deviation_across_repeated_seeds_v1",
        "field": "stddev",
        "unit": "evaluation_repeat",
        "seeds": [0, 1, 2],
        "repeat_count": 3,
        "ddof": 0,
        "estimable": True,
    }
    assert [
        entry["metadata"]["random_state"]
        for entry in metadata["repeated_seed_metadata"]
    ] == [0, 1, 2]
    assert all(
        entry["metadata"]["domias_protocol"] == {"auc_chance": 0.5}
        for entry in metadata["repeated_seed_metadata"]
    )
