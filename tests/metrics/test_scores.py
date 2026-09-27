import warnings

import numpy as np
import pytest

from synthcity.metrics.scores import ScoreEvaluator


def test_dataframe_aggregation_preserves_order_and_schema_without_concat_warning():
    scores = ScoreEvaluator()
    scores.add_multiple(
        "stats.failed_first",
        {},
        failed=1,
        duration=0.25,
        direction="minimize",
        error="first failure",
        error_type="ValueError",
    )
    scores.add("stats.success", 0.75, failed=0, duration=0.5, direction="maximize")
    scores.add_multiple(
        "stats.failed_last",
        {},
        failed=1,
        duration=0.75,
        direction="minimize",
        error="last failure",
        error_type="RuntimeError",
    )

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        report = scores.to_dataframe()

    assert not [
        warning
        for warning in warnings_record
        if issubclass(warning.category, FutureWarning)
        and "DataFrame concatenation with empty or all-NA entries"
        in str(warning.message)
    ]
    assert list(report.index) == [
        "stats.failed_first",
        "stats.success",
        "stats.failed_last",
    ]
    assert list(report.columns) == [
        "min",
        "max",
        "mean",
        "stddev",
        "median",
        "iqr",
        "rounds",
        "errors",
        "error_types",
        "error_messages",
        "durations",
        "direction",
    ]
    assert report.loc["stats.success", "mean"] == pytest.approx(0.75)
    assert report.loc["stats.success", "rounds"] == 1
    assert report.loc["stats.success", "errors"] == 0
    assert np.isnan(report.loc["stats.failed_first", "mean"])
    assert report.loc["stats.failed_first", "errors"] == 1
    assert report.loc["stats.failed_first", "error_types"] == "ValueError"
    assert report.loc["stats.failed_first", "error_messages"] == "first failure"
    assert np.isnan(report.loc["stats.failed_last", "mean"])
    assert report.loc["stats.failed_last", "error_types"] == "RuntimeError"
    assert report.dtypes.astype(str).to_dict() == {
        "min": "float64",
        "max": "float64",
        "mean": "float64",
        "stddev": "float64",
        "median": "float64",
        "iqr": "float64",
        "rounds": "object",
        "errors": "object",
        "error_types": "object",
        "error_messages": "object",
        "durations": "float64",
        "direction": "object",
    }


def test_dataframe_all_missing_scores_keep_object_stat_dtypes_without_concat_warning():
    scores = ScoreEvaluator()
    scores.add_multiple(
        "stats.failed",
        {},
        failed=1,
        duration=0.25,
        direction="minimize",
    )

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        report = scores.to_dataframe()

    assert not [
        warning
        for warning in warnings_record
        if issubclass(warning.category, FutureWarning)
        and "DataFrame concatenation with empty or all-NA entries"
        in str(warning.message)
    ]
    assert list(report.index) == ["stats.failed"]
    assert np.isnan(report.loc["stats.failed", "mean"])
    assert report.loc["stats.failed", "errors"] == 1
    assert report.dtypes.astype(str).to_dict() == {
        "min": "object",
        "max": "object",
        "mean": "object",
        "stddev": "object",
        "median": "object",
        "iqr": "object",
        "rounds": "object",
        "errors": "object",
        "error_types": "object",
        "error_messages": "object",
        "durations": "float64",
        "direction": "object",
    }


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
