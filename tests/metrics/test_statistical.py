# stdlib
import sys
from typing import Any, Tuple, Type

# third party
import numpy as np
import pandas as pd
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_iris
from torchvision import datasets

# synthcity absolute
from synthcity.metrics.eval_statistical import (
    AlphaPrecision,
    ChiSquaredTest,
    FrechetInceptionDistance,
    InverseKLDivergence,
    JensenShannonDistance,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    PRDCScore,
    SurvivalKMDistance,
    WassersteinDistance,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    ImageDataLoader,
    SurvivalAnalysisDataLoader,
    create_from_info,
)


def _eval_plugin(
    evaluator_t: Type, X: DataLoader, X_syn: DataLoader, **kwargs: Any
) -> Tuple:
    evaluator = evaluator_t(
        **kwargs,
        use_cache=False,
    )

    syn_score = evaluator.evaluate(X, X_syn)

    sz = len(X_syn)
    X_rnd = create_from_info(
        pd.DataFrame(np.random.uniform(size=(sz, len(X.columns))), columns=X.columns),
        X.info(),
    )
    rnd_score = evaluator.evaluate(
        X,
        X_rnd,
    )

    def_score = evaluator.evaluate_default(X, X_rnd)
    assert isinstance(def_score, float)

    return syn_score, rnd_score


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_kl_div(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(InverseKLDivergence, Xloader, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] > rnd_score[key]

    assert InverseKLDivergence.name() == "inv_kl_divergence"
    assert InverseKLDivergence.type() == "stats"
    assert InverseKLDivergence.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_kolmogorov_smirnov_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(KolmogorovSmirnovTest, Xloader, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] > rnd_score[key]

    assert KolmogorovSmirnovTest.name() == "ks_test"
    assert KolmogorovSmirnovTest.type() == "stats"
    assert KolmogorovSmirnovTest.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_chi_squared_test(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(ChiSquaredTest, Xloader, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] > rnd_score[key]

    assert ChiSquaredTest.name() == "chi_squared_test"
    assert ChiSquaredTest.type() == "stats"
    assert ChiSquaredTest.direction() == "maximize"


@pytest.mark.parametrize("kernel", ["linear", "rbf", "polynomial"])
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_maximum_mean_discrepancy(kernel: str, test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(MaximumMeanDiscrepancy, Xloader, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] < rnd_score[key]

    assert MaximumMeanDiscrepancy.name() == "max_mean_discrepancy"
    assert MaximumMeanDiscrepancy.type() == "stats"
    assert MaximumMeanDiscrepancy.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_avg_jensenshannon_distance(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(JensenShannonDistance, Xloader, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] < rnd_score[key]

    assert JensenShannonDistance.name() == "jensenshannon_dist"
    assert JensenShannonDistance.type() == "stats"
    assert JensenShannonDistance.direction() == "minimize"


def test_jensen_shannon_uses_declared_semantics_and_shared_support() -> None:
    real = GenericDataLoader(
        pd.DataFrame(
            {
                "category": ["a", "b", "a", "b"],
                "measurement": [0.0, 1.0, 2.0, 1.0],
                "measurement_2": [0.0, 1.0, 2.0, 1.0],
            }
        ),
        feature_types={
            "category": "categorical",
            "measurement": "continuous",
            "measurement_2": "continuous",
        },
        source_table={
            "category": "demographics",
            "measurement": "labs",
            "measurement_2": "labs",
        },
    )
    synthetic = GenericDataLoader(
        pd.DataFrame(
            {
                "category": ["a", "c", "a", "c"],
                "measurement": [0.0, 2.0, 3.0, 4.0],
                "measurement_2": [0.0, 0.0, 0.0, 0.0],
            }
        ),
        feature_types={
            "category": "categorical",
            "measurement": "continuous",
            "measurement_2": "continuous",
        },
        source_table={
            "category": "demographics",
            "measurement": "labs",
            "measurement_2": "labs",
        },
    )

    evaluator = JensenShannonDistance(n_histogram_bins=2, use_cache=False)
    results = evaluator._evaluate(real, synthetic)
    metadata = evaluator.result_metadata()

    assert results["marginal"] >= 0
    assert results["variable_v2.category"] > 0
    assert results["variable_v2.measurement"] > 0
    assert results["variable_v2.measurement_2"] > 0
    labs_mean = (
        results["variable_v2.measurement"] + results["variable_v2.measurement_2"]
    ) / 2
    assert results["source_table_macro_v2"] == pytest.approx(
        (results["variable_v2.category"] + labs_mean) / 2
    )
    assert results["max_variable_v2"] == pytest.approx(
        max(
            results["variable_v2.category"],
            results["variable_v2.measurement"],
            results["variable_v2.measurement_2"],
        )
    )
    assert metadata["version"] == "jsd_v2"
    assert metadata["variables"]["category"]["feature_type"] == "categorical"
    assert metadata["variables"]["category"]["source_table"] == "demographics"
    assert {item["repr"] for item in metadata["variables"]["category"]["support"]} == {
        "'a'",
        "'b'",
        "'c'",
    }
    assert metadata["variables"]["measurement"]["bin_edges"][-1] == 4.0
    assert metadata["aggregation_contract"]["schema_version"] == "source-table-aggregation-v1"
    assert metadata["aggregation_contract"]["source_table_macro_v2"]["source_tables"] == {
        "demographics": {
            "variables": ["category"],
            "n_variables": 1,
            "mean_distance": results["variable_v2.category"],
        },
        "labs": {
            "variables": ["measurement", "measurement_2"],
            "n_variables": 2,
            "mean_distance": labs_mean,
        },
    }


def test_jensen_shannon_cache_retains_source_table_metadata(tmp_path) -> None:
    real = GenericDataLoader(
        pd.DataFrame({"value": [0.0, 1.0, 2.0]}),
        feature_types={"value": "continuous"},
        source_table={"value": "labs"},
    )
    synthetic = GenericDataLoader(
        pd.DataFrame({"value": [0.0, 2.0, 3.0]}),
        feature_types={"value": "continuous"},
        source_table={"value": "labs"},
    )

    first_evaluator = JensenShannonDistance(workspace=tmp_path)
    first = first_evaluator.evaluate(real, synthetic)
    second_evaluator = JensenShannonDistance(workspace=tmp_path)
    second = second_evaluator.evaluate(real, synthetic)

    assert second == first
    assert second_evaluator.result_metadata()["variables"]["value"]["source_table"] == "labs"
    assert (
        second_evaluator.result_metadata()["aggregation_contract"]["schema_version"]
        == "source-table-aggregation-v1"
    )


def test_jensen_shannon_cache_is_namespaced_by_semantic_settings(tmp_path) -> None:
    real = GenericDataLoader(pd.DataFrame({"value": [0, 1, 2]}))
    synthetic = GenericDataLoader(pd.DataFrame({"value": [0, 2, 3]}))

    JensenShannonDistance(
        workspace=tmp_path,
        feature_types={"value": "continuous"},
        source_table={"value": "labs"},
    ).evaluate(real, synthetic)
    second_evaluator = JensenShannonDistance(
        workspace=tmp_path,
        feature_types={"value": "categorical"},
        source_table={"value": "survey"},
    )
    second_evaluator.evaluate(real, synthetic)

    metadata = second_evaluator.result_metadata()
    assert metadata["variables"]["value"]["feature_type"] == "categorical"
    assert metadata["variables"]["value"]["source_table"] == "survey"
    assert len(list(tmp_path.glob("sc_metric_cache_stats_jensenshannon_dist*"))) == 2


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_wasserstein_distance(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(WassersteinDistance, Xloader, X_gen)

    for key in syn_score:
        assert syn_score[key] > 0
        assert rnd_score[key] > 0
        assert syn_score[key] < rnd_score[key]

    assert WassersteinDistance.name() == "wasserstein_dist"
    assert WassersteinDistance.type() == "stats"
    assert WassersteinDistance.direction() == "minimize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("ctgan")])
def test_evaluate_prdc(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(1000)

    syn_score, rnd_score = _eval_plugin(PRDCScore, Xloader, X_gen)
    for key in [
        "precision",
        "recall",
        "density",
        "coverage",
    ]:
        assert key in syn_score

    for key in syn_score:
        assert syn_score[key] >= 0
        assert rnd_score[key] >= 0
        assert syn_score[key] >= rnd_score[key]

    assert PRDCScore.name() == "prdc"
    assert PRDCScore.type() == "stats"
    assert PRDCScore.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_alpha_precision(test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X["target"] = y
    Xloader = GenericDataLoader(X)

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(len(X))

    syn_score, rnd_score = _eval_plugin(AlphaPrecision, Xloader, X_gen)

    for key in [
        "delta_precision_alpha_OC",
        "delta_coverage_beta_OC",
        "authenticity_OC",
        "delta_precision_alpha_naive",
        "delta_coverage_beta_naive",
        "authenticity_naive",
    ]:
        assert key in syn_score
        assert key in rnd_score

    # fr best method
    assert syn_score["delta_precision_alpha_OC"] > rnd_score["delta_precision_alpha_OC"]
    assert syn_score["authenticity_OC"] < rnd_score["authenticity_OC"]

    # For naive method
    assert (
        syn_score["delta_precision_alpha_naive"]
        > rnd_score["delta_precision_alpha_naive"]
    )
    assert (
        syn_score["delta_coverage_beta_naive"] > rnd_score["delta_coverage_beta_naive"]
    )
    assert syn_score["authenticity_naive"] < rnd_score["authenticity_naive"]

    assert AlphaPrecision.name() == "alpha_precision"
    assert AlphaPrecision.type() == "stats"
    assert AlphaPrecision.direction() == "maximize"


@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluate_survival_km_distance(test_plugin: Plugin) -> None:
    X = load_rossi()
    Xloader = SurvivalAnalysisDataLoader(
        X,
        target_column="arrest",
        time_to_event_column="week",
        time_horizons=[25],
    )

    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(len(X))

    syn_score, rnd_score = _eval_plugin(
        SurvivalKMDistance,
        Xloader,
        X_gen,
        task_type="survival_analysis",
    )

    assert np.abs(syn_score["optimism"]) < np.abs(rnd_score["optimism"])
    assert syn_score["abs_optimism"] < rnd_score["abs_optimism"]
    assert syn_score["sightedness"] < rnd_score["sightedness"]

    assert SurvivalKMDistance.name() == "survival_km_distance"
    assert SurvivalKMDistance.type() == "stats"
    assert SurvivalKMDistance.direction() == "minimize"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_image_support() -> None:
    dataset = datasets.MNIST(".", download=True)

    X1 = ImageDataLoader(dataset).sample(100)
    X2 = ImageDataLoader(dataset).sample(100)

    for evaluator in [
        AlphaPrecision,
        ChiSquaredTest,
        InverseKLDivergence,
        JensenShannonDistance,
        KolmogorovSmirnovTest,
        MaximumMeanDiscrepancy,
        PRDCScore,
        WassersteinDistance,
    ]:
        score = evaluator().evaluate(X1, X2)
        assert isinstance(score, dict), evaluator
        for k in score:
            assert score[k] >= 0, evaluator
            assert not np.isnan(score[k]), evaluator

    # FID needs a bigger sample
    X1 = ImageDataLoader(dataset).sample(10000)
    X2 = ImageDataLoader(dataset).sample(10000)
    for evaluator in [
        FrechetInceptionDistance,
    ]:
        score = evaluator().evaluate(X1, X2)
        print(score)
        assert isinstance(score, dict), evaluator
        for k in score:
            assert score[k] >= 0, evaluator
            assert not np.isnan(score[k]), evaluator
