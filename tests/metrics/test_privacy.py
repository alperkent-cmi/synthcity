# stdlib
import sys
from pathlib import Path
from typing import Type

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from torchvision import datasets

# synthcity absolute
from synthcity.metrics.eval_privacy import (
    DeltaPresence,
    DOMIAS_RESULT_VERSION,
    DomiasMIABNAF,
    DomiasMIAKDE,
    DomiasMIAPrior,
    IdentifiabilityScore,
    effective_auc_v2,
    kAnonymization,
    kMap,
    lDiversityDistinct,
    IDENTIFIABILITY_RESULT_VERSION,
    STRUCTURAL_PRIVACY_RESULT_VERSION,
)
from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader, ImageDataLoader


@pytest.mark.parametrize(
    "evaluator_t",
    [
        DeltaPresence,
        kAnonymization,
        kMap,
        lDiversityDistinct,
        IdentifiabilityScore,
        DomiasMIABNAF,
        DomiasMIAKDE,
        DomiasMIAPrior,
    ],
)
@pytest.mark.parametrize("test_plugin", [Plugins().get("dummy_sampler")])
def test_evaluator(evaluator_t: Type, test_plugin: Plugin) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    Xloader = GenericDataLoader(
        X, sensitive_features=["sepal length (cm)", "sepal width (cm)"]
    )
    test_plugin.fit(Xloader)
    X_gen = test_plugin.generate(2 * len(X))

    evaluator = evaluator_t(
        use_cache=False,
    )
    if "DomiasMIA" in evaluator.name():
        X_ref_syn = test_plugin.generate(2 * len(X))
        score = evaluator.evaluate(
            Xloader,
            X_gen,
            Xloader.train(),
            X_ref_syn,
            reference_size=10,
        )
    else:
        score = evaluator.evaluate(Xloader, X_gen)

    for submetric in score:
        assert score[submetric] > 0

    assert evaluator.type() == "privacy"

    if "DomiasMIA" in evaluator.name():
        X_ref_syn = test_plugin.generate(2 * len(X))
        def_score = evaluator.evaluate_default(
            Xloader,
            X_gen,
            Xloader.train(),
            X_ref_syn,
            reference_size=10,
        )
    else:
        def_score = evaluator.evaluate_default(Xloader, X_gen)

    assert isinstance(def_score, (float, int))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux only for faster results")
def test_image_support(tmp_path: Path) -> None:
    dataset = datasets.MNIST(tmp_path, download=True)

    X1 = ImageDataLoader(dataset).sample(100)
    X2 = ImageDataLoader(dataset).sample(100)

    for evaluator in [
        IdentifiabilityScore,
    ]:
        score = evaluator().evaluate(X1, X2)
        assert isinstance(score, dict)
        for k in score:
            assert score[k] >= 0
            assert not np.isnan(score[k])


def test_identifiability_legacy_and_entropy_weighted_modes_are_separate() -> None:
    X, _ = load_iris(return_X_y=True, as_frame=True)
    loader = GenericDataLoader(X)
    evaluator = IdentifiabilityScore(use_cache=False)

    legacy = evaluator._compute_scores(loader, loader)
    legacy_repeat = evaluator._compute_scores(loader, loader)
    weighted = evaluator._compute_scores(
        loader,
        loader,
        weighting="entropy",
        output_suffix="_entropy_weighted",
    )

    assert legacy == legacy_repeat
    assert set(legacy) == {"score"}
    assert set(weighted) == {"score_entropy_weighted"}

    with pytest.raises(ValueError, match="Unknown identifiability weighting"):
        evaluator._compute_scores(loader, loader, weighting="unsupported")


def test_identifiability_entropy_weighting_changes_mixed_entropy_fixture() -> None:
    real = pd.DataFrame(
        {
            "low_entropy": [0, 0, 0, 0, 1, 1, 1, 1],
            "high_entropy": list(range(8)),
        }
    )
    synthetic = pd.DataFrame(
        {
            "low_entropy": [0, 0, 0, 0, 0, 1, 0, 0],
            "high_entropy": [-2, -1, 0, 1, 2, 3, 4, 5],
        }
    )
    evaluator = IdentifiabilityScore(use_cache=False)

    legacy = evaluator._compute_scores(
        GenericDataLoader(real), GenericDataLoader(synthetic)
    )["score"]
    weighted = evaluator._compute_scores(
        GenericDataLoader(real),
        GenericDataLoader(synthetic),
        weighting="entropy",
        output_suffix="_entropy_weighted",
    )["score_entropy_weighted"]

    assert legacy == pytest.approx(0.375)
    assert weighted == pytest.approx(0.625)


def test_identifiability_metadata_versions_calibration_variants(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "first": [0.0, 1.0, 0.0, 1.0],
            "second": [0.0, 0.0, 1.0, 1.0],
        }
    )
    loader = GenericDataLoader(frame)
    evaluator = IdentifiabilityScore(workspace=tmp_path)

    result = evaluator.evaluate(loader, loader)
    metadata = evaluator.result_metadata()

    assert set(result) == {
        "score",
        "score_OC",
        "score_entropy_weighted",
        "score_OC_entropy_weighted",
    }
    assert metadata["result_version"] == IDENTIFIABILITY_RESULT_VERSION
    assert metadata["calibration_only"] is True
    assert metadata["population_roles"]["reference"]["rows"] == len(loader)
    assert metadata["preprocessing"]["feature_weighting_legacy"] == "unit_weights_v1"
    assert metadata["variants"]["score"]["weighting"] == "legacy"
    assert metadata["variants"]["score_entropy_weighted"]["weighting"] == "entropy"
    assert len(list(tmp_path.glob("*identifiability-v2*"))) == 1


def test_structural_privacy_metadata_records_proxy_protocol() -> None:
    frame = pd.DataFrame(
        {
            "quasi_id": list(range(10)),
            "secret": [0, 1] * 5,
        }
    )
    loader = GenericDataLoader(frame, sensitive_features=["secret"])
    evaluator = kAnonymization(
        use_cache=False,
        random_state=7,
        structural_n_clusters=[2, 5],
        structural_min_rows_per_cluster=2,
    )

    evaluator.evaluate(loader, loader)
    metadata = evaluator.result_metadata()

    assert metadata["result_version"] == STRUCTURAL_PRIVACY_RESULT_VERSION
    assert metadata["proxy_label"] == "kmeans_partition_screen_not_formal_guarantee"
    assert metadata["calibration_only"] is True
    assert metadata["calibration_required"] is True
    assert metadata["feature_selection"]["real_columns"] == ["quasi_id"]
    assert metadata["feature_selection"]["sensitive_features_real"] == ["secret"]
    assert metadata["kmeans"]["n_clusters"] == [2, 5]
    assert metadata["kmeans"]["min_rows_per_cluster"] == 2
    assert metadata["kmeans"]["random_state"] == 7
    assert metadata["sample_sizes"] == {"real": 10, "synthetic": 10}


@pytest.mark.parametrize(
    ("evaluator_t", "result_key", "concentrated_relation"),
    [
        (kAnonymization, "syn", "greater"),
        (lDiversityDistinct, "syn", "less"),
        (kMap, "score", "greater"),
        (DeltaPresence, "score", "less"),
    ],
)
def test_structural_proxy_is_sensitive_to_concentrated_synthetic_features(
    evaluator_t: Type,
    result_key: str,
    concentrated_relation: str,
) -> None:
    real = pd.DataFrame(
        {
            "quasi_id": np.arange(20, dtype=float),
            "secret": [0, 1] * 10,
        }
    )
    distributed = real.copy()
    concentrated = real.copy()
    concentrated["quasi_id"] = 0.0
    real_loader = GenericDataLoader(real, sensitive_features=["secret"])

    distributed_result = evaluator_t(
        use_cache=False,
        random_state=0,
        structural_n_clusters=[2, 5],
        structural_min_rows_per_cluster=2,
    ).evaluate(real_loader, GenericDataLoader(distributed, sensitive_features=["secret"]))
    concentrated_result = evaluator_t(
        use_cache=False,
        random_state=0,
        structural_n_clusters=[2, 5],
        structural_min_rows_per_cluster=2,
    ).evaluate(real_loader, GenericDataLoader(concentrated, sensitive_features=["secret"]))

    distributed_value = distributed_result[result_key]
    concentrated_value = concentrated_result[result_key]
    assert concentrated_value != pytest.approx(distributed_value)
    if concentrated_relation == "greater":
        assert concentrated_value > distributed_value
    else:
        assert concentrated_value < distributed_value


@pytest.mark.parametrize(
    "calibration_axis",
    [
        "sensitive_features",
        "scaling",
        "sample_size",
        "cluster_count",
        "minimum_cluster_size",
        "seed",
    ],
)
def test_structural_proxy_cache_identity_binds_calibration_inputs(
    tmp_path, calibration_axis: str
) -> None:
    real = pd.DataFrame(
        {
            "quasi_id": np.arange(40, dtype=float),
            "shape": np.tile([0.0, 1.0], 20),
            "secret": np.tile([0, 1], 20),
            "alternate_secret": np.repeat([0, 1], 20),
        }
    )
    scaled = real.copy()
    scaled["quasi_id"] *= 100.0

    baseline = {
        "real": real,
        "synthetic": real.copy(),
        "sensitive_features": ["secret"],
        "random_state": 0,
        "structural_n_clusters": [2, 5],
        "structural_min_rows_per_cluster": 2,
    }
    variant = {
        **baseline,
        "real": real,
        "synthetic": real.copy(),
    }
    if calibration_axis == "sensitive_features":
        variant["sensitive_features"] = ["secret", "alternate_secret"]
    elif calibration_axis == "scaling":
        variant["real"] = scaled
        variant["synthetic"] = scaled.copy()
    elif calibration_axis == "sample_size":
        variant["synthetic"] = real.iloc[:20].copy()
    elif calibration_axis == "cluster_count":
        variant["structural_n_clusters"] = [2]
    elif calibration_axis == "minimum_cluster_size":
        variant["structural_min_rows_per_cluster"] = 4
    elif calibration_axis == "seed":
        variant["random_state"] = 17
    else:
        raise AssertionError(f"Unhandled structural calibration axis: {calibration_axis}")

    for case in (baseline, variant):
        evaluator = kAnonymization(
            workspace=tmp_path,
            random_state=case["random_state"],
            structural_n_clusters=case["structural_n_clusters"],
            structural_min_rows_per_cluster=case["structural_min_rows_per_cluster"],
        )
        evaluator.evaluate(
            GenericDataLoader(case["real"], sensitive_features=case["sensitive_features"]),
            GenericDataLoader(
                case["synthetic"], sensitive_features=case["sensitive_features"]
            ),
        )

    cache_files = list(tmp_path.glob("sc_metric_cache_privacy_k-anonymization_*.bkp"))
    assert len(cache_files) == 2


def test_structural_proxy_cache_identity_includes_configuration(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "quasi_id": np.arange(20, dtype=float),
            "secret": [0, 1] * 10,
        }
    )
    loader = GenericDataLoader(frame, sensitive_features=["secret"])

    kAnonymization(
        workspace=tmp_path,
        structural_n_clusters=[2],
        structural_min_rows_per_cluster=2,
    ).evaluate(loader, loader)
    kAnonymization(
        workspace=tmp_path,
        structural_n_clusters=[5],
        structural_min_rows_per_cluster=2,
    ).evaluate(loader, loader)

    cache_files = list(tmp_path.glob("sc_metric_cache_privacy_k-anonymization_*.bkp"))
    assert len(cache_files) == 2


def test_privacy_cache_retains_result_metadata(tmp_path, monkeypatch) -> None:
    evidence = GenericDataLoader(pd.DataFrame({"value": np.arange(8, dtype=float)}))
    synthetic = GenericDataLoader(pd.DataFrame({"value": np.arange(8, dtype=float) + 10}))
    members = GenericDataLoader(pd.DataFrame({"value": [30.0, 31.0]}))
    validation = GenericDataLoader(pd.DataFrame({"value": np.arange(8, dtype=float) + 20}))
    scores = np.ones(4)

    def controlled_density_scores(
        self, synth_set, synth_val_set, reference_set, X_test, device
    ):
        del self, synth_set, synth_val_set, reference_set, device
        return scores, np.ones(len(X_test))

    monkeypatch.setattr(DomiasMIAPrior, "evaluate_p_R", controlled_density_scores)
    first = DomiasMIAPrior(workspace=tmp_path)
    first_result = first.evaluate(evidence, synthetic, members, validation, reference_size=2)
    first_metadata = first.result_metadata()

    second = DomiasMIAPrior(workspace=tmp_path)
    second_result = second.evaluate(evidence, synthetic, members, validation, reference_size=2)

    assert second_result == first_result
    assert second.result_metadata() == first_metadata
    assert "domias_protocol" in second.result_metadata()


def test_privacy_protocols_bind_repeated_seed_metadata(tmp_path, monkeypatch) -> None:
    seeds = (0, 7, 17)
    structural_frame = pd.DataFrame(
        {
            "quasi_id": np.arange(12, dtype=float),
            "secret": [0, 1] * 6,
        }
    )
    structural_loader = GenericDataLoader(
        structural_frame,
        sensitive_features=["secret"],
    )
    structural_metadata = []
    identifiability_metadata = []
    domias_raw = []
    domias_effective = []
    domias_metadata = []

    def controlled_density_scores(synth_set, synth_val_set, reference_set, X_test, device):
        del synth_set, synth_val_set, reference_set, device
        member_scores = np.full(6, 0.8 if evaluator_seed % 2 == 0 else 0.2)
        non_member_scores = np.full(6, 0.2 if evaluator_seed % 2 == 0 else 0.8)
        return np.concatenate([member_scores, non_member_scores]), np.ones(len(X_test))

    for evaluator_seed in seeds:
        structural_evaluator = kAnonymization(
            workspace=tmp_path / f"structural-{evaluator_seed}",
            use_cache=False,
            random_state=evaluator_seed,
            structural_n_clusters=[2],
            structural_min_rows_per_cluster=2,
        )
        structural_evaluator.evaluate(structural_loader, structural_loader)
        structural_metadata.append(structural_evaluator.result_metadata())

        identifiability_evaluator = IdentifiabilityScore(
            workspace=tmp_path / f"identifiability-{evaluator_seed}",
            use_cache=False,
            random_state=evaluator_seed,
        )
        identifiability_evaluator.evaluate(structural_loader, structural_loader)
        identifiability_metadata.append(identifiability_evaluator.result_metadata())

        evaluator = DomiasMIAPrior(
            workspace=tmp_path / f"domias-{evaluator_seed}",
            use_cache=False,
            random_state=evaluator_seed,
        )
        monkeypatch.setattr(evaluator, "evaluate_p_R", controlled_density_scores)
        evidence = GenericDataLoader(pd.DataFrame({"value": np.arange(12, dtype=float)}))
        synthetic = GenericDataLoader(
            pd.DataFrame({"value": np.arange(12, dtype=float) + 20})
        )
        members = GenericDataLoader(pd.DataFrame({"value": np.arange(6, dtype=float) + 40}))
        validation = GenericDataLoader(
            pd.DataFrame({"value": np.arange(12, dtype=float) + 60})
        )
        result = evaluator.evaluate(
            evidence,
            synthetic,
            members,
            validation,
            reference_size=6,
        )
        domias_raw.append(result["aucroc"])
        domias_effective.append(result["effective_auc_v2"])
        domias_metadata.append(evaluator.result_metadata())

    assert [metadata["random_state"] for metadata in structural_metadata] == list(seeds)
    assert [metadata["random_state"] for metadata in identifiability_metadata] == list(seeds)
    assert [metadata["random_state"] for metadata in domias_metadata] == list(seeds)
    assert [metadata["domias_protocol"]["random_state"] for metadata in domias_metadata] == list(
        seeds
    )
    assert domias_raw == [1.0, 0.0, 0.0]
    assert domias_effective == [1.0, 1.0, 1.0]
    assert all(metadata["calibration_only"] for metadata in structural_metadata)
    assert all(metadata["calibration_only"] for metadata in identifiability_metadata)
    assert all(metadata["calibration_only"] for metadata in domias_metadata)


def test_domias_effective_auc_inverts_below_chance() -> None:
    assert effective_auc_v2(0.4) == pytest.approx(0.6)
    assert effective_auc_v2(0.6) == pytest.approx(0.6)
    assert effective_auc_v2(0.5) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="DOMIAS AUC"):
        effective_auc_v2(float("nan"))


def test_domias_rejects_insufficient_reference_population() -> None:
    frame = pd.DataFrame({"value": list(range(6))})
    loader = GenericDataLoader(frame)
    evaluator = DomiasMIAPrior(use_cache=False)

    with pytest.raises(ValueError, match="two disjoint evidence populations"):
        evaluator.evaluate(
            loader,
            loader,
            loader,
            loader,
            reference_size=4,
        )


def test_domias_grouped_reference_populations_are_disjoint(monkeypatch) -> None:
    evidence = pd.DataFrame({"value": list(range(8))})
    evidence_loader = GenericDataLoader(
        evidence,
        group_ids=["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
    )
    members = GenericDataLoader(
        pd.DataFrame({"value": [8, 9]}),
        group_ids=["train-p1", "train-p2"],
    )
    evaluator = DomiasMIAPrior(use_cache=False)
    monkeypatch.setattr(
        evaluator,
        "evaluate_p_R",
        lambda synth_set, synth_val_set, reference_set, X_test, device: (
            np.ones(len(X_test)),
            np.ones(len(X_test)),
        ),
    )

    evaluator.evaluate(
        evidence_loader,
        evidence_loader,
        members,
        evidence_loader,
        reference_size=2,
    )

    protocol = evaluator.result_metadata()["domias_protocol"]
    metadata = evaluator.result_metadata()
    assert metadata["result_version"] == DOMIAS_RESULT_VERSION
    assert metadata["calibration_only"] is True
    assert metadata["calibration_required"] is True
    assert metadata["population_roles"]["evidence"]["group_ids_present"] is True
    assert metadata["population_roles"]["synthetic"]["group_ids_present"] is True
    assert protocol["group_disjoint"] is True
    assert protocol["population_unit"] == "patient_group"
    assert protocol["population_selection"] == "group_disjoint_prefix_suffix_v1"
    assert protocol["non_member_rows"] == 2
    assert protocol["reference_rows"] == 2


@pytest.mark.parametrize(
    ("reference_size", "member_rows", "score_values", "expected_accuracy", "expected_auc"),
    [
        (
            5,
            5,
            [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0],
            0.4,
            0.4,
        ),
        (4, 4, None, 0.5, 0.5),
        (4, 2, None, 2 / 3, 0.5),
        (4, 6, None, 0.4, 0.5),
    ],
    ids=["balanced_inverted", "balanced_chance", "member_sparse", "member_majority"],
)
def test_domias_reports_bounded_population_and_inversion_matrix(
    monkeypatch,
    reference_size: int,
    member_rows: int,
    score_values: list[float] | None,
    expected_accuracy: float,
    expected_auc: float,
) -> None:
    evidence = GenericDataLoader(
        pd.DataFrame({"value": np.arange(2 * reference_size, dtype=float)})
    )
    synthetic = GenericDataLoader(
        pd.DataFrame({"value": np.arange(2 * reference_size, dtype=float) + 10})
    )
    validation = GenericDataLoader(
        pd.DataFrame({"value": np.arange(2 * reference_size, dtype=float) + 20})
    )
    members = GenericDataLoader(
        pd.DataFrame({"value": np.arange(member_rows, dtype=float) + 30})
    )
    scores = (
        np.ones(member_rows + reference_size)
        if score_values is None
        else np.asarray(score_values, dtype=float)
    )
    evaluator = DomiasMIAPrior(use_cache=False)
    monkeypatch.setattr(
        evaluator,
        "evaluate_p_R",
        lambda synth_set, synth_val_set, reference_set, X_test, device: (
            scores,
            np.ones(len(X_test)),
        ),
    )

    result = evaluator.evaluate(
        evidence,
        synthetic,
        members,
        validation,
        reference_size=reference_size,
    )

    assert result["accuracy"] == pytest.approx(expected_accuracy)
    assert result["aucroc"] == pytest.approx(expected_auc)
    assert result["effective_auc_v2"] == pytest.approx(max(expected_auc, 1 - expected_auc))
    metadata = evaluator.result_metadata()
    assert metadata["result_version"] == DOMIAS_RESULT_VERSION
    assert metadata["calibration_only"] is True
    assert metadata["calibration_required"] is True
    protocol = metadata["domias_protocol"]
    assert protocol["reference_size_requested"] == reference_size
    assert protocol["auc_chance"] == pytest.approx(0.5)
    assert protocol["accuracy_prevalence_dependent"] is True
    assert protocol["member_rows"] == member_rows
    assert protocol["non_member_rows"] == reference_size
    assert protocol["reference_rows"] == reference_size
    assert protocol["group_disjoint"] is False
    assert protocol["population_unit"] == "row"
    assert protocol["population_selection"] == "row_disjoint_prefix_suffix_v1"


def test_domias_cache_identity_includes_member_and_validation_populations(tmp_path) -> None:
    evidence = GenericDataLoader(pd.DataFrame({"value": np.arange(8, dtype=float)}))
    synthetic = GenericDataLoader(pd.DataFrame({"value": np.arange(8, dtype=float) + 10}))
    members_a = GenericDataLoader(pd.DataFrame({"value": [30.0, 31.0]}))
    members_b = GenericDataLoader(pd.DataFrame({"value": [40.0, 41.0]}))
    validation = GenericDataLoader(pd.DataFrame({"value": np.arange(8, dtype=float) + 20}))
    evaluator = DomiasMIAPrior(workspace=tmp_path)

    evaluator.evaluate_p_R = lambda synth_set, synth_val_set, reference_set, X_test, device: (
        np.ones(len(X_test)),
        np.ones(len(X_test)),
    )
    evaluator.evaluate(evidence, synthetic, members_a, validation, reference_size=2)
    evaluator.evaluate(evidence, synthetic, members_b, validation, reference_size=2)

    cache_files = list(tmp_path.glob("sc_metric_cache_privacy_DomiasMIA_prior_*.bkp"))
    assert len(cache_files) == 2
