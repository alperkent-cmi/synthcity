# stdlib
import hashlib
import json
import platform
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, Tuple, Union

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from scipy import stats
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics import _utils
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.constants import DEVICE
from synthcity.utils.serialization import load_from_file, save_to_file

# synthcity relative
from .core import MetricEvaluator

PRIVACY_RESULT_VERSION = "privacy-v2"
PRIVACY_RESULT_CACHE_SCHEMA_VERSION = "privacy-result-v2"
STRUCTURAL_PRIVACY_RESULT_VERSION = "structural-proxy-v2"
IDENTIFIABILITY_RESULT_VERSION = "identifiability-v2"
STRUCTURAL_KMEANS_N_CLUSTERS = (2, 5, 10, 15)
STRUCTURAL_MIN_ROWS_PER_CLUSTER = 10
DOMIAS_AUC_CHANCE = 0.5
DOMIAS_RESULT_VERSION = "domias-effective-auc-v2"


def _feature_columns_hash(columns: list[str]) -> str:
    payload = json.dumps(columns, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def effective_auc_v2(aucroc: float) -> float:
    """Return inversion-aware DOMIAS attack strength with balanced chance 0.5."""
    value = float(aucroc)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"DOMIAS AUC must be finite and within [0, 1], got {aucroc!r}")
    return max(value, 1.0 - value)


class PrivacyEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.PrivacyEvaluator
        :parts: 1
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._result_metadata: dict[str, Any] = {}

    def result_metadata(self) -> dict[str, Any]:
        return dict(self._result_metadata)

    def _prepare_result_metadata(
        self, X_gt: DataLoader, X_syn: DataLoader
    ) -> None:
        self._result_metadata = {}

    def _cache_context(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return ""

    @staticmethod
    def type() -> str:
        return "privacy"

    @abstractmethod
    def _evaluate(
        self, X_gt: DataLoader, X_syn: DataLoader, *args: Any, **kwargs: Any
    ) -> Dict:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self, X_gt: DataLoader, X_syn: DataLoader, *args: Any, **kwargs: Any
    ) -> Dict:
        self._prepare_result_metadata(X_gt, X_syn)
        cache_context = self._cache_context(*args, **kwargs)
        cache_suffix = f"_{cache_context}" if cache_context else ""
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{PRIVACY_RESULT_VERSION}_{PRIVACY_RESULT_CACHE_SCHEMA_VERSION}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}{cache_suffix}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            cached = load_from_file(cache_file)
            if (
                isinstance(cached, dict)
                and cached.get("cache_schema_version") == PRIVACY_RESULT_CACHE_SCHEMA_VERSION
            ):
                if "result" not in cached or not isinstance(cached.get("metadata"), dict):
                    raise ValueError(f"Malformed privacy metric cache envelope at {cache_file}")
                self._result_metadata = dict(cached["metadata"])
                return cached["result"]
            return cached
        results = self._evaluate(X_gt, X_syn, *args, **kwargs)
        metadata = self.result_metadata()
        if metadata:
            save_to_file(
                cache_file,
                {
                    "cache_schema_version": PRIVACY_RESULT_CACHE_SCHEMA_VERSION,
                    "result": results,
                    "metadata": metadata,
                },
            )
        else:
            save_to_file(cache_file, results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)[self._default_metric]


class StructuralPrivacyEvaluator(PrivacyEvaluator):
    """Configurable KMeans proxy screen with explicit calibration metadata."""

    def __init__(
        self,
        structural_n_clusters: Any = STRUCTURAL_KMEANS_N_CLUSTERS,
        structural_min_rows_per_cluster: int = STRUCTURAL_MIN_ROWS_PER_CLUSTER,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            n_clusters = tuple(int(value) for value in structural_n_clusters)
        except (TypeError, ValueError) as exc:
            raise ValueError("structural_n_clusters must contain positive integers") from exc
        if not n_clusters or any(value < 2 for value in n_clusters):
            raise ValueError("structural_n_clusters must contain values >= 2")
        if len(set(n_clusters)) != len(n_clusters):
            raise ValueError("structural_n_clusters must not contain duplicates")
        if structural_min_rows_per_cluster < 1:
            raise ValueError("structural_min_rows_per_cluster must be positive")
        self._structural_n_clusters = n_clusters
        self._structural_min_rows_per_cluster = int(structural_min_rows_per_cluster)

    def _cache_context(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        payload = {
            "result_version": STRUCTURAL_PRIVACY_RESULT_VERSION,
            "n_clusters": self._structural_n_clusters,
            "min_rows_per_cluster": self._structural_min_rows_per_cluster,
            "random_state": self._random_state,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    def _prepare_result_metadata(
        self, X_gt: DataLoader, X_syn: DataLoader
    ) -> None:
        gt_features = [str(feature) for feature in _utils.get_features(X_gt, X_gt.sensitive_features)]
        syn_features = [
            str(feature) for feature in _utils.get_features(X_syn, X_syn.sensitive_features)
        ]
        self._result_metadata = {
            "result_version": STRUCTURAL_PRIVACY_RESULT_VERSION,
            "proxy_label": "kmeans_partition_screen_not_formal_guarantee",
            "calibration_only": True,
            "calibration_required": True,
            "random_state": self._random_state,
            "feature_selection": {
                "real_columns": gt_features,
                "synthetic_columns": syn_features,
                "real_columns_hash": _feature_columns_hash(gt_features),
                "synthetic_columns_hash": _feature_columns_hash(syn_features),
                "sensitive_features_real": [str(value) for value in X_gt.sensitive_features],
                "sensitive_features_synthetic": [
                    str(value) for value in X_syn.sensitive_features
                ],
            },
            "preprocessing": "caller-provided DataLoader values; no metric-side scaling",
            "kmeans": {
                "n_clusters": list(self._structural_n_clusters),
                "min_rows_per_cluster": self._structural_min_rows_per_cluster,
                "init": "k-means++",
                "random_state": self._random_state,
            },
            "sample_sizes": {
                "real": len(X_gt),
                "synthetic": len(X_syn),
            },
        }

    def _iter_kmeans(self, X: DataLoader, features: list[str]):
        for n_clusters in self._structural_n_clusters:
            if len(X) / n_clusters < self._structural_min_rows_per_cluster:
                continue
            yield n_clusters, KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                random_state=self._random_state,
            ).fit(X[features])


class kAnonymization(StructuralPrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.kAnonymization
        :parts: 1

    Returns the k-anon ratio between the real data and the synthetic data.
    For each dataset, it is computed the value k which satisfies the k-anonymity rule: each record is similar to at least another k-1 other records on the potentially identifying variables.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="syn", **kwargs)

    @staticmethod
    def name() -> str:
        return "k-anonymization"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_data(self, X: DataLoader) -> int:
        features = _utils.get_features(X, X.sensitive_features)

        values = [999]
        for _, cluster in self._iter_kmeans(X, features):
            counts: dict = Counter(cluster.labels_)
            values.append(np.min(list(counts.values())))

        return int(np.min(values))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError(f"Metric {self.name()} doesn't support images")

        return {
            "gt": self.evaluate_data(X_gt),
            "syn": (self.evaluate_data(X_syn) + 1e-8),
        }


class lDiversityDistinct(StructuralPrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.lDiversityDistinct
        :parts: 1

    Returns the distinct l-diversity ratio between the real data and the synthetic data.

    For each dataset, it computes the minimum value l which satisfies the distinct l-diversity rule: every generalized block has to contain at least l different sensitive values.

    We simulate a set of the cluster over the dataset, and we return the minimum length of unique sensitive values for any cluster.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="syn", **kwargs)

    @staticmethod
    def name() -> str:
        return "distinct l-diversity"

    @staticmethod
    def direction() -> str:
        return "maximize"

    def evaluate_data(self, X: DataLoader) -> int:
        features = _utils.get_features(X, X.sensitive_features)

        values = [999]
        for n_clusters, model in self._iter_kmeans(X, features):
            clusters = model.predict(X.dataframe()[features])
            clusters_df = pd.Series(clusters, index=X.dataframe().index)
            for cluster in range(n_clusters):
                partition = X.dataframe()[clusters_df == cluster]
                uniq_values = partition[X.sensitive_features].drop_duplicates()
                values.append(len(uniq_values))

        return int(np.min(values))

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError(f"Metric {self.name()} doesn't support images")

        return {
            "gt": self.evaluate_data(X_gt),
            "syn": (self.evaluate_data(X_syn) + 1e-8),
        }


class kMap(StructuralPrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.kMap
        :parts: 1

    Returns the minimum value k that satisfies the k-map rule.

    The data satisfies k-map if every combination of values for the quasi-identifiers appears at least k times in the reidentification(synthetic) dataset.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "k-map"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError(f"Metric {self.name()} doesn't support images")

        features = _utils.get_features(X_gt, X_gt.sensitive_features)

        values = []
        for _, model in self._iter_kmeans(X_gt, features):
            clusters = model.predict(X_syn[features])
            counts: dict = Counter(clusters)
            values.append(np.min(list(counts.values())))

        if len(values) == 0:
            return {"score": 0}

        return {"score": int(np.min(values))}


class DeltaPresence(StructuralPrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.DeltaPresence
        :parts: 1

    Returns the maximum re-identification probability on the real dataset from the synthetic dataset.

    For each dataset partition, we report the maximum ratio of unique sensitive information between the real dataset and in the synthetic dataset.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score", **kwargs)

    @staticmethod
    def name() -> str:
        return "delta-presence"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            raise ValueError(f"Metric {self.name()} doesn't support images")

        features = _utils.get_features(X_gt, X_gt.sensitive_features)

        values = []
        for _, model in self._iter_kmeans(X_gt, features):
            clusters = model.predict(X_syn[features])
            synth_counts: dict = Counter(clusters)
            gt_counts: dict = Counter(model.labels_)

            for key in gt_counts:
                if key not in synth_counts:
                    continue
                gt_cnt = gt_counts[key]
                synth_cnt = synth_counts[key]

                delta = gt_cnt / (synth_cnt + 1e-8)

                values.append(delta)

        return {"score": float(np.max(values))}


class IdentifiabilityScore(PrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.IdentifiabilityScore
        :parts: 1

    Returns the re-identification score on the real dataset from the synthetic dataset.

    We estimate the risk of re-identifying any real data point using synthetic data.
    Intuitively, if the synthetic data are very close to the real data, the re-identification risk would be high.
    The precise formulation of the re-identification score is given in the reference below.

    Reference: Jinsung Yoon, Lydia N. Drumright, Mihaela van der Schaar,
    "Anonymization through Data Synthesis using Generative Adversarial Networks (ADS-GAN):
    A harmonizing advancement for AI in medicine,"
    IEEE Journal of Biomedical and Health Informatics (JBHI), 2019.
    Paper link: https://ieeexplore.ieee.org/document/9034117
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="score_OC", **kwargs)

    @staticmethod
    def name() -> str:
        return "identifiability_score"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def _cache_context(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return IDENTIFIABILITY_RESULT_VERSION

    def _prepare_result_metadata(
        self, X_gt: DataLoader, X_syn: DataLoader
    ) -> None:
        self._result_metadata = {
            "result_version": IDENTIFIABILITY_RESULT_VERSION,
            "calibration_only": True,
            "direction": self.direction(),
            "random_state": self._random_state,
            "population_roles": {
                "reference": {
                    "role": "reference",
                    "hash": X_gt.hash(),
                    "rows": len(X_gt),
                    "group_ids_present": X_gt.group_ids is not None,
                },
                "synthetic": {
                    "role": "synthetic",
                    "hash": X_syn.hash(),
                    "rows": len(X_syn),
                    "group_ids_present": X_syn.group_ids is not None,
                },
            },
            "preprocessing": {
                "representation": "flattened_dataloader_numpy_v1",
                "feature_weighting_legacy": "unit_weights_v1",
                "feature_weighting_entropy": "rounded_label_entropy_v1",
                "one_class_embedding": "oneclass_layer_v1",
            },
            "variants": {
                "score": {
                    "embedding": "raw",
                    "weighting": "legacy",
                    "lifecycle": "calibration_only",
                },
                "score_OC": {
                    "embedding": "one_class",
                    "weighting": "legacy",
                    "lifecycle": "calibration_only",
                },
                "score_entropy_weighted": {
                    "embedding": "raw",
                    "weighting": "entropy",
                    "lifecycle": "calibration_only",
                },
                "score_OC_entropy_weighted": {
                    "embedding": "one_class",
                    "weighting": "entropy",
                    "lifecycle": "calibration_only",
                },
            },
        }

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        results = self._compute_scores(X_gt, X_syn)

        oc_results = self._compute_scores(X_gt, X_syn, "OC")

        for key in oc_results:
            results[key] = oc_results[key]
        results.update(
            self._compute_scores(
                X_gt,
                X_syn,
                weighting="entropy",
                output_suffix="_entropy_weighted",
            )
        )
        results.update(
            self._compute_scores(
                X_gt,
                X_syn,
                "OC",
                weighting="entropy",
                output_suffix="_entropy_weighted",
            )
        )
        log.info("ID_score results: ", results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _compute_scores(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
        emb: str = "",
        weighting: str = "legacy",
        output_suffix: str = "",
    ) -> Dict:
        """Compare Wasserstein distance between original data and synthetic data.

        Args:
            orig_data: original data
            synth_data: synthetically generated data

        Returns:
            WD_value: Wasserstein distance
        """
        X_gt_ = X_gt.numpy().reshape(len(X_gt), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        if weighting not in {"legacy", "entropy"}:
            raise ValueError(f"Unknown identifiability weighting: {weighting!r}")

        embedding_suffix = f"_{emb}" if emb else ""
        if emb == "OC":
            oneclass_model = self._get_oneclass_model(X_gt_)
            X_gt_ = self._oneclass_predict(oneclass_model, X_gt_)
            X_syn_ = self._oneclass_predict(oneclass_model, X_syn_)
        else:
            if emb != "":
                raise RuntimeError(f" Invalid emb {emb}")

        # Entropy computation
        def compute_entropy(labels: np.ndarray) -> np.ndarray:
            value, counts = np.unique(np.round(labels), return_counts=True)
            return entropy(counts)

        # Parameters
        no, x_dim = X_gt_.shape

        # Weights
        W = np.zeros(
            [
                x_dim,
            ]
        )

        for i in range(x_dim):
            W[i] = compute_entropy(X_gt_[:, i])

        if weighting == "legacy":
            W = np.ones_like(W)

        for i in range(x_dim):
            W[i] = max(float(W[i]), 1e-16)

        X_hat = X_gt_ / W
        X_syn_hat = X_syn_ / W

        # r_i computation
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
        distance, _ = nbrs.kneighbors(X_hat)

        # hat{r_i} computation
        nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
        distance_hat, _ = nbrs_hat.kneighbors(X_hat)

        # See which one is bigger
        R_Diff = distance_hat[:, 0] - distance[:, 1]
        identifiability_value = np.sum(R_Diff < 0) / float(no)

        return {f"score{embedding_suffix}{output_suffix}": identifiability_value}


class DomiasMIA(PrivacyEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.domias
        :parts: 1

    DOMIAS is a membership inference attacker model against synthetic data, that incorporates
    density estimation to detect generative model overfitting. That is it uses local overfitting to
    detect whether a data point was used to train the generative model or not.

    AUC has balanced member/non-member chance level 0.5. ``accuracy`` depends
    on the population class balance and must not be used as a prevalence-free
    privacy score. ``effective_auc_v2`` treats an attacker that can invert its
    score as equally capable, so an observed AUC of 0.4 becomes 0.6 while the
    raw ``aucroc`` remains available for audit.

    Returns:
    A dictionary with a key for each of the `synthetic_sizes` values.
    For each `synthetic_sizes` value, the dictionary contains the keys:
        * `MIA_performance` : accuracy and AUCROC for each attack
        * `MIA_scores`: output scores for each attack

    Reference: Boris van Breugel, Hao Sun, Zhaozhi Qian,  Mihaela van der Schaar, AISTATS 2023.
    DOMIAS: Membership Inference Attacks against Synthetic Data through Overfitting Detection.

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="aucroc", **kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def _prepare_result_metadata(
        self, X_gt: DataLoader, X_syn: DataLoader
    ) -> None:
        self._result_metadata = {
            "result_version": DOMIAS_RESULT_VERSION,
            "calibration_only": True,
            "calibration_required": True,
            "direction": self.direction(),
            "random_state": self._random_state,
            "population_roles": {
                "evidence": {
                    "role": "X_gt",
                    "hash": X_gt.hash(),
                    "rows": len(X_gt),
                    "group_ids_present": X_gt.group_ids is not None,
                },
                "synthetic": {
                    "role": "X_syn",
                    "hash": X_syn.hash(),
                    "rows": len(X_syn),
                    "group_ids_present": X_syn.group_ids is not None,
                },
            },
        }

    def _cache_context(self, *args: Any, **kwargs: Any) -> str:
        reference_size = kwargs.get("reference_size")
        if reference_size is None and len(args) >= 3:
            reference_size = args[2]
        population_hashes = []
        for population in args[:2]:
            hash_method = getattr(population, "hash", None)
            population_hashes.append(
                hash_method() if callable(hash_method) else repr(population)
            )
        payload = {
            "reference_size": reference_size,
            "member_population": population_hashes[0] if population_hashes else None,
            "synthetic_validation_population": (
                population_hashes[1] if len(population_hashes) > 1 else None
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
        X_train: DataLoader,
        X_ref_syn: DataLoader,
        reference_size: int,
    ) -> float:
        return self.evaluate(
            X_gt,
            X_syn,
            X_train,
            X_ref_syn,
            reference_size=reference_size,
        )[self._default_metric]

    @abstractmethod
    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Any:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: Union[
            DataLoader, Any
        ],  # TODO: X_gt needs to be big enough that it can be split into non_mem_set and also ref_set
        synth_set: Union[DataLoader, Any],
        X_train: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_size: int = 100,  # look at default sizes
        device: Any = DEVICE,
    ) -> Dict:
        """
        Evaluate various Membership Inference Attacks, using the `generator` and the `dataset`.
        The provided generator must not be fitted.

        Args:
            generator: GeneratorInterface
                Generator with the `fit` and `generate` methods. The generator MUST not be fitted.
            X_gt: Union[DataLoader, Any]
                The evaluation dataset, used to derive the training and test datasets.
            synth_set: Union[DataLoader, Any]
                The synthetic dataset.
            X_train: Union[DataLoader, Any]
                The dataset used to create the mem_set.
            synth_val_set: Union[DataLoader, Any]
                The dataset used to calculate the density of the synthetic data
            reference_size: int
                The size of the reference dataset
            device: PyTorch device
                CPU or CUDA

        Returns:
            A dictionary with the AUCROC and accuracy scores for the attack.
        """

        if isinstance(reference_size, bool) or not isinstance(reference_size, int):
            raise ValueError(
                f"DOMIAS reference_size must be a positive integer, got {reference_size!r}"
            )
        if reference_size < 1:
            raise ValueError(
                f"DOMIAS reference_size must be positive, got {reference_size}"
            )

        gt_array = X_gt.numpy()
        group_ids = getattr(X_gt, "group_ids", None)
        train_group_ids = getattr(X_train, "group_ids", None)
        if group_ids is None:
            if len(X_gt) < 2 * reference_size:
                raise ValueError(
                    "DOMIAS reference_size requires at least two disjoint evidence "
                    f"populations; rows={len(X_gt)}, requested={reference_size}"
                )
            non_member_indices = np.arange(reference_size)
            reference_indices = np.arange(len(X_gt) - reference_size, len(X_gt))
            group_safe = False
        else:
            if train_group_ids is None:
                raise ValueError(
                    "DOMIAS grouped evidence requires group_ids for the member population"
                )
            group_series = pd.Series(list(group_ids))
            train_groups = set(train_group_ids)
            unique_groups = group_series.drop_duplicates().tolist()
            non_member_groups = []
            non_member_rows = 0
            for group in unique_groups:
                non_member_groups.append(group)
                non_member_rows += int((group_series == group).sum())
                if non_member_rows >= reference_size:
                    break
            reference_groups = []
            reference_rows = 0
            for group in reversed(unique_groups):
                if group in non_member_groups:
                    continue
                reference_groups.append(group)
                reference_rows += int((group_series == group).sum())
                if reference_rows >= reference_size:
                    break
            if non_member_rows < reference_size or reference_rows < reference_size:
                raise ValueError(
                    "DOMIAS group-disjoint reference populations are too small: "
                    f"requested={reference_size}, non_member_rows={non_member_rows}, "
                    f"reference_rows={reference_rows}"
                )
            if train_groups.intersection(non_member_groups + reference_groups):
                raise ValueError(
                    "DOMIAS member and evidence populations must be group-disjoint"
                )
            non_member_indices = np.flatnonzero(group_series.isin(non_member_groups).to_numpy())
            reference_indices = np.flatnonzero(group_series.isin(reference_groups).to_numpy())
            group_safe = True

        mem_set = X_train.dataframe()
        non_mem_set = gt_array[non_member_indices]
        reference_set = gt_array[reference_indices]
        self._result_metadata.update(
            {
                "domias_protocol": {
                    "result_version": DOMIAS_RESULT_VERSION,
                    "reference_size_requested": reference_size,
                    "auc_chance": DOMIAS_AUC_CHANCE,
                    "accuracy_prevalence_dependent": True,
                    "random_state": self._random_state,
                    "member_rows": int(len(mem_set)),
                    "non_member_rows": int(len(non_mem_set)),
                    "reference_rows": int(len(reference_set)),
                    "group_disjoint": group_safe,
                    "population_unit": "patient_group" if group_safe else "row",
                    "population_selection": (
                        "group_disjoint_prefix_suffix_v1"
                        if group_safe
                        else "row_disjoint_prefix_suffix_v1"
                    ),
                    "population_roles": {
                        "members": "X_train",
                        "non_members": "X_gt_prefix",
                        "reference": "X_gt_suffix",
                        "synthetic": "synth_set",
                        "synthetic_validation": "synth_val_set",
                    },
                }
            }
        )

        all_real_data = np.concatenate((X_train.numpy(), gt_array), axis=0)

        continuous = []
        for i in np.arange(all_real_data.shape[1]):
            if len(np.unique(all_real_data[:, i])) < 10:
                continuous.append(0)
            else:
                continuous.append(1)

        self.norm = _utils.normal_func_feat(all_real_data, continuous)

        """ 3. Synthesis with the GeneratorInferface"""

        # get real test sets of members and non members
        X_test = np.concatenate([mem_set, non_mem_set])
        Y_test = np.concatenate(
            [np.ones(mem_set.shape[0]), np.zeros(non_mem_set.shape[0])]
        ).astype(bool)

        """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
        # First, estimate density of synthetic data then
        # eqn2: \prop P_G(x_i)/P_X(x_i)
        # p_R estimation
        p_G_evaluated, p_R_evaluated = self.evaluate_p_R(
            synth_set, synth_val_set, reference_set, X_test, device
        )

        p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

        acc, auc = _utils.compute_metrics_baseline(p_rel, Y_test)
        accuracy = float(acc)
        aucroc = float(auc)
        if not 0.0 <= accuracy <= 1.0 or not np.isfinite(accuracy):
            raise ValueError(f"DOMIAS accuracy must be finite and within [0, 1], got {acc!r}")
        return {
            "accuracy": accuracy,
            "aucroc": aucroc,
            "effective_auc_v2": effective_auc_v2(aucroc),
        }


class DomiasMIAPrior(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_prior"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
        p_G_evaluated = density_gen(X_test.transpose(1, 0))
        p_R_evaluated = self.norm.pdf(X_test)
        return p_G_evaluated, p_R_evaluated


class DomiasMIAKDE(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_KDE"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if synth_set.shape[0] > X_test.shape[0]:
            log.debug(
                """
The data appears to lie in a lower-dimensional subspace of the space in which it is expressed.
This has resulted in a singular data covariance matrix, which cannot be treated using the algorithms
implemented in `gaussian_kde`. If you wish to use the density estimator `kde` or `prior`, consider performing principle component analysis / dimensionality reduction
and using `gaussian_kde` with the transformed data. Else consider using `bnaf` as the density estimator.
                """
            )

        density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
        density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
        p_G_evaluated = density_gen(X_test.transpose(1, 0))
        p_R_evaluated = density_data(X_test.transpose(1, 0))
        return p_G_evaluated, p_R_evaluated


class DomiasMIABNAF(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_BNAF"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, p_G_model = _utils.density_estimator_trainer(
            synth_set.values,
            synth_val_set.values[: int(0.5 * synth_val_set.shape[0])],
            synth_val_set.values[int(0.5 * synth_val_set.shape[0]) :],
        )
        _, p_R_model = _utils.density_estimator_trainer(reference_set)
        p_G_evaluated = np.exp(
            _utils.compute_log_p_x(
                p_G_model, torch.as_tensor(X_test).float().to(device)
            )
            .cpu()
            .detach()
            .numpy()
        )
        p_R_evaluated = np.exp(
            _utils.compute_log_p_x(
                p_R_model, torch.as_tensor(X_test).float().to(device)
            )
            .cpu()
            .detach()
            .numpy()
        )
        return p_G_evaluated, p_R_evaluated
