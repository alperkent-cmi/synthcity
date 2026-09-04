# stdlib
import hashlib
import json
import platform
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

# third party
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from pydantic import validate_arguments
from scipy import linalg
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import chisquare, ks_2samp
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics._utils import get_frequency
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.models.survival_analysis.metrics import (
    nonparametric_distance,
)
from synthcity.utils.reproducibility import clear_cache
from synthcity.utils.serialization import load_from_file, save_to_file

METRIC_CACHE_SCHEMA_VERSION = "metric-result-v2"


class StatisticalEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.StatisticalEvaluator
        :parts: 1

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "stats"

    @abstractmethod
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict: ...

    def _cache_context(self) -> Dict[str, Any]:
        return {}

    def _cache_context_digest(self) -> str:
        encoded = json.dumps(
            self._cache_context(),
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}_{METRIC_CACHE_SCHEMA_VERSION}_{self._cache_context_digest()}.bkp"
        )
        if hasattr(self, "_result_metadata"):
            self._result_metadata = {}
        if self.use_cache(cache_file):
            cached = load_from_file(cache_file)
            if (
                isinstance(cached, dict)
                and cached.get("cache_schema_version") == METRIC_CACHE_SCHEMA_VERSION
            ):
                if "result" not in cached or not isinstance(cached.get("metadata"), dict):
                    raise ValueError(f"Malformed metric cache envelope at {cache_file}")
                self._result_metadata = dict(cached["metadata"])
                return cached["result"]
            return cached

        clear_cache()
        results = self._evaluate(X_gt, X_syn)
        metadata_getter = getattr(self, "result_metadata", None)
        metadata = metadata_getter() if callable(metadata_getter) else {}
        if metadata:
            save_to_file(
                cache_file,
                {
                    "cache_schema_version": METRIC_CACHE_SCHEMA_VERSION,
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


class InverseKLDivergence(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.InverseKLDivergence
        :parts: 1


    Returns the average inverse of the Kullback–Leibler Divergence metric.

    Score:
        0: the datasets are from different distributions.
        1: the datasets are from the same distribution.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="marginal", **kwargs)

    @staticmethod
    def name() -> str:
        return "inv_kl_divergence"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        freqs = get_frequency(
            X_gt.dataframe(), X_syn.dataframe(), n_histogram_bins=self._n_histogram_bins
        )
        res = []
        for col in X_gt.columns:
            gt_freq, synth_freq = freqs[col]
            res.append(1 / (1 + np.sum(kl_div(gt_freq, synth_freq))))

        return {"marginal": float(self.reduction()(res))}


class KolmogorovSmirnovTest(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.KolmogorovSmirnovTest
        :parts: 1

    Performs the Kolmogorov-Smirnov test for goodness of fit.

    Score:
        0: the distributions are totally different.
        1: the distributions are identical.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="marginal", **kwargs)

    @staticmethod
    def name() -> str:
        return "ks_test"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        res = []
        for col in X_gt.columns:
            statistic, _ = ks_2samp(X_gt[col], X_syn[col])
            res.append(1 - statistic)

        return {"marginal": float(self.reduction()(res))}


class ChiSquaredTest(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.ChiSquaredTest
        :parts: 1

    Performs the one-way chi-square test.

    Returns:
        The p-value. A small value indicates that we can reject the null hypothesis and that the distributions are different.

    Score:
        0: the distributions are different
        1: the distributions are identical.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="marginal", **kwargs)

    @staticmethod
    def name() -> str:
        return "chi_squared_test"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        res = []
        freqs = get_frequency(
            X_gt.dataframe(), X_syn.dataframe(), n_histogram_bins=self._n_histogram_bins
        )

        for col in X_gt.columns:
            gt_freq, synth_freq = freqs[col]
            try:
                _, pvalue = chisquare(gt_freq, synth_freq)
                if np.isnan(pvalue):
                    pvalue = 0
            except BaseException:
                log.error("chisquare failed")
                pvalue = 0

            res.append(pvalue)

        return {"marginal": float(self.reduction()(res))}


class MaximumMeanDiscrepancy(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.MaximumMeanDiscrepancy
        :parts: 1

    Empirical maximum mean discrepancy. The lower the result the more evidence that distributions are the same.

    Args:
        kernel: "rbf", "linear" or "polynomial"

    Score:
        0: The distributions are the same.
        1: The distributions are totally different.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(self, kernel: str = "rbf", **kwargs: Any) -> None:
        super().__init__(default_metric="joint", **kwargs)

        self.kernel = kernel

    @staticmethod
    def name() -> str:
        return "max_mean_discrepancy"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if self.kernel == "linear":
            """
            MMD using linear kernel (i.e., k(x,y) = <x,y>)
            """
            delta_df = X_gt.dataframe().mean(axis=0) - X_syn.dataframe().mean(axis=0)
            delta = delta_df.values

            score = delta.dot(delta.T)
        elif self.kernel == "rbf":
            """
            MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
            """
            gamma = 1.0
            XX = metrics.pairwise.rbf_kernel(
                X_gt.numpy().reshape(len(X_gt), -1),
                X_gt.numpy().reshape(len(X_gt), -1),
                gamma,
            )
            YY = metrics.pairwise.rbf_kernel(
                X_syn.numpy().reshape(len(X_syn), -1),
                X_syn.numpy().reshape(len(X_syn), -1),
                gamma,
            )
            XY = metrics.pairwise.rbf_kernel(
                X_gt.numpy().reshape(len(X_gt), -1),
                X_syn.numpy().reshape(len(X_syn), -1),
                gamma,
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        elif self.kernel == "polynomial":
            """
            MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
            """
            degree = 2
            gamma = 1
            coef0 = 0
            XX = metrics.pairwise.polynomial_kernel(
                X_gt.numpy().reshape(len(X_gt), -1),
                X_gt.numpy().reshape(len(X_gt), -1),
                degree,
                gamma,
                coef0,
            )
            YY = metrics.pairwise.polynomial_kernel(
                X_syn.numpy().reshape(len(X_syn), -1),
                X_syn.numpy().reshape(len(X_syn), -1),
                degree,
                gamma,
                coef0,
            )
            XY = metrics.pairwise.polynomial_kernel(
                X_gt.numpy().reshape(len(X_gt), -1),
                X_syn.numpy().reshape(len(X_syn), -1),
                degree,
                gamma,
                coef0,
            )
            score = XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            raise ValueError(f"Unsupported kernel {self.kernel}")

        return {"joint": float(score)}


class JensenShannonDistance(StatisticalEvaluator):
    """Evaluate schema-aware per-variable Jensen-Shannon distances.

    Categorical variables use the deterministic union of real and synthetic
    values as their support. Continuous variables use shared edges computed
    from both populations. The legacy ``marginal`` aggregate remains the
    default result, while versioned variable rows and their source-table
    metadata are retained for downstream aggregation.
    """

    output_version = "jsd_v2"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        normalize: bool = True,
        feature_types: Optional[Dict[str, str]] = None,
        source_table: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(default_metric="marginal", **kwargs)

        self.normalize = normalize
        self.feature_types = dict(feature_types or {})
        self.source_table = dict(source_table or {})
        self._result_metadata: Dict[str, Any] = {}

    @staticmethod
    def name() -> str:
        return "jensenshannon_dist"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def _cache_context(self) -> Dict[str, Any]:
        return {
            "version": self.output_version,
            "normalize": self.normalize,
            "n_histogram_bins": self._n_histogram_bins,
            "feature_types": self.feature_types,
            "source_table": self.source_table,
        }

    @staticmethod
    def _is_missing(value: Any) -> bool:
        try:
            missing = pd.isna(value)
            return (
                bool(missing) if not isinstance(missing, (np.ndarray, list)) else False
            )
        except (TypeError, ValueError):
            return False

    @classmethod
    def _categorical_token(cls, value: Any) -> Tuple[str, str, str]:
        if cls._is_missing(value):
            return ("missing", "", "")
        value_type = type(value)
        return (
            "value",
            f"{value_type.__module__}.{value_type.__qualname__}",
            repr(value),
        )

    @classmethod
    def _categorical_counts(
        cls,
        real: pd.Series,
        synthetic: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
        tokens = {
            cls._categorical_token(value)
            for value in pd.concat([real, synthetic], ignore_index=True).tolist()
        }
        support = sorted(tokens)
        real_tokens = [cls._categorical_token(value) for value in real.tolist()]
        synthetic_tokens = [
            cls._categorical_token(value) for value in synthetic.tolist()
        ]
        real_counts = np.asarray(
            [real_tokens.count(token) for token in support], dtype=float
        )
        synthetic_counts = np.asarray(
            [synthetic_tokens.count(token) for token in support], dtype=float
        )
        support_metadata = [
            {"kind": kind, "type": value_type, "repr": representation}
            for kind, value_type, representation in support
        ]
        return real_counts, synthetic_counts, support_metadata

    @staticmethod
    def _numeric_values(series: pd.Series, column: str) -> np.ndarray:
        converted = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        invalid = series.notna().to_numpy() & np.isnan(converted)
        if invalid.any():
            raise ValueError(
                f"Continuous column {column!r} contains non-numeric values"
            )
        non_missing = converted[series.notna().to_numpy()]
        if not np.isfinite(non_missing).all():
            raise ValueError(f"Continuous column {column!r} contains non-finite values")
        return converted

    def _continuous_counts(
        self,
        real: pd.Series,
        synthetic: pd.Series,
        column: str,
    ) -> Tuple[np.ndarray, np.ndarray, list[float]]:
        real_values = self._numeric_values(real, column)
        synthetic_values = self._numeric_values(synthetic, column)
        finite_values = np.concatenate(
            [
                real_values[np.isfinite(real_values)],
                synthetic_values[np.isfinite(synthetic_values)],
            ]
        )
        if len(finite_values) == 0:
            edges = np.asarray([], dtype=float)
            real_counts = np.asarray([float(real_values.size)], dtype=float)
            synthetic_counts = np.asarray([float(synthetic_values.size)], dtype=float)
            return real_counts, synthetic_counts, []

        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
        unique_count = len(np.unique(finite_values))
        bin_count = max(1, min(self._n_histogram_bins, unique_count))
        if lower == upper:
            edges = np.asarray([lower - 0.5, upper + 0.5], dtype=float)
        else:
            edges = np.linspace(lower, upper, bin_count + 1)

        real_finite = real_values[np.isfinite(real_values)]
        synthetic_finite = synthetic_values[np.isfinite(synthetic_values)]
        real_counts = np.histogram(real_finite, bins=edges)[0].astype(float)
        synthetic_counts = np.histogram(synthetic_finite, bins=edges)[0].astype(float)
        real_counts = np.concatenate(
            [real_counts, np.asarray([float(np.isnan(real_values).sum())])]
        )
        synthetic_counts = np.concatenate(
            [synthetic_counts, np.asarray([float(np.isnan(synthetic_values).sum())])]
        )
        return real_counts, synthetic_counts, edges.tolist()

    def _feature_type(self, X_gt: DataLoader, column: str) -> str:
        loader_feature_types = getattr(X_gt, "feature_types", {})
        feature_type = self.feature_types.get(column, loader_feature_types.get(column))
        if feature_type is None:
            series = X_gt[column]
            feature_type = (
                "categorical"
                if (
                    pd.api.types.is_object_dtype(series)
                    or isinstance(series.dtype, pd.CategoricalDtype)
                    or pd.api.types.is_bool_dtype(series)
                )
                else "continuous"
            )
        if feature_type not in {"categorical", "continuous"}:
            raise ValueError(
                f"Unsupported feature type {feature_type!r} for column {column!r}"
            )
        return feature_type

    def result_metadata(self) -> Dict[str, Any]:
        return dict(self._result_metadata)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_stats(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Tuple[Dict, Dict, Dict]:
        stats_gt: Dict[str, np.ndarray] = {}
        stats_syn: Dict[str, np.ndarray] = {}
        stats_: Dict[str, float] = {}
        variable_metadata: Dict[str, Dict[str, Any]] = {}

        for col in X_gt.columns:
            feature_type = self._feature_type(X_gt, col)
            if feature_type == "categorical":
                real_counts, synthetic_counts, support = self._categorical_counts(
                    X_gt[col], X_syn[col]
                )
                edges: list[float] = []
            else:
                real_counts, synthetic_counts, edges = self._continuous_counts(
                    X_gt[col], X_syn[col], col
                )

            if self.normalize:
                real_distribution = (real_counts + 1) / (
                    real_counts.sum() + len(real_counts)
                )
                synthetic_distribution = (synthetic_counts + 1) / (
                    synthetic_counts.sum() + len(synthetic_counts)
                )
            else:
                real_distribution = real_counts + 1
                synthetic_distribution = synthetic_counts + 1
            stats_gt[col] = real_distribution
            stats_syn[col] = synthetic_distribution

            stats_[col] = float(
                jensenshannon(real_distribution, synthetic_distribution)
            )
            if np.isnan(stats_[col]):
                raise RuntimeError("NaNs in prediction")
            source_table = self.source_table.get(
                col, getattr(X_gt, "source_table", {}).get(col, "unassigned")
            )
            source_table = "unassigned" if source_table is None else str(source_table)
            variable_metadata[col] = {
                "feature_type": feature_type,
                "source_table": source_table,
                "support": support if feature_type == "categorical" else None,
                "bin_edges": edges if feature_type == "continuous" else None,
                "distance": stats_[col],
            }

        source_table_values: Dict[str, list[float]] = {}
        source_table_variables: Dict[str, list[str]] = {}
        for column, distance in stats_.items():
            source_table = variable_metadata[column]["source_table"]
            source_table_values.setdefault(source_table, []).append(float(distance))
            source_table_variables.setdefault(source_table, []).append(column)
        source_table_summary = {
            source_table: {
                "variables": variables,
                "n_variables": len(variables),
                "mean_distance": float(np.mean(source_table_values[source_table])),
            }
            for source_table, variables in source_table_variables.items()
        }
        source_table_macro = float(
            np.mean([summary["mean_distance"] for summary in source_table_summary.values()])
        )
        max_variable = max(stats_, key=stats_.get)
        self._result_metadata = {
            "metric": self.name(),
            "version": self.output_version,
            "aggregation": self._reduction,
            "variables": variable_metadata,
            "aggregation_contract": {
                "schema_version": "source-table-aggregation-v1",
                "source_table_macro_v2": {
                    "definition": "mean of per-source-table mean variable distances",
                    "source_tables": source_table_summary,
                    "value": source_table_macro,
                },
                "max_variable_v2": {
                    "definition": "maximum distance across declared variables",
                    "selected_variables": [
                        column
                        for column, distance in stats_.items()
                        if distance == stats_[max_variable]
                    ],
                    "value": float(stats_[max_variable]),
                },
            },
        }

        return stats_, stats_gt, stats_syn

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        stats_, _, _ = self._evaluate_stats(X_gt, X_syn)
        if not stats_:
            raise ValueError("Jensen-Shannon distance requires at least one column")

        results = {"marginal": float(self.reduction()(list(stats_.values())))}
        results.update(
            {f"variable_v2.{column}": distance for column, distance in stats_.items()}
        )
        aggregation_contract = self._result_metadata["aggregation_contract"]
        results["source_table_macro_v2"] = aggregation_contract["source_table_macro_v2"]["value"]
        results["max_variable_v2"] = aggregation_contract["max_variable_v2"]["value"]
        return results


class FrozenSupportJensenShannonDistance(JensenShannonDistance):
    """Leakage-safe, versioned elastic-net-style marginal JSD evidence.

    ``evaluate_frozen_support`` fits candidate supports on ``train`` and final
    supports on ``train`` plus ``tuning``. Neither synthetic nor evidence data
    can add categorical values or continuous bin edges. Invalid variables are
    retained in provenance and make the corresponding aggregate incomplete.
    """

    output_version = "jsd_elastic_net_v1"

    def __init__(
        self,
        patient_id_column: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.patient_id_column = patient_id_column

    @staticmethod
    def name() -> str:
        return "jensenshannon_dist_frozen"

    def _cache_context(self) -> Dict[str, Any]:
        context = super()._cache_context()
        context.update({"patient_id_column": self.patient_id_column})
        return context

    @staticmethod
    def _loader_frame(loader: DataLoader) -> pd.DataFrame:
        return loader.dataframe().copy()

    def _columns(self, frame: pd.DataFrame) -> list[str]:
        columns = list(frame.columns)
        if self.patient_id_column is not None:
            columns = [column for column in columns if column != self.patient_id_column]
        return columns

    def _support(
        self, frame: pd.DataFrame, column: str, feature_type: str
    ) -> Tuple[list[Any], list[float], Optional[str]]:
        series = frame[column]
        if feature_type == "categorical":
            tokens = sorted({self._categorical_token(value) for value in series.tolist()})
            return tokens, [], None
        values = self._numeric_values(series, column)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return [], [], None
        lower, upper = float(np.min(finite)), float(np.max(finite))
        unique_count = len(np.unique(finite))
        count = max(1, min(self._n_histogram_bins, unique_count))
        if lower == upper:
            edges = [lower - 0.5, upper + 0.5]
        else:
            edges = np.linspace(lower, upper, count + 1).tolist()
        return [], edges, None

    def _continuous_edges(self, frame: pd.DataFrame, column: str) -> list[float]:
        """Build frozen edges from finite fit values without hiding bad values."""
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return []
        lower, upper = float(np.min(finite)), float(np.max(finite))
        count = max(1, min(self._n_histogram_bins, len(np.unique(finite))))
        if lower == upper:
            return [lower - 0.5, upper + 0.5]
        return np.linspace(lower, upper, count + 1).tolist()

    def _frozen_variable(
        self,
        fit_frame: pd.DataFrame,
        observed_frame: pd.DataFrame,
        column: str,
        feature_type: str,
        source_table: str,
    ) -> Tuple[float, Dict[str, Any]]:
        if feature_type == "continuous":
            edges = self._continuous_edges(fit_frame, column)
            support = []
        else:
            support, edges, _ = self._support(fit_frame, column, feature_type)
        metadata: Dict[str, Any] = {
            "feature_type": feature_type,
            "source_table": source_table,
            "fit_role": "real_fit",
            "support": None,
            "bin_edges": None,
            "outcomes": [],
            "distance": None,
        }
        if fit_frame[column].isna().any() or observed_frame[column].isna().any():
            metadata["outcomes"].append("missing")
        if fit_frame[column].duplicated().any() or observed_frame[column].duplicated().any():
            metadata["outcomes"].append("duplicate")
        if feature_type == "continuous":
            for series in (fit_frame[column], observed_frame[column]):
                numeric = pd.to_numeric(series, errors="coerce")
                non_finite = series.notna() & (
                    numeric.isin([np.inf, -np.inf]) | numeric.isna()
                )
                if bool(non_finite.any()):
                    metadata["outcomes"].append("non_finite")
        if feature_type == "categorical":
            metadata["support"] = [
                {"kind": k, "type": t, "repr": r} for k, t, r in support
            ]
            fit_tokens = [self._categorical_token(v) for v in fit_frame[column].tolist()]
            observed_tokens = [self._categorical_token(v) for v in observed_frame[column].tolist()]
            unknown = sorted(set(observed_tokens) - set(support))
            if unknown:
                metadata["outcomes"].append("unknown")
            if not support:
                metadata["outcomes"].append("absent")
            observed_counts = np.asarray([observed_tokens.count(token) for token in support], dtype=float)
            fit_counts = np.asarray([fit_tokens.count(token) for token in support], dtype=float)
        else:
            metadata["bin_edges"] = edges
            fit_values = self._numeric_values(fit_frame[column], column)
            observed_values = self._numeric_values(observed_frame[column], column)
            fit_finite = fit_values[np.isfinite(fit_values)]
            observed_finite = observed_values[np.isfinite(observed_values)]
            if len(edges) == 0:
                metadata["outcomes"].append("absent")
                fit_counts = np.asarray([len(fit_values)], dtype=float)
                observed_counts = np.asarray([len(observed_values)], dtype=float)
            else:
                under = (observed_finite < edges[0]).any()
                over = (observed_finite > edges[-1]).any()
                if under:
                    metadata["outcomes"].append("underflow")
                if over:
                    metadata["outcomes"].append("overflow")
                fit_counts = np.histogram(fit_finite, bins=edges)[0].astype(float)
                observed_counts = np.histogram(observed_finite, bins=edges)[0].astype(float)
                fit_counts = np.concatenate([fit_counts, [float(np.isnan(fit_values).sum())]])
                observed_counts = np.concatenate([observed_counts, [float(np.isnan(observed_values).sum())]])
        if not len(fit_counts) or not len(observed_counts):
            metadata["outcomes"].append("absent")
        distance = float(
            jensenshannon(
                (fit_counts + 1) / (fit_counts.sum() + len(fit_counts)),
                (observed_counts + 1) / (observed_counts.sum() + len(observed_counts)),
            )
        )
        metadata["distance"] = distance
        return distance, metadata

    def _feature_type_from_loader(self, loader: DataLoader, column: str) -> str:
        feature_type = self.feature_types.get(column)
        if feature_type is None:
            feature_type = getattr(loader, "feature_types", {}).get(column)
        if feature_type is None:
            series = loader[column]
            feature_type = (
                "categorical"
                if pd.api.types.is_object_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(series)
                else "continuous"
            )
        if feature_type not in {"categorical", "continuous"}:
            raise ValueError(f"Unsupported feature type {feature_type!r} for {column!r}")
        return feature_type

    def _score_frame(
        self,
        fit_frame: pd.DataFrame,
        observed_frame: pd.DataFrame,
        schema_loader: DataLoader,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        columns = self._columns(fit_frame)
        variables: Dict[str, Any] = {}
        distances: list[float] = []
        for column in columns:
            if column not in observed_frame:
                feature_type = self._feature_type_from_loader(schema_loader, column)
                source_table = self.source_table.get(
                    column,
                    getattr(schema_loader, "source_table", {}).get(
                        column, "unassigned"
                    ),
                )
                source_table = "unassigned" if source_table is None else str(source_table)
                support: Optional[list[dict[str, str]]] = None
                bin_edges: Optional[list[float]] = None
                if feature_type == "categorical":
                    support_tokens, _, _ = self._support(
                        fit_frame, column, feature_type
                    )
                    support = [
                        {"kind": kind, "type": value_type, "repr": representation}
                        for kind, value_type, representation in support_tokens
                    ]
                else:
                    bin_edges = self._continuous_edges(fit_frame, column)
                variables[column] = {
                    "feature_type": feature_type,
                    "source_table": source_table,
                    "fit_role": "real_fit",
                    "support": support,
                    "bin_edges": bin_edges,
                    "outcomes": ["absent"],
                    "distance": None,
                }
                continue
            feature_type = None
            source_table = "unassigned"
            frozen_edges: Optional[list[float]] = None
            try:
                feature_type = self._feature_type_from_loader(schema_loader, column)
                source_table = self.source_table.get(
                    column, getattr(schema_loader, "source_table", {}).get(column, "unassigned")
                )
                source_table = "unassigned" if source_table is None else str(source_table)
                if feature_type == "continuous":
                    frozen_edges = self._continuous_edges(fit_frame, column)
                distance, metadata = self._frozen_variable(
                    fit_frame,
                    observed_frame,
                    column,
                    feature_type,
                    source_table,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                distance, metadata = float("nan"), {
                    "feature_type": feature_type,
                    "source_table": source_table,
                    "fit_role": "real_fit",
                    "support": None,
                    "bin_edges": frozen_edges,
                    "outcomes": ["non_finite" if "finite" in str(error) else "invalid"],
                    "error": str(error),
                    "distance": None,
                }
            variables[column] = metadata
            distances.append(distance)
        invalid_outcomes = {
            "missing", "unknown", "underflow", "overflow", "absent", "non_finite", "invalid"
        }
        invalid = [
            column
            for column, data in variables.items()
            if invalid_outcomes.intersection(data.get("outcomes", []))
        ]
        if invalid or not distances:
            aggregate = None
        else:
            values = np.asarray(distances, dtype=float)
            weights = np.full(len(values), 1.0 / len(values))
            aggregate = float(np.clip(0.5 * np.sum(weights * values) + 0.5 * np.sqrt(np.sum(weights * values**2)), 0.0, 1.0))
        source_tables: Dict[str, list[float]] = {}
        for variable in variables.values():
            if variable.get("distance") is not None:
                source_tables.setdefault(variable.get("source_table", "unassigned"), []).append(variable["distance"])
        source_table_summary = {
            table: {"n_variables": len(values), "mean_distance": float(np.mean(values))}
            for table, values in source_tables.items()
        }
        return aggregate, {
            "variables": variables,
            "invalid_variables": invalid,
            "complete": not invalid and bool(distances),
            "aggregation_contract": {
                "schema_version": "source-table-aggregation-v1",
                "source_table_macro_v2": {
                    "definition": "mean of per-source-table mean variable distances",
                    "source_tables": source_table_summary,
                    "value": float(np.mean([item["mean_distance"] for item in source_table_summary.values()])) if source_table_summary else None,
                },
                "max_variable_v2": {
                    "definition": "maximum distance across declared variables",
                    "value": float(max((item["distance"] for item in variables.values() if item.get("distance") is not None), default=0.0)),
                },
            },
        }

    def evaluate_frozen_support(
        self,
        train: DataLoader,
        tuning: DataLoader,
        synthetic: DataLoader,
        evidence: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Evaluate candidate and final evidence using frozen real-data supports."""
        train_frame = self._loader_frame(train)
        tuning_frame = self._loader_frame(tuning)
        observed = self._loader_frame(synthetic)
        final_frame = pd.concat([train_frame, tuning_frame], ignore_index=True)
        candidate, candidate_meta = self._score_frame(train_frame, observed, train)
        final, final_meta = self._score_frame(final_frame, observed, train)
        result: Dict[str, Any] = {
            "candidate": candidate,
            "final": final,
            "version": self.output_version,
            "metadata": {"candidate": candidate_meta, "final": final_meta, "patient_id_column": self.patient_id_column},
        }
        if evidence is not None:
            evidence_frame = self._loader_frame(evidence)
            evidence_score, evidence_meta = self._score_frame(final_frame, evidence_frame, train)
            result["evidence"] = evidence_score
            result["metadata"]["evidence"] = evidence_meta
        self._result_metadata = result["metadata"]
        return result


# Descriptive aliases for callers using either terminology from the contract.
ElasticNetJensenShannonDistance = FrozenSupportJensenShannonDistance
FrozenSupportJSD = FrozenSupportJensenShannonDistance


class WassersteinDistance(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.WassersteinDistance
        :parts: 1

    Compare Wasserstein distance between original data and synthetic data.

    Args:
        X: original data
        X_syn: synthetically generated data

    Returns:
        WD_value: Wasserstein distance
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="joint", **kwargs)

    @staticmethod
    def name() -> str:
        return "wasserstein_dist"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        if len(X_) > len(X_syn_):
            X_syn_ = np.concatenate(
                [X_syn_, np.zeros((len(X_) - len(X_syn_), X_.shape[1]))]
            )

        scaler = MinMaxScaler().fit(X_)

        X_ = scaler.transform(X_)
        X_syn_ = scaler.transform(X_syn_)

        X_ten = torch.from_numpy(X_)
        Xsyn_ten = torch.from_numpy(X_syn_)
        OT_solver = SamplesLoss(loss="sinkhorn")

        return {"joint": OT_solver(X_ten, Xsyn_ten).cpu().numpy().item()}


class PRDCScore(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.PRDCScore
        :parts: 1


    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        nearest_k: int.
    """

    def __init__(self, nearest_k: int = 5, **kwargs: Any) -> None:
        super().__init__(default_metric="precision", **kwargs)

        self.nearest_k = nearest_k

    @staticmethod
    def name() -> str:
        return "prdc"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        # Default representation
        results = self._compute_prdc(X_, X_syn_)

        return results

    def _compute_pairwise_distance(
        self, data_x: np.ndarray, data_y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x

        dists = metrics.pairwise_distances(data_x, data_y)
        return dists

    def _get_kth_value(
        self, unsorted: np.ndarray, k: int, axis: int = -1
    ) -> np.ndarray:
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def _compute_nearest_neighbour_distances(
        self, input_features: np.ndarray, nearest_k: int
    ) -> np.ndarray:
        """
        Args:
            input_features: numpy.ndarray
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self._compute_pairwise_distance(input_features)
        radii = self._get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def _compute_prdc(
        self, real_features: np.ndarray, fake_features: np.ndarray
    ) -> Dict:
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            dict of precision, recall, density, and coverage.
        """

        real_nearest_neighbour_distances = self._compute_nearest_neighbour_distances(
            real_features, self.nearest_k
        )
        fake_nearest_neighbour_distances = self._compute_nearest_neighbour_distances(
            fake_features, self.nearest_k
        )
        distance_real_fake = self._compute_pairwise_distance(
            real_features, fake_features
        )

        precision = (
            (
                distance_real_fake
                < np.expand_dims(real_nearest_neighbour_distances, axis=1)
            )
            .any(axis=0)
            .mean()
        )

        recall = (
            (
                distance_real_fake
                < np.expand_dims(fake_nearest_neighbour_distances, axis=0)
            )
            .any(axis=1)
            .mean()
        )

        density = (1.0 / float(self.nearest_k)) * (
            distance_real_fake
            < np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).sum(axis=0).mean()

        coverage = (
            distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
        ).mean()

        return dict(
            precision=precision, recall=recall, density=density, coverage=coverage
        )


class AlphaPrecision(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.AlphaPrecision
        :parts: 1

    Evaluates the alpha-precision, beta-recall, and authenticity scores.

    The class evaluates the synthetic data using a tuple of three metrics:
    alpha-precision, beta-recall, and authenticity.
    Note that these metrics can be evaluated for each synthetic data point (which are useful for auditing and
    post-processing). Here we average the scores to reflect the overall quality of the data.
    The formal definitions can be found in the reference below:

    Alaa, Ahmed, Boris Van Breugel, Evgeny S. Saveliev, and Mihaela van der Schaar. "How faithful is your synthetic
    data? sample-level metrics for evaluating and auditing generative models."
    In International Conference on Machine Learning, pp. 290-306. PMLR, 2022.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="authenticity_OC", **kwargs)

    @staticmethod
    def name() -> str:
        return "alpha_precision"

    @staticmethod
    def direction() -> str:
        return "maximize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def metrics(
        self,
        X: np.ndarray,
        X_syn: np.ndarray,
        emb_center: Optional[np.ndarray] = None,
    ) -> Tuple:
        if len(X) != len(X_syn):
            raise RuntimeError("The real and synthetic data must have the same length")

        if emb_center is None:
            emb_center = np.mean(X, axis=0)

        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)

        Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)

        synth_center = np.mean(X_syn, axis=0)

        alpha_precision_curve = []
        beta_coverage_curve = []

        synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))

        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
        real_to_real, _ = nbrs_real.kneighbors(X)

        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)

        # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()

        real_synth_closest = X_syn[real_to_synth_args]

        real_synth_closest_d = np.sqrt(
            np.sum((real_synth_closest - synth_center) ** 2, axis=1)
        )
        closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)

        for k in range(len(Radii)):
            precision_audit_mask = synth_to_center <= Radii[k]
            alpha_precision = np.mean(precision_audit_mask)

            beta_coverage = np.mean(
                (
                    (real_to_synth <= real_to_real)
                    * (real_synth_closest_d <= closest_synth_Radii[k])
                )
            )

            alpha_precision_curve.append(alpha_precision)
            beta_coverage_curve.append(beta_coverage)

        # See which one is bigger

        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen)

        Delta_precision_alpha = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(alpha_precision_curve))
        ) / np.sum(alphas)

        if Delta_precision_alpha < 0:
            raise RuntimeError("negative value detected for Delta_precision_alpha")

        Delta_coverage_beta = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(beta_coverage_curve))
        ) / np.sum(alphas)

        if Delta_coverage_beta < 0:
            raise RuntimeError("negative value detected for Delta_coverage_beta")

        return (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        )

    def _normalize_covariates(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """_normalize_covariates
        This is an internal method to replicate the old, naive method for evaluating
        AlphaPrecision.

        Args:
            X (DataLoader): The ground truth dataset.
            X_syn (DataLoader): The synthetic dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: normalised version of the datasets
        """
        X_gt_norm = X.dataframe().copy()
        X_syn_norm = X_syn.dataframe().copy()
        if self._task_type != "survival_analysis":
            if hasattr(X, "target_column"):
                X_gt_norm = X_gt_norm.drop(columns=[X.target_column])
            if hasattr(X_syn, "target_column"):
                X_syn_norm = X_syn_norm.drop(columns=[X_syn.target_column])
        scaler = MinMaxScaler().fit(X_gt_norm)
        if hasattr(X, "target_column"):
            X_gt_norm_df = pd.DataFrame(
                scaler.transform(X_gt_norm),
                columns=[
                    col
                    for col in X.train().dataframe().columns
                    if col != X.target_column
                ],
            )
        else:
            X_gt_norm_df = pd.DataFrame(
                scaler.transform(X_gt_norm), columns=X.train().dataframe().columns
            )

        if hasattr(X_syn, "target_column"):
            X_syn_norm_df = pd.DataFrame(
                scaler.transform(X_syn_norm),
                columns=[
                    col
                    for col in X_syn.dataframe().columns
                    if col != X_syn.target_column
                ],
            )
        else:
            X_syn_norm_df = pd.DataFrame(
                scaler.transform(X_syn_norm), columns=X_syn.dataframe().columns
            )

        return (X_gt_norm_df, X_syn_norm_df)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        results = {}

        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        # OneClass representation
        emb = "_OC"
        oneclass_model = self._get_oneclass_model(X_)
        X_ = self._oneclass_predict(oneclass_model, X_)
        X_syn_ = self._oneclass_predict(oneclass_model, X_syn_)
        emb_center = oneclass_model.c.detach().cpu().numpy()

        (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        ) = self.metrics(X_, X_syn_, emb_center=emb_center)

        results[f"delta_precision_alpha{emb}"] = Delta_precision_alpha
        results[f"delta_coverage_beta{emb}"] = Delta_coverage_beta
        results[f"authenticity{emb}"] = authenticity

        X_df, X_syn_df = self._normalize_covariates(X, X_syn)
        (
            alphas_naive,
            alpha_precision_curve_naive,
            beta_coverage_curve_naive,
            Delta_precision_alpha_naive,
            Delta_coverage_beta_naive,
            authenticity_naive,
        ) = self.metrics(X_df.to_numpy(), X_syn_df.to_numpy(), emb_center=None)

        results["delta_precision_alpha_naive"] = Delta_precision_alpha_naive
        results["delta_coverage_beta_naive"] = Delta_coverage_beta_naive
        results["authenticity_naive"] = authenticity_naive

        return results


class SurvivalKMDistance(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.SurvivalKMDistance
        :parts: 1

    The distance between two Kaplan-Meier plots. Used for survival analysis"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(default_metric="optimism", **kwargs)

    @staticmethod
    def name() -> str:
        return "survival_km_distance"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if self._task_type != "survival_analysis":
            raise RuntimeError(
                f"The metric is valid only for survival analysis tasks, but got {self._task_type}"
            )
        if X.type() != "survival_analysis" or X_syn.type() != "survival_analysis":
            raise RuntimeError(
                f"The metric is valid only for survival analysis tasks, but got datasets {X.type()} and {X_syn.type()}"
            )

        _, real_T, real_E = X.unpack()
        _, syn_T, syn_E = X_syn.unpack()

        optimism, abs_optimism, sightedness = nonparametric_distance(
            (real_T, real_E), (syn_T, syn_E)
        )

        return {
            "optimism": optimism,
            "abs_optimism": abs_optimism,
            "sightedness": sightedness,
        }


class FrechetInceptionDistance(StatisticalEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_statistical.FrechetInceptionDistance
        :parts: 1

    Calculates the Frechet Inception Distance (FID) to evalulate GANs.

    Paper: GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.

    The FID metric calculates the distance between two distributions of images.
    Typically, we have summary statistics (mean & covariance matrix) of one of these distributions, while the 2nd distribution is given by a GAN.

    Adapted by Boris van Breugel(bv292@cam.ac.uk)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "fid"

    @staticmethod
    def direction() -> str:
        return "minimize"

    def _fit_gaussian(self, act: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculation of the statistics used by the FID.
        Params:
        -- act   : activations
        Returns:
        -- mu    : The mean over samples of the activations
        -- sigma : The covariance matrix of the activations
        """
        mu = np.mean(act, axis=0)
        sigma = np.cov(act.T)
        return mu, sigma

    def _calculate_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                 inception net ( like returned by the function 'get_predictions')
                 for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                   on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                   generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                   precalcualted on an representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        if mu1.shape != mu2.shape:
            raise RuntimeError("Training and test mean vectors have different lengths")

        if sigma1.shape != sigma2.shape:
            raise RuntimeError(
                "Training and test covariances have different dimensions"
            )

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=2e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate(
        self,
        X: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        if X.type() != "images":
            raise RuntimeError(
                f"The metric is valid only for image tasks, but got datasets {X.type()} and {X_syn.type()}"
            )

        X1 = X.numpy().reshape(len(X), -1)
        X2 = X_syn.numpy().reshape(len(X_syn), -1)

        mu1, cov1 = self._fit_gaussian(X1)
        mu2, cov2 = self._fit_gaussian(X2)
        score = self._calculate_frechet_distance(mu1, cov1, mu2, cov2)

        return {
            "score": score,
        }
