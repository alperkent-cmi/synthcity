# stdlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_call

# synthcity absolute
from synthcity.plugins.core.dataloader import (
    DataLoader,
    GenericDataLoader,
    create_from_info,
)

# synthcity relative
from .eval_attacks import DataLeakageLinear, DataLeakageMLP, DataLeakageXGB
from .eval_detection import (
    SyntheticDetectionLinear,
    SyntheticDetectionMLP,
    SyntheticDetectionXGB,
)
from .eval_performance import (
    AugmentationPerformanceEvaluatorLinear,
    AugmentationPerformanceEvaluatorMLP,
    AugmentationPerformanceEvaluatorXGB,
    FeatureImportanceRankDistance,
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
)
from .eval_privacy import (
    DeltaPresence,
    DomiasMIABNAF,
    DomiasMIAKDE,
    DomiasMIAPrior,
    IdentifiabilityScore,
    STRUCTURAL_KMEANS_N_CLUSTERS,
    STRUCTURAL_MIN_ROWS_PER_CLUSTER,
    kAnonymization,
    kMap,
    lDiversityDistinct,
)
from .eval_sanity import (
    CloseValuesProbability,
    CommonRowsProportion,
    DataMismatchScore,
    DistantValuesProbability,
    NearestSyntheticNeighborDistance,
)
from .eval_statistical import (
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
from .scores import ScoreEvaluator

standard_metrics = [
    # sanity tests
    DataMismatchScore,
    CommonRowsProportion,
    NearestSyntheticNeighborDistance,
    CloseValuesProbability,
    DistantValuesProbability,
    # statistical tests
    JensenShannonDistance,
    ChiSquaredTest,
    InverseKLDivergence,
    KolmogorovSmirnovTest,
    MaximumMeanDiscrepancy,
    WassersteinDistance,
    PRDCScore,
    AlphaPrecision,
    SurvivalKMDistance,
    FrechetInceptionDistance,
    # performance tests
    PerformanceEvaluatorLinear,
    PerformanceEvaluatorMLP,
    PerformanceEvaluatorXGB,
    AugmentationPerformanceEvaluatorLinear,
    AugmentationPerformanceEvaluatorMLP,
    AugmentationPerformanceEvaluatorXGB,
    FeatureImportanceRankDistance,
    # synthetic detection tests
    SyntheticDetectionXGB,
    SyntheticDetectionMLP,
    SyntheticDetectionLinear,
    # privacy tests
    DeltaPresence,
    kAnonymization,
    kMap,
    lDiversityDistinct,
    IdentifiabilityScore,
    DomiasMIABNAF,  # TODO: This takes too long to include as default
    DomiasMIAKDE,
    DomiasMIAPrior,
    # attribute-inference attack tests
    DataLeakageLinear,
    DataLeakageMLP,
    DataLeakageXGB,
]

GROUP_UNSAFE_MODALITIES = frozenset(
    {"syn_seq", "images", "time_series", "time_series_survival"}
)


def _group_unsafe_report(
    metrics: Optional[Dict],
    task_type: str,
    semantic_context: Optional[Dict[str, Any]] = None,
    *,
    group_mode: str = "patient_group",
    loader_types: Optional[Dict[str, str]] = None,
    reason: Optional[str] = None,
) -> pd.DataFrame:
    """Materialize selected grouped non-tabular metrics as blocked evidence."""
    selected_metrics = []
    requested_metrics = Metrics.list() if metrics is None else metrics
    for metric in standard_metrics:
        metric_names = requested_metrics.get(metric.type(), [])
        if metric.name() in metric_names:
            selected_metrics.append(metric)

    reason = reason or (
        "Grouped SynthCity evaluation is not supported for non-tabular task "
        f"{task_type!r}; metric internals require a group-aware implementation"
    )
    columns = [
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
    report = pd.DataFrame(
        [
            {
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "stddev": np.nan,
                "median": np.nan,
                "iqr": np.nan,
                "rounds": 0,
                "errors": 1,
                "error_types": "GroupUnsafe",
                "error_messages": reason,
                "durations": 0.0,
                "direction": metric.direction(),
            }
            for metric in selected_metrics
        ],
        index=[metric.fqdn() for metric in selected_metrics],
        columns=columns,
    )
    report.attrs["group_safety"] = {
        "schema_version": "group-safety-v1",
        "status": "group_unsafe",
        "group_mode": group_mode,
        "task_type": task_type,
        "reason": reason,
        "loader_types": dict(loader_types or {}),
        "metrics": [metric.fqdn() for metric in selected_metrics],
    }
    if semantic_context is not None:
        report.attrs["semantic_context"] = dict(semantic_context)
    return report


class Metrics:
    @staticmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        X_gt: Union[DataLoader, pd.DataFrame],
        X_syn: Union[DataLoader, pd.DataFrame],
        X_train: Optional[Union[DataLoader, pd.DataFrame]] = None,
        X_ref_syn: Optional[Union[DataLoader, pd.DataFrame]] = None,
        X_augmented: Optional[Union[DataLoader, pd.DataFrame]] = None,
        reduction: str = "mean",
        n_histogram_bins: int = 10,
        metrics: Optional[Dict] = None,
        task_type: str = "classification",
        group_mode: str = "row",
        random_state: int = 0,
        workspace: Path = Path("workspace"),
        use_cache: bool = True,
        n_folds: int = 5,
        quasi_identifier_columns: Optional[List[str]] = None,
        sensitive_target_types: Optional[Dict[str, str]] = None,
        classification_score: str = "balanced_accuracy",
        feature_types: Optional[Dict[str, str]] = None,
        source_table: Optional[Dict[str, str]] = None,
        structural_n_clusters: Optional[List[int]] = None,
        structural_min_rows_per_cluster: int = STRUCTURAL_MIN_ROWS_PER_CLUSTER,
        domias_reference_size: Optional[int] = None,
        X_gt_group_ids: Optional[Any] = None,
        X_syn_group_ids: Optional[Any] = None,
        X_train_group_ids: Optional[Any] = None,
        X_ref_syn_group_ids: Optional[Any] = None,
        X_augmented_group_ids: Optional[Any] = None,
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Core evaluation logic for the metrics

        X_gt: Dataloader or DataFrame
            Reference real data
        X_syn: Dataloader or DataFrame
            Synthetic data
        X_train: Dataloader or DataFrame
            The data used to train the synthetic model (used for domias metrics only).
        X_ref_syn: Dataloader or DataFrame
            Reference synthetic data (used for domias metrics only).
        X_augmented: Dataloader or DataFrame
            Augmented data
        metrics: dict
            the dictionary of metrics to evaluate
            Full dictionary of metrics is:
            {
                'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
                'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
                'detection': ['detection_xgb', 'detection_mlp', 'detection_linear'],
                'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score'],
                'attack': ['data_leakage_linear', 'data_leakage_mlp', 'data_leakage_xgb']
            }
        reduction: str
            The way to aggregate metrics across folds. Can be: 'mean', "min", or "max".
        n_histogram_bins: int
            The number of bins used in histogram calculation of a given metric. Defaults to 10.
        task_type: str
            The type of problem. Relevant for evaluating the downstream models with the correct metrics. Valid tasks are:  "classification", "regression", "survival_analysis", "time_series", "time_series_survival".
        random_state: int
            random seed
        quasi_identifier_columns: list[str]
            Explicit columns available to attribute-inference attacks. If not
            provided, attack evaluators fail closed unless the real loader has
            explicit ``important_features``.
        sensitive_target_types: dict[str, str]
            Optional schema-derived ``categorical``/``continuous`` target kinds
            for sensitive attributes, resolved before SynthCity encoding.
        structural_n_clusters: list[int]
            KMeans cluster counts for structural privacy proxy screens.
        structural_min_rows_per_cluster: int
            Minimum average rows per cluster before a structural screen runs a
            configured KMeans partition.
        workspace: Path
            The folder for caching intermediary results.
        use_cache: bool
            If the a metric has been previously run and is cached, it will be reused for the experiments. Defaults to True.
        """
        supported_tasks = [
            "classification",
            "regression",
            "survival_analysis",
            "time_series",
            "time_series_survival",
        ]
        if task_type not in supported_tasks:
            raise ValueError(
                f"Invalid task type {task_type}. Supported: {supported_tasks}"
            )
        if group_mode not in {"row", "patient_group"}:
            raise ValueError(
                f"Invalid group mode {group_mode!r}. Supported: ['row', 'patient_group']"
            )

        supplied_loaders = {
            label: loader
            for label, loader in (
                ("X_gt", X_gt),
                ("X_syn", X_syn),
                ("X_train", X_train),
                ("X_ref_syn", X_ref_syn),
                ("X_augmented", X_augmented),
            )
            if loader is not None
        }
        loader_types = {
            label: loader.type() if isinstance(loader, DataLoader) else "generic"
            for label, loader in supplied_loaders.items()
        }
        unsupported_modalities = sorted(
            {
                loader_type
                for loader_type in loader_types.values()
                if loader_type in GROUP_UNSAFE_MODALITIES
            }
        )
        if group_mode == "patient_group" and unsupported_modalities:
            return _group_unsafe_report(
                metrics,
                task_type,
                semantic_context,
                group_mode=group_mode,
                loader_types=loader_types,
                reason=(
                    "Patient-group SynthCity evaluation is not supported for loader type(s) "
                    f"{unsupported_modalities}; metric internals require a group-aware implementation"
                ),
            )

        workspace.mkdir(parents=True, exist_ok=True)

        def _check_group_ids(loader: DataLoader, group_ids: Any, label: str) -> None:
            if group_ids is None:
                return
            if loader.group_ids is None:
                raise ValueError(f"{label} loader does not contain group_ids")
            expected_values = list(group_ids)
            expected = np.empty(len(expected_values), dtype=object)
            expected[:] = expected_values
            actual_values = list(loader.group_ids)
            actual = np.empty(len(actual_values), dtype=object)
            actual[:] = actual_values
            if len(expected) != len(actual) or not np.array_equal(expected, actual):
                raise ValueError(f"{label} loader group_ids do not match the supplied values")

        def _info_without_groups(loader: DataLoader, group_ids: Any = None) -> dict:
            info = dict(loader.info())
            info.pop("group_ids", None)
            if group_ids is not None:
                info["group_ids"] = list(group_ids)
            return info

        if not isinstance(X_gt, DataLoader):
            X_gt = GenericDataLoader(X_gt, group_ids=X_gt_group_ids)
        else:
            _check_group_ids(X_gt, X_gt_group_ids, "X_gt")
        if not isinstance(X_syn, DataLoader):
            X_syn = create_from_info(X_syn, _info_without_groups(X_gt, X_syn_group_ids))
        else:
            _check_group_ids(X_syn, X_syn_group_ids, "X_syn")
        if X_train is not None and not isinstance(X_train, DataLoader):
            X_train = GenericDataLoader(X_train, group_ids=X_train_group_ids)
        elif X_train is not None:
            _check_group_ids(X_train, X_train_group_ids, "X_train")
        if X_ref_syn is not None and not isinstance(X_ref_syn, DataLoader):
            X_ref_syn = create_from_info(
                X_ref_syn, _info_without_groups(X_gt, X_ref_syn_group_ids)
            )
        elif X_ref_syn is not None:
            _check_group_ids(X_ref_syn, X_ref_syn_group_ids, "X_ref_syn")
        if X_augmented is not None and not isinstance(X_augmented, DataLoader):
            X_augmented = create_from_info(
                X_augmented, _info_without_groups(X_gt, X_augmented_group_ids)
            )
        elif X_augmented is not None:
            _check_group_ids(X_augmented, X_augmented_group_ids, "X_augmented")

        grouped_loaders = [
            loader
            for loader in (X_gt, X_syn, X_train, X_ref_syn, X_augmented)
            if loader is not None and loader.group_ids is not None
        ]
        if group_mode == "patient_group":
            missing_group_loaders = [
                label
                for label, loader in (
                    ("X_gt", X_gt),
                    ("X_syn", X_syn),
                    ("X_train", X_train),
                    ("X_ref_syn", X_ref_syn),
                    ("X_augmented", X_augmented),
                )
                if loader is not None and loader.group_ids is None
            ]
            if missing_group_loaders:
                return _group_unsafe_report(
                    metrics,
                    task_type,
                    semantic_context,
                    group_mode=group_mode,
                    loader_types=loader_types,
                    reason=(
                        "Patient-group SynthCity evaluation requires aligned group IDs for "
                        f"loader(s) {missing_group_loaders}"
                    ),
                )
        if grouped_loaders and task_type not in {"classification", "regression"}:
            return _group_unsafe_report(
                metrics,
                task_type,
                semantic_context,
                group_mode=group_mode,
                loader_types=loader_types,
            )

        resolved_feature_types = dict(
            feature_types or getattr(X_gt, "feature_types", {}) or {}
        )
        if not resolved_feature_types:
            for column in X_gt.dataframe().columns:
                series = X_gt.dataframe()[column]
                resolved_feature_types[column] = (
                    "categorical"
                    if (
                        pd.api.types.is_object_dtype(series)
                        or isinstance(series.dtype, pd.CategoricalDtype)
                        or pd.api.types.is_bool_dtype(series)
                    )
                    else "continuous"
                )
        resolved_source_table = dict(
            source_table or getattr(X_gt, "source_table", {}) or {}
        )

        if X_gt.type() != X_syn.type():
            raise ValueError("Different dataloader types")

        if task_type == "survival_analysis":
            if (
                X_gt.type() != "survival_analysis"
                and X_train.type() != "survival_analysis"
            ):
                raise ValueError("Invalid dataloader for survival analysis")
        elif task_type == "time_series":
            if X_gt.type() != "time_series" and X_train.type() != "time_series":
                raise ValueError("Invalid dataloader for time series")
        elif task_type == "time_series_survival":
            if (
                X_gt.type() != "time_series_survival"
                and X_train.type() != "time_series_survival"
            ):
                raise ValueError("Invalid dataloader for time series survival analysis")

        if metrics is None:
            metrics = Metrics.list()

        resolved_sensitive_target_types = dict(sensitive_target_types or {})
        if metrics.get("attack"):
            missing_target_types = sorted(
                set(X_gt.sensitive_features) - set(resolved_sensitive_target_types)
            )
            if missing_target_types:
                raise ValueError(
                    "Attribute-inference attacks require schema-defined sensitive_target_types "
                    f"for targets: {missing_target_types}"
                )

        domias_selected = any(
            "DomiasMIA" in metric_name
            for metric_names in metrics.values()
            for metric_name in metric_names
        )
        if domias_selected:
            if X_train is None or X_ref_syn is None:
                raise ValueError("DOMIAS metrics require X_train and X_ref_syn loaders")
            resolved_domias_reference_size = (
                len(X_gt) // 2 if domias_reference_size is None else domias_reference_size
            )
            if (
                isinstance(resolved_domias_reference_size, bool)
                or not isinstance(resolved_domias_reference_size, int)
                or resolved_domias_reference_size < 1
            ):
                raise ValueError(
                    "domias_reference_size must be a positive integer or None, got "
                    f"{domias_reference_size!r}"
                )
        else:
            resolved_domias_reference_size = None

        """Fit categorical encoders on real fit data and transform every other role.

        Unknown categories are deliberately rejected by the fitted encoders. A
        category appearing only in evidence or synthetic data must be recorded
        as an evaluation failure rather than changing the feature mapping.
        """
        fit_loader = X_train if X_train is not None else X_gt
        _, encoders = fit_loader.encode()

        # now we encode the data
        X_gt, _ = X_gt.encode(encoders)
        X_syn, _ = X_syn.encode(encoders)

        if X_train:
            X_train, _ = X_train.encode(encoders)
        if X_ref_syn:
            X_ref_syn, _ = X_ref_syn.encode(encoders)
        if X_augmented:
            X_augmented, _ = X_augmented.encode(encoders)

        scores = ScoreEvaluator()

        def _metric_instance(metric):
            metric_args = {
                "reduction": reduction,
                "n_histogram_bins": n_histogram_bins,
                "task_type": task_type,
                "random_state": random_state,
                "workspace": workspace,
                "use_cache": use_cache,
                "n_folds": n_folds,
            }
            if metric.type() == "attack":
                metric_args.update(
                    {
                        "quasi_identifier_columns": quasi_identifier_columns,
                        "sensitive_target_types": resolved_sensitive_target_types,
                        "classification_score": classification_score,
                    }
                )
            if metric.name() in {
                "delta-presence",
                "k-anonymization",
                "k-map",
                "distinct l-diversity",
            }:
                metric_args.update(
                    {
                        "structural_n_clusters": tuple(
                            structural_n_clusters or STRUCTURAL_KMEANS_N_CLUSTERS
                        ),
                        "structural_min_rows_per_cluster": structural_min_rows_per_cluster,
                    }
                )
            if metric.name() == "jensenshannon_dist":
                metric_args.update(
                    feature_types=resolved_feature_types,
                    source_table=resolved_source_table,
                )
            return metric(**metric_args)

        eval_cnt = min(len(X_gt), len(X_syn))
        for metric in standard_metrics:
            if metric.type() not in metrics:
                continue
            if metric.name() not in metrics[metric.type()]:
                continue
            if X_augmented and "augmentation" in metric.name():
                scores.queue(
                    _metric_instance(metric),
                    X_gt,
                    X_augmented,
                )
            elif "DomiasMIA" in metric.name():
                scores.queue(
                    _metric_instance(metric),
                    X_gt,
                    X_syn,
                    X_train,
                    X_ref_syn,
                    reference_size=resolved_domias_reference_size,
                )
            else:
                scores.queue(
                    _metric_instance(metric),
                    X_gt.sample(eval_cnt),
                    X_syn.sample(eval_cnt),
                )

        scores.compute()

        report = scores.to_dataframe()
        report.attrs["group_mode"] = group_mode
        if semantic_context is not None:
            report.attrs["semantic_context"] = dict(semantic_context)
        return report

    @staticmethod
    def list() -> dict:
        available_metrics: Dict[str, List] = {}
        for metric in standard_metrics:
            if metric.type() not in available_metrics:
                available_metrics[metric.type()] = []
            available_metrics[metric.type()].append(metric.name())

        return available_metrics
