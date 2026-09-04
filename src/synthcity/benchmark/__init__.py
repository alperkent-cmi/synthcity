# stdlib
import hashlib
import json
import platform
import random
from copy import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from IPython.display import display
from pydantic import validate_call

# synthcity absolute
import synthcity.logger as log
from synthcity.benchmark.utils import augment_data, get_json_serializable_kwargs
from synthcity.metrics import Metrics
from synthcity.metrics.eval import GROUP_UNSAFE_MODALITIES, _group_unsafe_report
from synthcity.metrics.scores import ScoreEvaluator
from synthcity.plugins import Plugins
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
from synthcity.utils.serialization import load_from_file, save_to_file


def print_score(mean: pd.Series, std: pd.Series) -> pd.Series:
    pd.options.mode.chained_assignment = None

    mean.loc[(mean < 1e-3) & (mean != 0)] = 1e-3
    std.loc[(std < 1e-3) & (std != 0)] = 1e-3

    # coerce to numeric (empty or object dtypes become float64/NaN)
    mean_num = pd.to_numeric(mean, errors="coerce")
    std_num = pd.to_numeric(std, errors="coerce")

    # round & stringify
    mean_str = mean_num.round(3).astype(str)
    stddev_str = std_num.round(3).astype(str)

    # if both were empty, this will just return an empty Series
    return mean_str + " ± " + stddev_str


class Benchmarks:
    @staticmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        tests: List[Tuple[str, str, dict]],  # test name, plugin name, plugin args
        X: DataLoader,
        X_test: Optional[DataLoader] = None,
        metrics: Optional[Dict] = None,
        repeats: int = 3,
        synthetic_size: Optional[int] = None,
        synthetic_constraints: Optional[Constraints] = None,
        synthetic_cache: bool = True,
        synthetic_reuse_if_exists: bool = True,
        augmented_reuse_if_exists: bool = True,
        task_type: str = "classification",  # classification, regression, survival_analysis, time_series
        group_mode: str = "row",
        workspace: Path = Path("workspace"),
        augmentation_rule: str = "equal",
        strict_augmentation: bool = False,
        ad_hoc_augment_vals: Optional[Dict] = None,
        use_metric_cache: bool = True,
        n_eval_folds: int = 5,
        classification_score: str = "balanced_accuracy",
        semantic_context: Optional[Dict[str, Any]] = None,
        fit_on_X: bool = False,
        candidate_screen: Optional[Callable[[pd.DataFrame], Any]] = None,
        **generate_kwargs: Any,
    ) -> pd.DataFrame:
        """Benchmark the performance of several algorithms.

        Args:
            tests: List[Tuple[str, str, dict]]
                Tuples of form (testname: str, plugin_name, str, plugin_args: dict)
            X: DataLoader
                The baseline dataset to learn
            X_test: Optional[DataLoader]
                Optional test dataset for evaluation. If None, X will be split in train/test datasets.
            metrics:
                List of metrics to test. By default, all metrics are evaluated.
                Full dictionary of metrics is:
                {
                    'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
                    'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc', 'alpha_precision', 'survival_km_distance'],
                    'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
                    'detection': ['detection_xgb', 'detection_mlp', 'detection_linear'],
                    'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score', 'DomiasMIA_BNAF', 'DomiasMIA_KDE', 'DomiasMIA_prior']
                }
            repeats:
                Number of test repeats
            synthetic_size: int
                The size of the synthetic dataset. By default, it is len(X).
            synthetic_constraints:
                Optional constraints on the synthetic data. By default, it inherits the constraints from X.
            synthetic_cache: bool
                Enable experiment caching
            synthetic_reuse_if_exists: bool
                If the current synthetic dataset is cached, it will be reused for the experiments. Defaults to True.
            augmented_reuse_if_exists: bool
                If the current augmented dataset is cached, it will be reused for the experiments. Defaults to True.
            task_type: str
                The type of problem. Relevant for evaluating the downstream models with the correct metrics. Valid tasks are:  "classification", "regression", "survival_analysis", "time_series", "time_series_survival".
            group_mode: str
                The resolved population grouping policy. ``patient_group`` requires
                group-safe evaluator behavior and blocks unsupported modalities.
            workspace: Path
                Path for caching experiments. Default: "workspace".
            augmentation_rule: str
                The rule used to achieve the desired proportion records with each value in the fairness column. Possible values are: 'equal', 'log', and 'ad-hoc'. Defaults to "equal".
            strict_augmentation: bool
                Flag to ensure that the condition for generating synthetic data is strictly met. Defaults to False.
            ad_hoc_augment_vals: Dict
                A dictionary containing the number of each class to augment the real data with. This is only required if using the rule="ad-hoc" option. Defaults to None.
            use_metric_cache: bool
                If the current metric has been previously run and is cached, it will be reused for the experiments. Defaults to True.
            n_eval_folds: int
                the KFolds used by MetricEvaluators in the benchmarks. Defaults to 5.
            classification_score: str
                The categorical attribute-inference score. Valid values are
                ``balanced_accuracy`` and ``macro_f1``.
            semantic_context: Optional[Dict[str, Any]]
                Versioned root semantic context to retain on benchmark reports.
            fit_on_X: bool
                Fit generators directly on ``X`` and use ``X_test`` as the
                evaluation role. This disables SynthCity's historical internal
                train/test split for callers with explicit role loaders.
            candidate_screen: Optional[Callable[[pd.DataFrame], Any]]
                Optional callback invoked after candidate generation or cache
                reload and before any metric evaluation. Exceptions propagate
                so callers can persist a failed screen and prune a trial.
            plugin_kwargs:
                Optional kwargs for each algorithm. Example {"adsgan": {"n_iter": 10}},
        """
        out = {}

        if classification_score not in {"balanced_accuracy", "macro_f1"}:
            raise ValueError(
                "classification_score must be 'balanced_accuracy' or 'macro_f1', "
                f"got {classification_score!r}"
            )
        if group_mode not in {"row", "patient_group"}:
            raise ValueError(
                f"Invalid group mode {group_mode!r}. Supported: ['row', 'patient_group']"
            )

        loader_types = {"X": X.type()}
        if X_test is not None:
            loader_types["X_test"] = X_test.type()
        unsupported_modalities = sorted(
            {
                loader_type
                for loader_type in loader_types.values()
                if loader_type in GROUP_UNSAFE_MODALITIES
            }
        )
        if group_mode == "patient_group" and unsupported_modalities:
            reason = (
                "Patient-group SynthCity benchmarking is not supported for loader type(s) "
                f"{unsupported_modalities}; generator and metric internals require a "
                "group-aware implementation"
            )
            for testcase, _plugin, _kwargs in tests:
                report = _group_unsafe_report(
                    metrics,
                    task_type,
                    semantic_context,
                    group_mode=group_mode,
                    loader_types=loader_types,
                    reason=reason,
                )
                report.attrs["testcase"] = testcase
                out[testcase] = report
            return out

        if group_mode == "patient_group":
            missing_group_loaders = [
                label
                for label, loader in (("X", X), ("X_test", X_test))
                if loader is not None and getattr(loader, "group_ids", None) is None
            ]
            if missing_group_loaders:
                raise ValueError(
                    "Patient-group SynthCity benchmarking requires group IDs for "
                    f"loader(s): {missing_group_loaders}"
                )

        experiment_name = X.hash()

        workspace.mkdir(parents=True, exist_ok=True)

        plugin_cats = ["generic", "privacy", "domain_adaptation"]
        if X.type() == "images":
            plugin_cats.append("images")
        elif task_type == "survival_analysis":
            plugin_cats.append("survival_analysis")
        elif task_type == "time_series" or task_type == "time_series_survival":
            plugin_cats.append("time_series")

        for testcase, plugin, kwargs in tests:
            log.info(f"Testcase : {testcase}")
            if not isinstance(kwargs, dict):
                raise ValueError(f"'kwargs' must be a dict for {testcase}:{plugin}")

            scores = ScoreEvaluator()

            kwargs_hash = ""
            if len(kwargs) > 0:
                serializable_kwargs = get_json_serializable_kwargs(kwargs)
                kwargs_hash_raw = json.dumps(
                    serializable_kwargs, sort_keys=True
                ).encode()
                hash_object = hashlib.sha256(kwargs_hash_raw)
                kwargs_hash = hash_object.hexdigest()

            augmentation_arguments = {
                "augmentation_rule": augmentation_rule,
                "strict_augmentation": strict_augmentation,
                "ad_hoc_augment_vals": ad_hoc_augment_vals,
            }
            augmentation_arguments_hash_raw = json.dumps(
                copy(augmentation_arguments), sort_keys=True
            ).encode()
            augmentation_hash_object = hashlib.sha256(augmentation_arguments_hash_raw)
            augmentation_hash = augmentation_hash_object.hexdigest()

            repeats_list = list(range(repeats))
            random.shuffle(repeats_list)
            failure_report = None
            evaluation_group_safety = None

            def _generation_kwargs(namespace: str) -> dict:
                namespace_kwargs = dict(generate_kwargs)
                namespace_kwargs["_group_namespace"] = namespace
                return namespace_kwargs

            for repeat in repeats_list:
                enable_reproducible_results(repeat)

                kwargs["workspace"] = workspace
                kwargs["random_state"] = repeat

                clear_cache()

                X_syn_cache_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
                )
                X_ref_syn_cache_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_{repeat}_reference.bkp"
                )
                generator_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
                )
                X_augment_cache_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_augmentation_{augmentation_hash}_{kwargs_hash}_{platform.python_version()}_{repeat}.bkp"
                )
                augment_generator_file = (
                    workspace
                    / f"{experiment_name}_{testcase}_{plugin}_augmentation_{augmentation_hash}_{kwargs_hash}_{platform.python_version()}_generator_{repeat}.bkp"
                )

                log.info(
                    f"[testcase] Experiment repeat: {repeat} task type: {task_type} Train df hash = {experiment_name}"
                )
                fit_loader = X if fit_on_X else X.train()

                # TODO: caches should be from the same version of Synthcity. Different APIs will crash.
                if generator_file.exists() and synthetic_reuse_if_exists:
                    generator = load_from_file(generator_file)
                else:
                    try:
                        generator = Plugins(categories=plugin_cats).get(
                            plugin,
                            **kwargs,
                        )
                    except (OSError, TypeError, ValueError, RuntimeError) as exc:
                        log.error(
                            "[%s][take %d] generator construction failed for testcase=%s: %s",
                            plugin,
                            repeat,
                            testcase,
                            exc,
                        )
                        failure_report = pd.DataFrame(
                            [
                                {
                                    "status": "failed",
                                    "stage": "construction",
                                    "testcase": testcase,
                                    "plugin": plugin,
                                    "repeat": repeat,
                                    "error": str(exc),
                                    "error_type": type(exc).__name__,
                                }
                            ],
                            index=[testcase],
                        )
                        break

                    try:
                        generator.fit(fit_loader)
                    except (OSError, TypeError, ValueError, RuntimeError) as exc:
                        log.error(
                            "[%s][take %d] generator fit failed for testcase=%s: %s",
                            plugin,
                            repeat,
                            testcase,
                            exc,
                        )
                        failure_report = pd.DataFrame(
                            [
                                {
                                    "status": "failed",
                                    "stage": "fit",
                                    "testcase": testcase,
                                    "plugin": plugin,
                                    "repeat": repeat,
                                    "error": str(exc),
                                    "error_type": type(exc).__name__,
                                }
                            ],
                            index=[testcase],
                        )
                        break

                    if synthetic_cache:
                        save_to_file(generator_file, generator)

                accounting_getter = getattr(generator, "get_accounting_metadata", None)
                if accounting_getter is not None:
                    if not callable(accounting_getter):
                        raise TypeError(
                            f"Generator {plugin!r} exposes a non-callable accounting accessor"
                        )
                    accounting_metadata = accounting_getter()
                    if not isinstance(accounting_metadata, dict):
                        raise TypeError(
                            f"Generator {plugin!r} accounting metadata must be a dict, got "
                            f"{type(accounting_metadata).__name__}"
                        )
                    scores.result_metadata[f"generator.{plugin}"] = {
                        "schema_version": "generator-runtime-metadata-v2",
                        "plugin_name": plugin,
                        "plugin_fqdn": generator.fqdn(),
                        "requested_parameters": get_json_serializable_kwargs(
                            {key: value for key, value in kwargs.items() if key != "workspace"}
                        ),
                        "n_samples": int(synthetic_size if synthetic_size is not None else len(fit_loader)),
                        "random_state": int(repeat),
                        "privacy_claim_type": (
                            "formal_dp" if plugin == "pategan" else "none"
                        ),
                        "accounting": accounting_metadata,
                    }

                if X_syn_cache_file.exists() and synthetic_reuse_if_exists:
                    X_syn = load_from_file(X_syn_cache_file)
                else:
                    try:
                        X_syn = generator.generate(
                            count=synthetic_size,
                            constraints=synthetic_constraints,
                            **_generation_kwargs("synthetic"),
                        )
                        if len(X_syn) == 0:
                            raise RuntimeError("Plugin failed to generate data")
                    except (OSError, TypeError, ValueError, RuntimeError) as exc:
                        log.error(
                            "[%s][take %d] synthetic generation failed for testcase=%s: %s",
                            plugin,
                            repeat,
                            testcase,
                            exc,
                        )
                        failure_report = pd.DataFrame(
                            [
                                {
                                    "status": "failed",
                                    "stage": "synthetic_generation",
                                    "testcase": testcase,
                                    "plugin": plugin,
                                    "repeat": repeat,
                                    "error": str(exc),
                                    "error_type": type(exc).__name__,
                                }
                            ],
                            index=[testcase],
                        )
                        break

                    if synthetic_cache:
                        save_to_file(X_syn_cache_file, X_syn)

                if candidate_screen is not None:
                    candidate_screen(X_syn.dataframe())

                # X_ref_syn is the reference synthetic data used for DomiasMIA metrics
                if X_ref_syn_cache_file.exists() and synthetic_reuse_if_exists:
                    X_ref_syn = load_from_file(X_ref_syn_cache_file)
                else:
                    try:
                        X_ref_syn = generator.generate(
                            count=synthetic_size,
                            constraints=synthetic_constraints,
                            **_generation_kwargs("reference_synthetic"),
                        )
                        if len(X_ref_syn) == 0:
                            raise RuntimeError("Plugin failed to generate reference data")
                    except (OSError, TypeError, ValueError, RuntimeError) as exc:
                        log.error(
                            "[%s][take %d] reference generation failed for testcase=%s: %s",
                            plugin,
                            repeat,
                            testcase,
                            exc,
                        )
                        failure_report = pd.DataFrame(
                            [
                                {
                                    "status": "failed",
                                    "stage": "reference_generation",
                                    "testcase": testcase,
                                    "plugin": plugin,
                                    "repeat": repeat,
                                    "error": str(exc),
                                    "error_type": type(exc).__name__,
                                }
                            ],
                            index=[testcase],
                        )
                        break

                    if synthetic_cache:
                        save_to_file(X_ref_syn_cache_file, X_ref_syn)

                # Augmentation
                if metrics and any(
                    "augmentation" in metric
                    for metric in [x for v in metrics.values() for x in v]
                ):
                    if augment_generator_file.exists() and augmented_reuse_if_exists:
                        augment_generator = load_from_file(augment_generator_file)
                    else:
                        try:
                            augment_generator = Plugins(categories=plugin_cats).get(
                                plugin,
                                **kwargs,
                            )
                        except (OSError, TypeError, ValueError, RuntimeError) as exc:
                            log.error(
                                "[%s][take %d] augmentation generator construction failed "
                                "for testcase=%s: %s",
                                plugin,
                                repeat,
                                testcase,
                                exc,
                            )
                            failure_report = pd.DataFrame(
                                [
                                    {
                                        "status": "failed",
                                        "stage": "augmentation_construction",
                                        "testcase": testcase,
                                        "plugin": plugin,
                                        "repeat": repeat,
                                        "error": str(exc),
                                        "error_type": type(exc).__name__,
                                    }
                                ],
                                index=[testcase],
                            )
                            break
                        try:
                            if not X.get_fairness_column():
                                raise ValueError(
                                    "To use the augmentation metrics, `fairness_column` must be set to a string representing the name of a column in the DataLoader."
                                )
                            augment_generator.fit(
                                fit_loader,
                                cond=fit_loader[X.get_fairness_column()],
                            )
                        except (OSError, TypeError, ValueError, RuntimeError) as exc:
                            log.error(
                                "[%s][take %d] augmentation generator fit failed for "
                                "testcase=%s: %s",
                                plugin,
                                repeat,
                                testcase,
                                exc,
                            )
                            failure_report = pd.DataFrame(
                                [
                                    {
                                        "status": "failed",
                                        "stage": "augmentation_fit",
                                        "testcase": testcase,
                                        "plugin": plugin,
                                        "repeat": repeat,
                                        "error": str(exc),
                                        "error_type": type(exc).__name__,
                                    }
                                ],
                                index=[testcase],
                            )
                            break
                        if synthetic_cache:
                            save_to_file(augment_generator_file, augment_generator)

                    if X_augment_cache_file.exists() and augmented_reuse_if_exists:
                        X_augmented = load_from_file(X_augment_cache_file)
                    else:
                        try:
                            X_augmented = augment_data(
                                fit_loader,
                                augment_generator,
                                rule=augmentation_rule,
                                strict=strict_augmentation,
                                ad_hoc_augment_vals=ad_hoc_augment_vals,
                                **_generation_kwargs("augmented"),
                            )
                            if len(X_augmented) == 0:
                                raise RuntimeError("Plugin failed to generate data")
                        except (OSError, TypeError, ValueError, RuntimeError) as exc:
                            log.error(
                                "[%s][take %d] augmentation generation failed for "
                                "testcase=%s: %s",
                                plugin,
                                repeat,
                                testcase,
                                exc,
                            )
                            failure_report = pd.DataFrame(
                                [
                                    {
                                        "status": "failed",
                                        "stage": "augmentation_generation",
                                        "testcase": testcase,
                                        "plugin": plugin,
                                        "repeat": repeat,
                                        "error": str(exc),
                                        "error_type": type(exc).__name__,
                                    }
                                ],
                                index=[testcase],
                            )
                            break
                        if synthetic_cache:
                            save_to_file(X_augment_cache_file, X_augmented)
                else:
                    X_augmented = None
                try:
                    evaluation = Metrics.evaluate(
                        X_test if X_test is not None else X.test(),
                        X_syn,
                        fit_loader,
                        X_ref_syn,
                        X_augmented,
                        metrics=metrics,
                        task_type=task_type,
                        group_mode=group_mode,
                        classification_score=classification_score,
                        semantic_context=semantic_context,
                        random_state=repeat,
                        workspace=workspace,
                        use_cache=use_metric_cache,
                        n_folds=n_eval_folds,
                    )
                    if not isinstance(evaluation, pd.DataFrame):
                        raise TypeError("Metrics.evaluate must return a pandas DataFrame")
                    if evaluation.empty:
                        raise RuntimeError("Metrics.evaluate returned no metric rows")
                    required_columns = {"mean", "errors", "durations", "direction"}
                    missing_columns = required_columns.difference(evaluation.columns)
                    if missing_columns:
                        raise ValueError(
                            "Metrics.evaluate returned an incomplete result frame; "
                            f"missing columns={sorted(missing_columns)}"
                        )

                    evaluation_attrs = getattr(evaluation, "attrs", {})
                    evaluation_group_safety = evaluation_attrs.get("group_safety")
                    evaluation_metadata = evaluation_attrs.get(
                        "result_metadata",
                        evaluation_attrs.get("metric_metadata", {}),
                    )
                    if evaluation_metadata:
                        scores.add_result_metadata(evaluation_metadata, repeat=repeat)
                    mean_score = evaluation["mean"].to_dict()
                    errors = evaluation["errors"].to_dict()
                    duration = evaluation["durations"].to_dict()
                    direction = evaluation["direction"].to_dict()

                    for key in mean_score:
                        scores.add(
                            key,
                            mean_score[key],
                            errors[key],
                            duration[key],
                            direction[key],
                        )
                except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                    log.error(
                        "[%s][take %d] metric evaluation failed for testcase=%s: %s",
                        plugin,
                        repeat,
                        testcase,
                        exc,
                    )
                    failure_report = pd.DataFrame(
                        [
                            {
                                "status": "failed",
                                "stage": "metric_evaluation",
                                "testcase": testcase,
                                "plugin": plugin,
                                "repeat": repeat,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            }
                        ],
                        index=[testcase],
                    )
                    break
            if failure_report is not None:
                failure_report.attrs["group_mode"] = group_mode
                if semantic_context is not None:
                    failure_report.attrs["semantic_context"] = dict(semantic_context)
                out[testcase] = failure_report
            else:
                report = scores.to_dataframe()
                report.attrs["group_mode"] = group_mode
                if isinstance(evaluation_group_safety, dict):
                    report.attrs["group_safety"] = dict(evaluation_group_safety)
                if semantic_context is not None:
                    report.attrs["semantic_context"] = dict(semantic_context)
                out[testcase] = report

        return out

    @staticmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def print(
        results: Dict,
        only_comparatives: bool = True,
    ) -> None:
        pd.set_option("display.max_rows", None, "display.max_columns", None)

        means = []
        for plugin in results:
            mean = results[plugin]["mean"]
            stddev = results[plugin]["stddev"]
            means.append(print_score(mean, stddev))

        avg = pd.concat(means, axis=1)
        avg = avg.set_axis(results.keys(), axis=1)

        if len(means) > 1:
            print()
            print("\033[4m" + "\033[1m" + "Comparatives" + "\033[0m" + "\033[0m")
            display(avg)

            if only_comparatives:
                return

        for plugin in results:
            print()
            print("\033[4m" + "\033[1m" + f"Plugin : {plugin}" + "\033[0m" + "\033[0m")

            display(results[plugin].drop(columns=["direction"]))
            print()

    @staticmethod
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def highlight(
        results: Dict,
    ) -> None:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        means = []
        for plugin in results:
            data = results[plugin]["mean"]
            directions = results[plugin]["direction"].to_dict()
            means.append(data)

        out = pd.concat(means, axis=1)
        out = out.set_axis(list(results.keys()), axis=1, copy=False)

        bad_highlight = "background-color: lightcoral;"
        ok_highlight = "background-color: green;"
        default = ""

        def highlights(row: pd.Series) -> Any:
            metric = row.name
            if directions[metric] == "minimize":
                best_val = np.min(row.values)
                worst_val = np.max(row)
            else:
                best_val = np.max(row.values)
                worst_val = np.min(row)

            styles = []
            for val in row.values:
                if val == best_val:
                    styles.append(ok_highlight)
                elif val == worst_val:
                    styles.append(bad_highlight)
                else:
                    styles.append(default)

            return styles

        out.style.apply(highlights, axis=1)

        return out
