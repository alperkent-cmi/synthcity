# stdlib
import math
import time
from collections.abc import Mapping
from numbers import Real
from typing import Any, Dict, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from scipy.stats import iqr
from tqdm import tqdm

# synthcity absolute
import synthcity.logger as log

# synthcity relative
from .core.metric import MetricEvaluator

REPEATED_SEED_UNCERTAINTY_SCHEMA_VERSION = "repeated-seed-uncertainty-v1"


def _safe_evaluate(
    evaluator: MetricEvaluator,
    *args: Any,
    **kwargs: Any,
) -> Tuple[str, Dict, bool, float, str, Optional[str], Optional[str]]:
    start = time.time()
    log.debug(f" >> Evaluating metric {evaluator.fqdn()}")
    failed = False
    err = None
    error_type = None
    try:
        result = evaluator.evaluate(*args, **kwargs)
    except Exception as exc:
        err = str(exc)
        error_type = type(exc).__name__
        result = {}
        failed = True

    duration = float(time.time() - start)
    log.debug(f" >> Evaluating metric {evaluator.fqdn()} done. Duration: {duration} s")

    if err is not None:
        log.error(f" >> Evaluator {evaluator.fqdn()} failed: {err}")

    return evaluator.fqdn(), result, failed, duration, evaluator.direction(), err, error_type


class ScoreEvaluator:
    def __init__(self) -> None:
        self.scores: dict = {}
        self.result_metadata: dict = {}
        self._repeated_result_metadata: dict[str, list[dict[str, Any]]] = {}
        self.pending_tasks: list = []

    def add_result_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        repeat: int | None = None,
    ) -> None:
        if not isinstance(metadata, Mapping):
            raise TypeError("Metric result metadata must be a mapping")
        if repeat is not None and (isinstance(repeat, bool) or not isinstance(repeat, int)):
            raise TypeError("Metric metadata repeat must be an integer or None")

        for key, value in metadata.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"Metric result metadata for {key!r} must be a mapping")
            normalized_key = str(key)
            normalized_value = dict(value)
            if repeat is None:
                self.result_metadata[normalized_key] = normalized_value
                continue

            entries = self._repeated_result_metadata.setdefault(normalized_key, [])
            if any(entry["repeat"] == repeat for entry in entries):
                raise ValueError(
                    f"Metric result metadata already contains repeat {repeat} for {normalized_key!r}"
                )
            entries.append(
                {
                    "repeat": int(repeat),
                    "metadata": normalized_value,
                }
            )
            self.result_metadata[normalized_key] = normalized_value

    def add(
        self,
        key: str,
        result: float,
        failed: int,
        duration: float,
        direction: str,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        if isinstance(result, bool) or not isinstance(result, Real):
            result_kind = "non-real"
        else:
            try:
                result_kind = None if math.isfinite(float(result)) else "non-finite"
            except (OverflowError, TypeError, ValueError):
                result_kind = "non-finite"
        if result_kind is not None:
            original_result = result
            result = np.nan
            failed = 1
            error = error or (
                f"Metric evaluator returned a {result_kind} result: {original_result!r}"
            )
            error_type = error_type or (
                "InvalidMetricResult"
                if result_kind == "non-real"
                else "NonFiniteMetricResult"
            )
        if key not in self.scores:
            self.scores[key] = {
                "values": [],
                "errors": 0,
                "durations": [],
                "direction": direction,
                "error_messages": [],
                "error_types": [],
            }
        self.scores[key]["durations"].append(duration)
        self.scores[key]["errors"] += int(failed)
        self.scores[key]["values"].append(result)
        if error:
            self.scores[key]["error_messages"].append(error)
        if error_type:
            self.scores[key]["error_types"].append(error_type)

    def add_multiple(
        self,
        key: str,
        results: Dict,
        failed: int,
        duration: float,
        direction: str,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> None:
        if not results:
            self.add(
                key,
                np.nan,
                failed=1,
                duration=duration,
                direction=direction,
                error=error or "Metric evaluator returned no result values",
                error_type=error_type or "EmptyMetricResult",
            )
            return
        for subkey in results:
            self.add(
                f"{key}.{subkey}",
                results[subkey],
                failed,
                duration,
                direction,
                error=error,
                error_type=error_type,
            )

    def queue(
        self,
        evaluator: MetricEvaluator,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.pending_tasks.append((evaluator, args, kwargs))

    def compute(self) -> None:
        # Metrics previously ran via `joblib.Parallel(n_jobs=1)`, which is
        # sequential in this process anyway -- iterate directly (instead of
        # hiding the loop inside joblib's generator dispatch) so a tqdm bar
        # can show which metric is currently running. Without this there is
        # no way to tell a merely-slow metric (e.g. DomiasMIA*, which can
        # take minutes) from a genuinely hung one. Mirrors the per-metric
        # progress bar added to the syntheval fork (`SynthEval.evaluate()`).
        pbar = tqdm(self.pending_tasks, desc="synthcity metrics", unit="metric")
        results = []
        for evaluator, args, kwargs in pbar:
            pbar.set_postfix_str(evaluator.fqdn(), refresh=True)
            key, result, failed, duration, direction, err, error_type = _safe_evaluate(
                evaluator, *args, **kwargs
            )
            metadata_getter = getattr(evaluator, "result_metadata", None)
            if callable(metadata_getter):
                metadata = metadata_getter()
                if metadata:
                    self.add_result_metadata({key: metadata})
            if failed:
                pbar.write(
                    f"[synthcity] '{key}' failed after {duration:.1f}s: {err}"
                )
            results.append((key, result, failed, duration, direction, err, error_type))
        self.pending_tasks = []

        for key, result, failed, duration, direction, err, error_type in results:
            self.add_multiple(
                key,
                result,
                failed,
                duration,
                direction,
                error=err,
                error_type=error_type,
            )

    def to_dataframe(self) -> pd.DataFrame:
        output_metrics = [
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
        output_rows = []
        for metric in self.scores:
            errors = self.scores[metric]["errors"]
            direction = self.scores[metric]["direction"]
            durations = round(np.mean(self.scores[metric]["durations"]), 2)
            values = self.scores[metric]["values"]
            error_types = "; ".join(sorted(set(self.scores[metric]["error_types"])))
            error_messages = " | ".join(self.scores[metric]["error_messages"])
            score_min = np.min(values)
            score_max = np.max(values)
            score_mean = np.mean(values)
            score_median = np.median(values)
            score_stddev = np.std(values)
            score_iqr = iqr(values)
            score_rounds = len(values)
            output_rows.append(
                [
                    score_min,
                    score_max,
                    score_mean,
                    score_stddev,
                    score_median,
                    score_iqr,
                    score_rounds,
                    errors,
                    error_types,
                    error_messages,
                    durations,
                    direction,
                ]
            )

        if output_rows:
            output = pd.DataFrame(
                output_rows,
                columns=output_metrics,
                index=list(self.scores),
            )
            output["rounds"] = output["rounds"].astype(object)
            output["errors"] = output["errors"].astype(object)
            score_metrics = ["min", "max", "mean", "stddev", "median", "iqr"]
            if output[score_metrics].isna().all().all():
                output[score_metrics] = output[score_metrics].astype(object)
        else:
            output = pd.DataFrame([], columns=output_metrics)

        result_metadata = {
            key: dict(value) for key, value in self.result_metadata.items()
        }
        for key, entries in self._repeated_result_metadata.items():
            ordered_entries = sorted(entries, key=lambda entry: entry["repeat"])
            metadata = dict(ordered_entries[0]["metadata"])
            metadata["repeated_seed_uncertainty"] = {
                "schema_version": REPEATED_SEED_UNCERTAINTY_SCHEMA_VERSION,
                "method": "standard_deviation_across_repeated_seeds_v1",
                "field": "stddev",
                "unit": "evaluation_repeat",
                "seeds": [entry["repeat"] for entry in ordered_entries],
                "repeat_count": len(ordered_entries),
                "ddof": 0,
                "estimable": len(ordered_entries) >= 2,
            }
            metadata["repeated_seed_metadata"] = [
                {
                    "seed": entry["repeat"],
                    "metadata": dict(entry["metadata"]),
                }
                for entry in ordered_entries
            ]
            result_metadata[key] = metadata

        output.attrs["metric_metadata"] = dict(result_metadata)
        output.attrs["result_metadata"] = dict(result_metadata)
        return output
