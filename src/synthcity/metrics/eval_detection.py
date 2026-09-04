# stdlib
import math
import platform
from typing import Any, Dict, Optional

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics.core import MetricEvaluator
from synthcity.plugins.core.dataloader import DataLoader, _group_codes
from synthcity.plugins.core.dataset import NumpyDataset
from synthcity.plugins.core.models.convnet import suggest_image_classifier_arch
from synthcity.plugins.core.models.mlp import MLP
from synthcity.utils.reproducibility import clear_cache
from synthcity.utils.serialization import load_from_file, save_to_file

DETECTION_RESULT_VERSION = "auc-v2"


def _detection_groups(
    X_gt: DataLoader,
    X_syn: DataLoader,
) -> Optional[np.ndarray]:
    gt_groups = getattr(X_gt, "group_ids", None)
    syn_groups = getattr(X_syn, "group_ids", None)
    if gt_groups is None and syn_groups is None:
        return None
    if gt_groups is None or syn_groups is None:
        raise ValueError(
            "Grouped detection requires group_ids for both real and synthetic populations"
        )
    return _group_codes(list(gt_groups) + list(syn_groups))


def _detection_cv_splits(
    X_gt: DataLoader,
    X_syn: DataLoader,
    data: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
    random_state: int,
):
    groups = _detection_groups(X_gt, X_syn)
    if groups is None:
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        return splitter.split(data, labels)
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    return splitter.split(data, labels, groups)


class DetectionEvaluator(MetricEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.DetectionEvaluator
        :parts: 1


    Train a SKLearn classifier to detect the synthetic data from real data.

    Synthetic and real data are combined to form a new dataset.
    K-fold cross validation is performed to see how well a classifier can distinguish real from synthetic.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def type() -> str:
        return "detection"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def effective_auc(raw_auc: float) -> float:
        """Return inversion-aware attacker AUC while preserving raw AUC separately."""
        raw_auc = float(raw_auc)
        if not math.isfinite(raw_auc) or not 0.0 <= raw_auc <= 1.0:
            raise ValueError(f"AUC must be finite and within [0, 1], got {raw_auc!r}")
        return max(raw_auc, 1.0 - raw_auc)

    def result_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": DETECTION_RESULT_VERSION,
            "raw_auc_key": "raw_auc",
            "reducer": self._reduction,
            "effective_auc_key": "effective_auc_v2",
            "default_key": "effective_auc_v2",
            "chance_auc": 0.5,
        }

    @staticmethod
    def name() -> str:
        raise NotImplementedError()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        raise NotImplementedError()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_detection_generic(
        self,
        model_template: Any,
        X_gt: DataLoader,
        X_syn: DataLoader,
        **model_args: Any,
    ) -> Dict:
        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{DETECTION_RESULT_VERSION}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            results = load_from_file(cache_file)
            log.info(
                f" Synthetic-real data discrimination using {self.name()}. AUCROC : {results}"
            )
            return results

        arr_gt = X_gt.numpy().reshape(len(X_gt), -1)
        labels_gt = np.asarray([0] * len(X_gt))

        arr_syn = X_syn.numpy().reshape(len(X_syn), -1)
        labels_syn = np.asarray([1] * len(X_syn))

        data = np.concatenate([arr_gt, arr_syn])
        labels = np.concatenate([labels_gt, labels_syn])
        groups = _detection_groups(X_gt, X_syn)

        res = []

        cv_splits = _detection_cv_splits(
            X_gt,
            X_syn,
            data,
            labels,
            self._n_folds,
            self._random_state,
        )
        for train_idx, test_idx in cv_splits:
            train_data = data[train_idx]
            train_labels = labels[train_idx]
            test_data = data[test_idx]
            test_labels = labels[test_idx]

            model = model_template(**model_args)
            if groups is not None and isinstance(model, MLP):
                model.fit(
                    train_data.astype(float),
                    train_labels,
                    groups=groups[train_idx],
                )
            else:
                model.fit(train_data.astype(float), train_labels)

            test_pred = model.predict_proba(test_data.astype(float))[:, 1]

            score = roc_auc_score(test_labels, test_pred)
            res.append(score)

        raw_auc = float(self.reduction()(res))
        results = {
            self._reduction: raw_auc,
            "mean": float(np.mean(res)),
            "raw_auc": raw_auc,
            "effective_auc_v2": self.effective_auc(raw_auc),
        }
        log.info(
            f" Synthetic-real data discrimination using {self.name()}. AUCROC : {results}"
        )

        save_to_file(cache_file, results)

        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> float:
        return self.evaluate(X_gt, X_syn)["effective_auc_v2"]


class SyntheticDetectionXGB(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionXGB
        :parts: 1

    Train a XGBoostclassifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_xgb"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        # TODO: investigate why XGBoost always has high AUCROC for the detection
        model_template = XGBClassifier
        model_args = {
            "n_jobs": 2,
            "verbosity": 0,
            "depth": 3,
            "random_state": self._random_state,
        }

        return self._evaluate_detection_generic(
            model_template, X_gt, X_syn, **model_args
        )


class SyntheticDetectionMLP(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionMLP
        :parts: 1

    Train a MLP classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_mlp"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _evaluate_image_detection(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        clear_cache()

        cache_file = (
            self._workspace
            / f"sc_metric_cache_{self.type()}_{self.name()}_{DETECTION_RESULT_VERSION}_{X_gt.hash()}_{X_syn.hash()}_{self._reduction}_{platform.python_version()}.bkp"
        )
        if self.use_cache(cache_file):
            results = load_from_file(cache_file)
            log.info(
                f" Synthetic-real data discrimination using {self.name()}. AUCROC : {results}"
            )
            return results

        data_gt = X_gt.numpy()
        data_syn = X_syn.numpy()
        data = np.concatenate([data_gt, data_syn], axis=0)

        labels_gt = np.asarray([0] * len(X_gt))
        labels_syn = np.asarray([1] * len(X_syn))
        labels = np.concatenate([labels_gt, labels_syn])
        groups = _detection_groups(X_gt, X_syn)

        cv_splits = _detection_cv_splits(
            X_gt,
            X_syn,
            data,
            labels,
            self._n_folds,
            self._random_state,
        )
        res = []
        for train_idx, test_idx in cv_splits:
            train_X = data[train_idx]
            train_y = labels[train_idx]
            test_X = data[test_idx]
            test_y = labels[test_idx]

            clf = suggest_image_classifier_arch(
                n_channels=X_gt.info()["channels"],
                height=X_gt.info()["height"],
                width=X_gt.info()["width"],
                classes=2,
            )
            train_dataset = NumpyDataset(train_X, train_y)

            clf.fit(
                train_dataset,
                groups=None if groups is None else groups[train_idx],
            )
            test_pred = clf.predict_proba(torch.from_numpy(test_X))[:, 1].cpu().numpy()

            score = roc_auc_score(test_y, test_pred)
            res.append(score)

        raw_auc = float(self.reduction()(res))
        results = {
            self._reduction: raw_auc,
            "mean": float(np.mean(res)),
            "raw_auc": raw_auc,
            "effective_auc_v2": self.effective_auc(raw_auc),
        }
        log.info(
            f" Synthetic-real data discrimination using {self.name()}. AUCROC : {results}"
        )

        save_to_file(cache_file, results)
        return results

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        if X_gt.type() == "images":
            return self._evaluate_image_detection(X_gt, X_syn)

        model_args = {
            "task_type": "classification",
            "n_units_in": X_gt.shape[1],
            "n_units_out": 2,
            "random_state": self._random_state,
        }
        return self._evaluate_detection_generic(
            MLP,
            X_gt,
            X_syn,
            **model_args,
        )


class SyntheticDetectionLinear(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionLinear
        :parts: 1

    Train a LogisticRegression classifier to detect the synthetic data.

    Returns:
        The average AUCROC score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_linear"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self, X_gt: DataLoader, X_syn: DataLoader) -> Dict:
        model_args = {
            "random_state": self._random_state,
            "n_jobs": -1,
            "max_iter": 10000,
        }
        return self._evaluate_detection_generic(
            LogisticRegression,
            X_gt,
            X_syn,
            **model_args,
        )


class SyntheticDetectionGMM(DetectionEvaluator):
    """
    .. inheritance-diagram:: synthcity.metrics.eval_detection.SyntheticDetectionGMM
        :parts: 1

    Train a GaussianMixture model to detect synthetic data.

    Returns:
        The average score for detecting synthetic data.

    Score:
        0: The datasets are indistinguishable.
        1: The datasets are totally distinguishable.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "detection_gmm"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
    ) -> Dict:
        del X_gt, X_syn
        raise RuntimeError(
            "SyntheticDetectionGMM is audit-only: GaussianMixture does not provide "
            "a valid binary real-versus-synthetic detector contract"
        )
