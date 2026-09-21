import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
import synthcity.metrics.eval_detection as detection
from synthcity.utils.optimizer import search_parameters


def _repository_root() -> Path:
    """Find outer SynthData root, or this standalone SynthCity checkout root."""
    test_path = Path(__file__).resolve()
    candidates = tuple(test_path.parents)
    synthdata_roots = [
        candidate
        for candidate in candidates
        if (candidate / "AGENTS.md").is_file() and (candidate / "pyproject.toml").is_file()
    ]
    if synthdata_roots:
        return synthdata_roots[-1]

    checkout_roots = [
        candidate
        for candidate in candidates
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").is_file()
    ]
    if checkout_roots:
        return checkout_roots[0]
    raise RuntimeError(f"Could not identify repository root for {test_path}")


REPOSITORY_ROOT = _repository_root()
_OWNED_WORKSPACES: dict[Path, str] = {}
_OWNER_MARKER = ".synthcity-test-owner"


def _owned_workspace(name: str) -> Path:
    """Create one validated, checkout-local workspace owned by this test."""
    tmp_root = REPOSITORY_ROOT / "tmp"
    scratch_root = tmp_root / "synthcity-tests"
    if tmp_root.is_symlink() or scratch_root.is_symlink():
        raise RuntimeError(f"Refusing symlink scratch root: {scratch_root}")
    scratch_root = scratch_root.resolve()
    workspace = scratch_root / f"{name}-{os.getpid()}-{uuid4().hex}"
    if workspace.is_symlink():
        raise RuntimeError(f"Refusing symlink workspace: {workspace}")
    workspace = workspace.resolve()
    if not workspace.is_relative_to(scratch_root):
        raise RuntimeError(f"Refusing workspace outside checkout scratch: {workspace}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=False, exist_ok=False)
    marker = workspace / _OWNER_MARKER
    token = f"{os.getpid()}:{uuid4().hex}"
    try:
        with marker.open("x", encoding="utf-8") as marker_file:
            marker_file.write(f"{token}\n")
    except OSError as exc:
        if workspace.is_dir() and not workspace.is_symlink() and not any(workspace.iterdir()):
            workspace.rmdir()
        else:
            raise RuntimeError(
                f"Workspace setup failed with ambiguous ownership state: {workspace}"
            ) from exc
        raise RuntimeError(f"Workspace marker setup failed: {workspace}") from exc
    _OWNED_WORKSPACES[workspace] = token
    return workspace


def _remove_owned_workspace(workspace: Path) -> None:
    """Remove only this test's validated workspace."""
    if workspace.is_symlink():
        raise RuntimeError(f"Refusing cleanup of symlink workspace: {workspace}")
    tmp_root = REPOSITORY_ROOT / "tmp"
    scratch_root = tmp_root / "synthcity-tests"
    if tmp_root.is_symlink() or scratch_root.is_symlink():
        raise RuntimeError(f"Refusing symlink scratch root: {scratch_root}")
    scratch_root = scratch_root.resolve()
    resolved = workspace.resolve()
    marker = resolved / _OWNER_MARKER
    if (
        not resolved.is_absolute()
        or not resolved.is_relative_to(scratch_root)
        or resolved.parent != scratch_root
        or resolved not in _OWNED_WORKSPACES
        or not resolved.is_dir()
        or not marker.is_file()
        or marker.is_symlink()
        or marker.read_text(encoding="utf-8") != f"{_OWNED_WORKSPACES.get(resolved)}\n"
    ):
        raise RuntimeError(f"Refusing cleanup outside owned scratch: {resolved}")
    shutil.rmtree(resolved)
    del _OWNED_WORKSPACES[resolved]


def test_search_parameters_uses_group_disjoint_hpo_data(monkeypatch) -> None:
    fitted_groups = []

    class RecordingModel:
        def __init__(self, **kwargs):
            del kwargs

        @staticmethod
        def name() -> str:
            return "recording"

        @staticmethod
        def hyperparameter_space() -> list:
            return []

        def fit(self, data, groups=None):
            assert groups is not None
            fitted_groups.append((set(data.index), tuple(groups)))
            return self

        def generate(self, count: int) -> pd.DataFrame:
            return pd.DataFrame(np.zeros((count, 1)), columns=["feature"])

    class RecordingMetric:
        def evaluate(self, real, synthetic):
            assert real.group_ids is not None
            assert synthetic.group_ids is not None
            assert len(synthetic.group_ids) == len(synthetic)
            return {"mean": 0.5}

    monkeypatch.setattr(
        "synthcity.utils.optimizer.create_study",
        lambda **kwargs: (None, None),
    )
    monkeypatch.setattr(
        detection,
        "SyntheticDetectionMLP",
        lambda: RecordingMetric(),
    )

    data = pd.DataFrame({"feature": np.arange(12, dtype=float)})
    groups = np.repeat([f"p{index}" for index in range(6)], 2)

    workspace = _owned_workspace("optimizer-grouped")
    try:
        result = search_parameters(
            RecordingModel,
            data,
            n_trials=0,
            workspace=workspace,
            groups=(group for group in groups),
        )
    finally:
        _remove_owned_workspace(workspace)

    assert result == {}
    assert len(fitted_groups) == 1
    _, train_groups = fitted_groups[0]
    assert all(train_groups.count(group) == 2 for group in set(train_groups))
