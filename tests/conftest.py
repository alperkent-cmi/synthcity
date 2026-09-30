# stdlib
import os
import time
from pathlib import Path
from typing import Generator
from uuid import uuid4

# third party
import pytest

# synthcity absolute
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results


def _repository_root() -> Path:
    """Find outer SynthData root, or nearest standalone SynthCity checkout."""
    test_path = Path(__file__).resolve()
    candidates = tuple(test_path.parents)
    synthdata_roots = [
        candidate.resolve()
        for candidate in candidates
        if (candidate.resolve() / "synthdata" / "__init__.py").is_file()
        and (candidate.resolve() / "pyproject.toml").is_file()
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
    raise RuntimeError(f"Could not identify checkout root for {test_path}")


REPOSITORY_ROOT = _repository_root()


def _reject_symlink_components(path: Path) -> None:
    """Reject scratch path components that could redirect pytest outside checkout."""
    current = Path(path.anchor) if path.is_absolute() else Path()
    for component in path.parts[1:] if path.is_absolute() else path.parts:
        current /= component
        if current.is_symlink():
            raise RuntimeError(f"Refusing symlink in pytest scratch path: {current}")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Set unique checkout-local basetemp before pytest creates tmp fixtures."""
    scratch_root = REPOSITORY_ROOT / "tmp" / "pytest"
    _reject_symlink_components(scratch_root)
    scratch_root = scratch_root.resolve()
    run_id = f"pytest-{os.getpid()}-{time.time_ns()}-{uuid4().hex}"
    basetemp = scratch_root / run_id
    _reject_symlink_components(basetemp)
    basetemp = basetemp.resolve()
    if not basetemp.is_absolute() or not basetemp.is_relative_to(scratch_root):
        raise RuntimeError(f"Refusing pytest basetemp outside checkout scratch: {basetemp}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    basetemp.mkdir(parents=False, exist_ok=False)
    config.option.basetemp = str(basetemp)


@pytest.fixture(autouse=True, scope="session")
def run_before_tests() -> Generator:
    enable_reproducible_results(0)
    clear_cache()

    yield
