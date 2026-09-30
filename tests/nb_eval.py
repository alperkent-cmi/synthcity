# stdlib
import os
import shutil
from pathlib import Path
from time import time, time_ns
from typing import List
from uuid import uuid4

# third party
import click
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

_WORKSPACE_MARKER = ".synthcity-owned-workspace"
_OWNED_WORKSPACES: dict[Path, str] = {}


def _repository_root() -> Path:
    """Find outer SynthData root, or nearest standalone SynthCity checkout."""
    test_path = Path(__file__).resolve()
    candidates = tuple(test_path.parents)
    synthdata_roots = [
        candidate
        for candidate in candidates
        if (candidate / "synthdata" / "__init__.py").is_file()
        and (candidate / "pyproject.toml").is_file()
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


def _new_owned_workspace() -> Path:
    """Create unique notebook workspace below checkout-local scratch."""
    scratch_root = _repository_root() / "tmp"
    if scratch_root.is_symlink():
        raise RuntimeError(f"Refusing symlink scratch root: {scratch_root}")
    scratch_root = scratch_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    workspace = scratch_root / f"notebook-{os.getpid()}-{time_ns()}-{uuid4().hex}"
    if workspace.is_symlink():
        raise RuntimeError(f"Refusing symlink workspace: {workspace}")
    workspace = workspace.resolve()
    if not workspace.is_relative_to(scratch_root):
        raise RuntimeError(f"Refusing workspace outside checkout scratch: {workspace}")
    workspace.mkdir(parents=False, exist_ok=False)
    token = uuid4().hex
    marker = workspace / _WORKSPACE_MARKER
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
    """Remove only an owned notebook workspace after containment checks."""
    if workspace.is_symlink():
        raise RuntimeError(f"Refusing cleanup of symlink workspace: {workspace}")
    scratch_root = _repository_root() / "tmp"
    if scratch_root.is_symlink():
        raise RuntimeError(f"Refusing symlink scratch root: {scratch_root}")
    scratch_root = scratch_root.resolve()
    resolved = workspace.resolve()
    marker = resolved / _WORKSPACE_MARKER
    token = _OWNED_WORKSPACES.get(resolved)
    if not resolved.is_relative_to(scratch_root):
        raise RuntimeError(f"Refusing cleanup outside checkout scratch: {resolved}")
    if (
        token is None
        or not resolved.is_dir()
        or resolved.is_symlink()
        or not marker.is_file()
        or marker.is_symlink()
    ):
        raise RuntimeError(f"Refusing cleanup of unowned workspace: {resolved}")
    if marker.read_text() != f"{token}\n":
        raise RuntimeError(f"Refusing cleanup with invalid ownership marker: {resolved}")
    shutil.rmtree(resolved)
    del _OWNED_WORKSPACES[resolved]


def run_notebook(notebook_path: Path, timeout: int, workspace: Path) -> None:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=timeout)
    # Will raise on cell error
    proc.preprocess(nb, {"metadata": {"path": workspace}})


try:
    # synthcity absolute
    from synthcity.plugins.core.models.tabular_goggle import TabularGoggle  # noqa: F401

    goggle_disabled = False
except ImportError:
    goggle_disabled = True

all_tests = [
    "basic_examples",
    "benchmarks",
    "survival_analysis",
    "time_series",
    "time_series_data_preparation",
    "hyperparameter_optimization",
    "plugin_adsgan",
    "plugin_ctgan",
    "plugin_nflow",
    "plugin_tvae",
    "plugin_radialgan",
    "plugin_arf",
    "plugin_bayesian_network",
    "plugin_ddpm",
    "plugin_dummy_sampler",
    "plugin_marginal_distribution",
    "plugin_uniform_sampler",
    "plugin_image_adsgan",
    "plugin_image_cgan",
    "plugin_decaf",
    "plugin_dpgan",
    "plugin_pategan",
    "plugin_privbayes",
    "plugin_ctgan(generic)",
    "plugin_fourier_flows",
    "plugin_timegan",
    "plugin_aim",
    "plugin_arf",
    "plugin_great",
]

if not goggle_disabled:
    all_tests.append("plugin_goggle")

minimal_tests = [
    "basic_examples",
    "plugin_adsgan",
    "plugin_ctgan",
    "plugin_nflow",
    "plugin_tvae",
]

# For extras
goggle_tests = ["plugin_goggle"]


@click.command()
@click.option("--nb_dir", type=str, default=".")
@click.option(
    "--tutorial_tests",
    type=click.Choice(
        ["minimal_tests", "all_tests", "goggle_tests"],
        case_sensitive=False,
    ),
    default="minimal_tests",
)
@click.option(
    "--timeout",
    type=int,
    default=1800,
    help="Timeout for notebook execution in seconds.",
)
def main(nb_dir: Path, tutorial_tests: str, timeout: int) -> None:
    nb_dir = Path(nb_dir)
    enabled_tests: List = []
    if tutorial_tests == "all_tests":
        enabled_tests = all_tests
    elif tutorial_tests == "minimal_tests":
        enabled_tests = minimal_tests

    workspace = _new_owned_workspace()
    try:
        for p in nb_dir.rglob("*"):
            if p.suffix != ".ipynb":
                continue

            if "checkpoint" in p.name:
                continue

            ignore = True
            for val in enabled_tests:
                if val in p.name:
                    ignore = False
                    break

            if ignore:
                continue

            print("Testing ", p.name)
            start = time()
            try:
                run_notebook(p, timeout, workspace)
            except BaseException as e:
                print("FAIL", p.name, e)

                raise e
            finally:
                print(f"Tutorial {p.name} tool {time() - start}")
    finally:
        _remove_owned_workspace(workspace)


if __name__ == "__main__":
    main()
