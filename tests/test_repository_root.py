"""Source-checkout detection for repository-local test scratch."""

from pathlib import Path

import pytest
from conftest import _repository_root


def test_checkout_selection_uses_source_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_root = _repository_root()
    is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: (
            (
                path.name == "pyproject.toml"
                or (path.name == "__init__.py" and path.parent.name == "synthdata")
            )
            and is_file(path)
        ),
    )

    assert _repository_root() == expected_root
