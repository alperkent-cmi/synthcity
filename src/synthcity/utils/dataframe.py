# stdlib
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Iterator, Optional

# third party
import pandas as pd

# Columns the caller declared discrete. When set, every discrete/continuous
# decision in synthcity (encoders, TabDDPM, ARF, metrics) follows it instead of
# guessing from the number of distinct values.
_DECLARED_DISCRETE: ContextVar[Optional[frozenset]] = ContextVar(
    "synthcity_declared_discrete", default=None
)


@contextmanager
def declared_discrete_columns(columns: Iterable) -> Iterator[None]:
    """Treat exactly ``columns`` as discrete (all others continuous) inside the block."""
    token = _DECLARED_DISCRETE.set(frozenset(columns))
    try:
        yield
    finally:
        _DECLARED_DISCRETE.reset(token)


def is_discrete(name: object, values: pd.Series, max_classes: int) -> bool:
    """Declared type of ``name`` if declared, else the distinct-value heuristic."""
    declared = _DECLARED_DISCRETE.get()
    if declared is not None:
        return name in declared
    return values.nunique() <= max_classes


def constant_columns(dataframe: pd.DataFrame) -> list:
    """
    Find constant value columns in a pandas dataframe.
    """
    return [col for col, vals in dataframe.items() if vals.nunique() <= 1]


def discrete_columns(
    dataframe: pd.DataFrame, max_classes: int = 10, return_counts: bool = False
) -> list:
    """
    Find columns containing discrete values in a pandas dataframe.

    Inside ``declared_discrete_columns`` the declared columns are returned.
    """
    return [
        (col, vals.nunique()) if return_counts else col
        for col, vals in dataframe.items()
        if is_discrete(col, vals, max_classes)
    ]
