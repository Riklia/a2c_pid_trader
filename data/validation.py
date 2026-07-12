import pandas as pd


def require_columns(df: pd.DataFrame | pd.Series, required: set[str], context: str) -> None:
    """Raises ValueError if required fields are missing, with a clear message.

    Args:
        df: DataFrame or Series; a Series' index is treated as its field names.
        required: field names that must be present.
        context: short label of the caller for the error message, e.g. "StaticPID.act".
    """
    have = set(df.columns) if isinstance(df, pd.DataFrame) else set(df.index)
    missing = required - have
    if missing:
        raise ValueError(f"{context}: missing required fields: {sorted(missing)}")


def require_sorted_unique_datetime_index(df: pd.DataFrame, context: str) -> None:
    """Raises ValueError if the index is not a sorted, unique DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{context}: index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{context}: index must be sorted ascending")
    if df.index.has_duplicates:
        raise ValueError(f"{context}: index has duplicate timestamps")
