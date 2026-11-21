"""Load and validate the raw weekly ward dataset (CSV or Excel)."""
from pathlib import Path

import pandas as pd

from dengue.config import RAW_DATA_PATH, REQUIRED_COLUMNS, TARGET_COL


def _parse_dates(s: pd.Series) -> pd.Series:
    """Accept real dates, ISO strings, or Excel serial day numbers."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s).dt.normalize()
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        # Excel stores dates as days since 1899-12-30.
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30").dt.normalize()
    return pd.to_datetime(s).dt.normalize()


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = _parse_dates(df["date"])
    validate_raw(df)
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def validate_raw(df: pd.DataFrame) -> None:
    """Fail fast with a clear message if the data is not usable."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    nulls = df[REQUIRED_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise ValueError(f"Dataset has empty values: {nulls.to_dict()}")

    dupes = int(df.duplicated(["ward_id", "date"]).sum())
    if dupes:
        raise ValueError(f"Dataset has {dupes} duplicate (ward_id, date) rows")

    gaps = df.sort_values(["ward_id", "date"]).groupby("ward_id")["date"].diff().dropna()
    bad = gaps[gaps != pd.Timedelta(days=7)]
    if not bad.empty:
        raise ValueError(f"Dataset is not strictly weekly: {len(bad)} irregular gaps found")

    if (df[TARGET_COL] < 0).any():
        raise ValueError("Dataset has negative dengue case counts")
