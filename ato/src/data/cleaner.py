import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd

from src.data.loader import load
from src.data.quality import check_data_quality, _print_report

DATA_DIR = Path(__file__).parents[2] / "data"
OUTPUT_PATH = DATA_DIR / "cleaned.csv"

TARGET_COL = "Is Account Takeover"
TIMESTAMP_COL = "Login Timestamp"
CRITICAL_NULL_COL_THRESHOLD = 0.50

CATEGORICAL_COLS = [
    "IP Address", "Country", "Region", "City",
    "User Agent String", "Browser Name and Version",
    "OS Name and Version", "Device Type",
]
NUMERIC_COLS = ["User ID", "ASN", "Round-Trip Time [ms]"]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _drop_high_null_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    null_rates = df.isnull().mean()
    drop_cols = null_rates[null_rates > CRITICAL_NULL_COL_THRESHOLD].index.tolist()
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} column(s) with >{CRITICAL_NULL_COL_THRESHOLD:.0%} nulls: {drop_cols}")
    return df.drop(columns=drop_cols), drop_cols


def _drop_null_targets(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with null target '{TARGET_COL}'")
    return df


def _handle_remaining_nulls(df: pd.DataFrame) -> pd.DataFrame:
    is_time_series = TIMESTAMP_COL in df.columns
    if is_time_series:
        print(f"  Time-series detected ('{TIMESTAMP_COL}') — sorting and forward-filling remaining nulls")
        df = df.sort_values(TIMESTAMP_COL).ffill()
        # drop any rows still null after ffill (leading rows with no prior value)
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped:,} leading rows that could not be forward-filled")
    else:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped:,} rows with remaining nulls")
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} exact duplicate rows")
    else:
        print("  No duplicate rows found")
    return df


def _convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # use Int64 for integer-like, float64 for fractional
        if df[col].dropna().apply(float.is_integer).all():
            df[col] = df[col].astype("Int64")
        else:
            df[col] = df[col].astype("float64")

    for col in ["Login Successful", "Is Attack IP", TARGET_COL]:
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(bool)

    return df


def _save(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    size_mb = path.stat().st_size / 1_048_576
    print(f"  Saved cleaned data → {path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("\n--- Step 1: Drop high-null columns ---")
    df, _ = _drop_high_null_columns(df)

    print("\n--- Step 2: Drop null targets ---")
    df = _drop_null_targets(df)

    print("\n--- Step 3: Handle remaining nulls ---")
    df = _handle_remaining_nulls(df)

    print("\n--- Step 4: Remove duplicates ---")
    df = _drop_duplicates(df)

    print("\n--- Step 5: Convert dtypes ---")
    df = _convert_dtypes(df)
    print("  Dtype conversion complete")

    print("\n--- Step 6: Save cleaned CSV ---")
    _save(df)

    print("\n--- Step 7: Re-run quality gate ---")
    quality_result = check_data_quality(df)
    _print_report(quality_result)

    return df, quality_result


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    print("Loading raw data...")
    raw = load(path)
    rows_before = len(raw)
    print(f"Raw shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")

    df_clean, result = clean_data(raw)

    rows_after = len(df_clean)
    print(f"Before: {rows_before:,} rows  →  After: {rows_after:,} rows  "
          f"(removed {rows_before - rows_after:,} / {(rows_before - rows_after) / rows_before:.1%})")
    print(f"Quality gate: {'PASSED' if result['success'] else 'FAILED'}")
    sys.exit(0 if result["success"] else 1)
