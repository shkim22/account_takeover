import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd

from src.data.loader import load

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
REQUIRED_SCHEMA: dict[str, str] = {
    "Login Timestamp": "object",
    "User ID": "int64",
    "IP Address": "object",
    "Country": "object",
    "Login Successful": "bool",
    "Is Attack IP": "bool",
    "Is Account Takeover": "bool",
}

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_ROWS = 100
WARN_ROWS = 1_000

CRITICAL_NULL_RATE = 0.50
WARN_NULL_RATE = 0.20

# (inclusive min, inclusive max); None means unbounded
NUMERIC_BOUNDS: dict[str, tuple] = {
    "Round-Trip Time [ms]": (0, 1_000_000),
    "ASN": (1, 4_294_967_295),
}

TARGET_COL = "Is Account Takeover"
MIN_CLASS_FREQ = 0.05  # warn if any class < 5%


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_schema(df: pd.DataFrame) -> tuple[list, list]:
    failures, warnings = [], []
    missing_cols = [c for c in REQUIRED_SCHEMA if c not in df.columns]
    if missing_cols:
        failures.append(f"Missing required columns: {missing_cols}")
    for col, expected in REQUIRED_SCHEMA.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if actual != expected:
            warnings.append(
                f"Column '{col}' dtype is '{actual}', expected '{expected}'"
            )
    return failures, warnings


def _check_row_count(df: pd.DataFrame) -> tuple[list, list]:
    failures, warnings = [], []
    n = len(df)
    if n < MIN_ROWS:
        failures.append(f"Only {n:,} rows — minimum required is {MIN_ROWS:,}")
    elif n < WARN_ROWS:
        warnings.append(f"Only {n:,} rows — recommend at least {WARN_ROWS:,} for robust modelling")
    return failures, warnings


def _check_null_rates(df: pd.DataFrame) -> tuple[list, list]:
    failures, warnings = [], []
    rates = df.isnull().mean()
    for col, rate in rates.items():
        if rate > CRITICAL_NULL_RATE:
            failures.append(
                f"Column '{col}' is {rate:.1%} null — exceeds critical threshold of {CRITICAL_NULL_RATE:.0%}"
            )
        elif rate > WARN_NULL_RATE:
            warnings.append(f"Column '{col}' is {rate:.1%} null")
    return failures, warnings


def _check_value_ranges(df: pd.DataFrame) -> tuple[list, list]:
    failures, warnings = [], []
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if lo is not None and (series < lo).any():
            bad = int((series < lo).sum())
            failures.append(
                f"Column '{col}' has {bad:,} value(s) below minimum {lo}"
            )
        if hi is not None and (series > hi).any():
            bad = int((series > hi).sum())
            warnings.append(
                f"Column '{col}' has {bad:,} value(s) above maximum {hi}"
            )
    return failures, warnings


def _check_target_distribution(df: pd.DataFrame) -> tuple[list, list]:
    failures, warnings = [], []
    if TARGET_COL not in df.columns:
        warnings.append(
            f"Target column '{TARGET_COL}' not found — skipping distribution check"
        )
        return failures, warnings
    freqs = df[TARGET_COL].value_counts(normalize=True)
    if len(freqs) < 2:
        failures.append(
            f"Target '{TARGET_COL}' has only {len(freqs)} class — need at least 2"
        )
        return failures, warnings
    for cls, freq in freqs.items():
        if freq < MIN_CLASS_FREQ:
            warnings.append(
                f"Target class '{cls}' is only {freq:.3%} of data — dataset is highly imbalanced"
            )
    return failures, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_data_quality(df: pd.DataFrame) -> dict:
    all_failures: list[str] = []
    all_warnings: list[str] = []

    checks = [
        _check_schema,
        _check_row_count,
        _check_null_rates,
        _check_value_ranges,
        _check_target_distribution,
    ]
    for check in checks:
        f, w = check(df)
        all_failures.extend(f)
        all_warnings.extend(w)

    null_counts = df.isnull().sum()
    statistics = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls_by_column": null_counts[null_counts > 0].to_dict(),
        "null_rate_by_column": df.isnull().mean()[df.isnull().mean() > 0].round(4).to_dict(),
        "checks_run": len(checks),
        "failures_count": len(all_failures),
        "warnings_count": len(all_warnings),
    }

    return {
        "success": len(all_failures) == 0,
        "failures": all_failures,
        "warnings": all_warnings,
        "statistics": statistics,
    }


def _print_report(result: dict) -> None:
    status = "PASSED" if result["success"] else "FAILED"
    print(f"\n{'=' * 60}")
    print(f"DATA QUALITY GATE: {status}")
    print(f"{'=' * 60}")

    if result["failures"]:
        print(f"\nCRITICAL FAILURES ({len(result['failures'])}):")
        for msg in result["failures"]:
            print(f"  [FAIL] {msg}")

    if result["warnings"]:
        print(f"\nWARNINGS ({len(result['warnings'])}):")
        for msg in result["warnings"]:
            print(f"  [WARN] {msg}")

    stats = result["statistics"]
    print(f"\nSTATISTICS:")
    print(f"  Rows: {stats['total_rows']:,}  |  Columns: {stats['total_columns']}")
    print(
        f"  Checks run: {stats['checks_run']}  |  "
        f"Failures: {stats['failures_count']}  |  Warnings: {stats['warnings_count']}"
    )

    if stats["null_rate_by_column"]:
        print(f"\n  Null rates (affected columns only):")
        for col, rate in stats["null_rate_by_column"].items():
            print(f"    {col:<35} {rate:.1%}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load(path)
    result = check_data_quality(df)
    _print_report(result)
    sys.exit(0 if result["success"] else 1)
