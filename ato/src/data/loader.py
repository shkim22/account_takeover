import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"


def find_csv(data_dir: Path = DATA_DIR) -> Path:
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    if len(csvs) > 1:
        print(f"Multiple CSVs found; loading first: {csvs[0].name}")
    return csvs[0]


def load(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else find_csv()
    return pd.read_csv(path)


def report(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"SHAPE:  {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nCOLUMNS & DTYPES:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<35} {dtype}")

    print("\nSUMMARY STATISTICS (numeric):")
    numeric = df.select_dtypes("number")
    if numeric.empty:
        print("  No numeric columns.")
    else:
        stats = numeric.agg(["mean", "std", "min", "max"])
        print(stats.to_string())

    print("\nMISSING VALUES:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No missing values.")
    else:
        pct = (missing / len(df) * 100).round(2)
        summary = pd.DataFrame({"count": missing, "pct": pct})
        print(summary.to_string())

    print("=" * 60)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load(path)
    report(df)
