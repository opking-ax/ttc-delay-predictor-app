import pandas as pd
from pathlib import Path


DELAY_THRESHOLD = 15


def load_raw(folder: Path) -> pd.DataFrame:
    """
    Loads the raw TTC XLSX from Open Data Toronto, renaming columns
    and return a single concatenated DataFrame
    """
    frames = []
    xlsx_files = sorted(folder.glob("*.xlsx"))

    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX files in {folder}")

    for fp in xlsx_files:
        df = pd.read_excel(fp)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined):,} rows from {len(xlsx_files)} files.")
    return combined


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are not needed for the project
    """
    df = df.copy()
    unrequired = ["min_gap", "vehicle", "location"]
    df = df.drop(unrequired, axis=1, errors="ignore")
    print(f"[preprocess] Dropped columns: {unrequired}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns where:
        - the target and/or key features are missing from the file
    """
    df = df.copy()
    required = ["min_delay", "date", "time", "route"]
    before = len(df)
    df = df.dropna(subset=required)
    print(f"[preprocess] Dropped {before - len(df)} rows with nulls in {required}")
    return df.reset_index(drop=True)


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the datetime features from the appropriate columns.
    Works for bus, streetcar & subway
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce",
        dayfirst=False,
    )
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Rush Hours (AM: 8-9; PM: 16-18)
    df["is_am_rush"] = df["hour"].between(6, 9).astype(int)
    df["is_pm_rush"] = df["hour"].between(15, 19).astype(int)

    unrequired = ["datetime"]
    df = df.drop(unrequired, axis=1, errors="ignore")

    return df


def add_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin `hour` into named periods that matter for transit analysis:
        Overnight Low   00-05 (approx)
        Morning Peak    06-09
        Midday          09-15
        Afternoon Peak  15-19
        Evening         19-23
    """
    bins = [-1, 5, 9, 15, 19, 23]
    labels = ["overnight", "am_peak", "midday", "pm_peak", "evening"]
    df["time_of_day"] = pd.cut(df["hour"], bins=bins, labels=labels)
    return df


def make_target(df: pd.DataFrame, threshold: int = DELAY_THRESHOLD) -> pd.DataFrame:
    """
    Binary classification label:
        1 -> delayed by more than `threshold` minutes
        0 -> on time (or minor delay)
    """
    df = df.copy()
    df["is_delayed"] = (df["min_delay"] > threshold).astype(int)
    print(
        f"[preprocess] Delay Rate: "
        f"{df['is_delayed'].mean():.1%} of trips delayed > {threshold} mins"
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encoding high-cardinality categoricals
    """
    df = df.copy()
    route_list = (
        df.groupby("route")["is_delayed"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "count", "mean": "route_encoded"})
    )

    df = df.merge(route_list[["route", "route_encoded"]], on="route", how="left")

    valid_dirs = {"N", "S", "E", "W", "B"}
    df["direction"] = df["direction"].str.strip().str.upper()
    df["direction"] = df["direction"].where(df["direction"].isin(valid_dirs), "U")

    df["incident"] = (
        df["incident"]
        .str.lower()
        .str.strip()
        .str.replace(r"\s*-\s*", "_", regex=True)
        .str.replace(" ", "_")
    )

    return df


def preprocess(raw_folder: Path, output_path: Path) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Returns the processed DataFrame and saves it as a CSV file.
    """
    df = load_raw(raw_folder)
    df = clean(df)
    df = drop_columns(df)
    df = parse_datetime(df)
    df = add_time_of_day(df)
    df = make_target(df)
    df = encode_categoricals(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[preprocess] Saved processed data → {output_path}")
    return df
