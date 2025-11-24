"""
CSV data loading and filtering functions.

This module provides functions to load CSV files from the data directory
and filter them by TimeBarStart range.
"""

import pandas as pd
from pathlib import Path


def load_all_csv_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load all CSV files from the data directory and combine them into a single DataFrame.

    Recursively searches through all subdirectories in the data directory
    and reads all CSV files, combining them into one DataFrame.

    Args:
        data_dir: Path to the data directory. Defaults to "data".

    Returns:
        A pandas DataFrame containing all data from all CSV files.

    Example:
        >>> df = load_all_csv_data()
        >>> print(df.shape)
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory '{data_dir}' does not exist")

    # Find all CSV files recursively
    csv_files = list(data_path.rglob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in '{data_dir}' directory")

    # Read and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


def filter_by_time_range(
    df: pd.DataFrame, start_time: str, end_time: str, time_column: str = "TimeBarStart"
) -> pd.DataFrame:
    """
    Filter DataFrame rows by TimeBarStart range.

    Filters the DataFrame to include only rows where the TimeBarStart
    value is within the specified range (inclusive). TimeBarStart format
    is "HH:MM" (e.g., "09:30", "15:59").

    Args:
        df: The pandas DataFrame to filter.
        start_time: Start time in "HH:MM" format (inclusive).
        end_time: End time in "HH:MM" format (inclusive).
        time_column: Name of the time column to filter on. Defaults to "TimeBarStart".

    Returns:
        A filtered pandas DataFrame containing only rows within the time range.

    Example:
        >>> df = load_all_csv_data()
        >>> filtered_df = filter_by_time_range(df, "09:30", "10:00")
        >>> print(filtered_df.shape)
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame")

    # Filter rows where TimeBarStart is within the range
    # Since format is "HH:MM", string comparison works correctly
    mask = (df[time_column] >= start_time) & (df[time_column] <= end_time)
    filtered_df = df[mask].copy()

    return filtered_df
