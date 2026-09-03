import os
from pathlib import Path

import pandas as pd


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

def load_data():
    """
    Load the UCI Student Performance dataset.

    The original UCI dataset uses ';' as the separator.
    """
    print("Loading dataset...")

    df = pd.read_csv(RAW_DATA_PATH, sep=";")

    print(f"Dataset loaded successfully.")
    print(f"Shape: {df.shape}")

    return df


def clean_data(df):
    """
    Clean and prepare the dataset for AutoML.
    """

    print("\nStarting data preprocessing...")

    # Display missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Remove duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"\nRemoved {duplicates} duplicate rows.")
    else:
        print("\nNo duplicate rows found.")

    # Remove rows containing missing values
    missing_rows = df.isnull().sum().sum()

    if missing_rows > 0:
        df = df.dropna()
        print(f"Removed rows containing missing values.")
    else:
        print("No missing values found.")


    columns_to_remove = [
        "G1",
        "G2",
        "romantic",
        "Dalc",
        "Walc"
    ]

    existing_columns = [
        column for column in columns_to_remove
        if column in df.columns
    ]

    if existing_columns:
        df = df.drop(columns=existing_columns)
        print(f"\nRemoved columns: {existing_columns}")

    # Make sure the target column G3 is numeric
    df["G3"] = pd.to_numeric(df["G3"], errors="coerce")

    # Remove rows where G3 could not be converted
    df = df.dropna(subset=["G3"])

    print("\nPreprocessing completed.")
    print(f"Final dataset shape: {df.shape}")

    return df


def save_data(df):
    """
    Save the processed dataset.
    """

    # Create directory if it doesn't exist
    os.makedirs(
        os.path.dirname(PROCESSED_DATA_PATH),
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(f"\nProcessed dataset saved to:")
    print(PROCESSED_DATA_PATH)


def main():
    """
    Main preprocessing pipeline.
    """

    df = load_data()

    print("\nOriginal columns:")
    print(df.columns.tolist())

    df = clean_data(df)

    print("\nProcessed columns:")
    print(df.columns.tolist())

    save_data(df)


if __name__ == "__main__":
    main()