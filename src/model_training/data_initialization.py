# src/dataingestion/ingest_data.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(raw_file: str):
    """
    Load raw data, split into train/test, and save into data folder.
    Parameters
    ----------
    raw_file : str
        Filename of the raw CSV (inside data/raw/).
    test_size : float
        Fraction of data to be used for test split.
    """

    # Step back two directories to reach project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Paths
    raw_path = os.path.join(base_dir, "data", raw_file)
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")

    # Load data
    raw_df = pd.read_csv(raw_path)
    print(f"Loaded raw data: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    #drop columns
    df = raw_df.drop(['id','Unnamed: 32'], axis=1)
    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # Save files
    train_file = os.path.join(data_dir, "train.csv")
    test_file = os.path.join(data_dir, "test.csv")

    return train_df,test_df
