import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.raw_path = "data/raw/sensor_data.csv"
        self.processed_path = "data/processed/features.csv"

    def load_data(self):
        logger.info("Loading sensor data...")
        df = pd.read_csv(self.raw_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded: {df.shape} | "
                   f"Failures: {df['failure'].sum()}")
        return df

    def validate_data(self, df):
        logger.info("Validating data...")
        # Check missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values: {missing}")
            df = df.fillna(df.mean(numeric_only=True))

        # Check data types
        assert 'timestamp' in df.columns
        assert 'failure' in df.columns
        assert df['failure'].isin([0,1]).all()

        logger.info("Validation passed!")
        return df

    def split_data(self, df):
        # Time-based split (important for time series!)
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        logger.info(f"Train: {train.shape} | "
                   f"Test: {test.shape}")
        return train, test

if __name__ == "__main__":
    ingestion = DataIngestion()
    df = ingestion.load_data()
    df = ingestion.validate_data(df)
    train, test = ingestion.split_data(df)
    print("Data ingestion complete!")