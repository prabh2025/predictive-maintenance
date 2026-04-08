import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = [
            'temperature', 'vibration',
            'pressure', 'rpm',
            'voltage', 'current'
        ]

    def create_features(self, df):
        logger.info("Creating features...")
        df = df.copy()

        # Rolling statistics (last 5 readings)
        for col in self.feature_cols:
            df[f'{col}_mean_5'] = (
                df.groupby('machine_id')[col]
                .transform(lambda x: x.rolling(5, min_periods=1).mean())
            )
            df[f'{col}_std_5'] = (
                df.groupby('machine_id')[col]
                .transform(lambda x: x.rolling(5, min_periods=1).std())
                .fillna(0)
            )
            df[f'{col}_max_5'] = (
                df.groupby('machine_id')[col]
                .transform(lambda x: x.rolling(5, min_periods=1).max())
            )

        # Temperature-Vibration interaction
        df['temp_vib_ratio'] = (
            df['temperature'] / (df['vibration'] + 0.001)
        )

        # Power consumption
        df['power'] = df['voltage'] * df['current']

        logger.info(f"Features created: {df.shape[1]} columns")
        return df

    def scale_features(self, df, fit=True):
        feature_cols = [
            c for c in df.columns
            if c not in ['timestamp', 'machine_id', 'failure']
        ]

        if fit:
            df[feature_cols] = self.scaler.fit_transform(
                df[feature_cols]
            )
            joblib.dump(self.scaler, 'models/scaler.pkl')
            logger.info("Scaler fitted and saved")
        else:
            df[feature_cols] = self.scaler.transform(
                df[feature_cols]
            )

        return df, feature_cols

if __name__ == "__main__":
    from data_ingestion import DataIngestion
    ingestion = DataIngestion()
    df = ingestion.load_data()
    df = ingestion.validate_data(df)

    fe = FeatureEngineering()
    df = fe.create_features(df)
    df, features = fe.scale_features(df, fit=True)
    df.to_csv('data/processed/features.csv', index=False)
    print(f"Feature engineering complete! Shape: {df.shape}")