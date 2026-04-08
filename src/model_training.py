import mlflow
import mlflow.sklearn
import mlflow.xgboost
import dagshub
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    roc_auc_score, classification_report
)
import joblib
import logging

logger = logging.getLogger(__name__)

# Connect to DagsHub
dagshub.init(
    repo_owner='prabh2025',
    repo_name='predictive-maintenance',
    mlflow=True
)
mlflow.set_experiment("predictive_maintenance")

class ModelTrainer:
    def __init__(self):
        self.models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
        self.best_model = None
        self.best_score = 0
        self.best_name = ""

    def train_all(self, X_train, X_test,
                  y_train, y_test):
        logger.info("Starting model training...")

        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                # Train
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                proba = model.predict_proba(X_test)[:,1]

                # Metrics
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds,
                              average='weighted')
                precision = precision_score(
                    y_test, preds, average='weighted'
                )
                recall = recall_score(
                    y_test, preds, average='weighted'
                )
                auc = roc_auc_score(y_test, proba)

                # Log to MLflow
                mlflow.log_params(
                    {"model": name, **model.get_params()}
                )
                mlflow.log_metrics({
                    "accuracy": acc,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "roc_auc": auc
                })

                logger.info(
                    f"{name} → "
                    f"Acc:{acc:.4f} "
                    f"F1:{f1:.4f} "
                    f"AUC:{auc:.4f}"
                )

                # Track best
                if f1 > self.best_score:
                    self.best_score = f1
                    self.best_model = model
                    self.best_name = name

                    # Register best model
                    mlflow.sklearn.log_model(
                        model,
                        "best_model",
                        registered_model_name=
                        "predictive_maintenance_model"
                    )

        # Save best model
        joblib.dump(
            self.best_model,
            'models/best_model.pkl'
        )
        logger.info(
            f"Best: {self.best_name} "
            f"F1={self.best_score:.4f}"
        )
        return self.best_model

if __name__ == "__main__":
    import sys
    sys.path.append('src')
    from data_ingestion import DataIngestion
    from feature_engineering import FeatureEngineering

    # Load and process data
    ingestion = DataIngestion()
    df = ingestion.load_data()
    df = ingestion.validate_data(df)
    train_df, test_df = ingestion.split_data(df)

    fe = FeatureEngineering()
    train_df = fe.create_features(train_df)
    train_df, feature_cols = fe.scale_features(
        train_df, fit=True
    )
    test_df = fe.create_features(test_df)
    test_df, _ = fe.scale_features(
        test_df, fit=False
    )

    feature_cols = [
        c for c in train_df.columns
        if c not in ['timestamp', 'machine_id', 'failure']
    ]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df['failure']
    y_test = test_df['failure']

    # Train
    trainer = ModelTrainer()
    best_model = trainer.train_all(
        X_train, X_test, y_train, y_test
    )
    print("Training complete!")