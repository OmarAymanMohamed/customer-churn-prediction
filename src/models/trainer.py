"""
Model training module for the customer churn prediction project.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import joblib
import json
from pathlib import Path
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.config.config import (
    PROCESSED_DATA_DIR,
    MODEL_ARTIFACTS_DIR,
    RANDOM_STATE
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.feature_names = None
        
    def load_data(self):
        """Load processed data."""
        logger.info("Loading processed data")
        
        self.X_train = joblib.load(PROCESSED_DATA_DIR / "X_train.joblib")
        self.X_test = joblib.load(PROCESSED_DATA_DIR / "X_test.joblib")
        self.y_train = joblib.load(PROCESSED_DATA_DIR / "y_train.joblib")
        self.y_test = joblib.load(PROCESSED_DATA_DIR / "y_test.joblib")
        
        self.feature_names = list(self.X_train.columns)
        logger.info("Processed data loaded")
        
    def train_model(self):
        """Train and evaluate multiple models."""
        logger.info("Training models")
        
        # Define base models with hyperparameter grids
        models = {
            'rf': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced']
                }
            },
            'gb': {
                'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgb': {
                'model': XGBClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'scale_pos_weight': [1]
                }
            }
        }
        
        # Train and tune base models
        base_models = []
        for name, config in models.items():
            logger.info(f"Training {name}")
            grid = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid.fit(self.X_train, self.y_train)
            base_models.append((name, grid.best_estimator_))
            logger.info(f"{name} best score: {grid.best_score_:.4f}")
        
        # Create stacking ensemble
        estimators = base_models
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=RANDOM_STATE),
            cv=5
        )
        
        # Train stacking ensemble
        logger.info("Training stacking ensemble")
        stacking.fit(self.X_train, self.y_train)
        
        # Evaluate all models
        models_to_evaluate = {
            **{name: model for name, model in base_models},
            'stacking': stacking
        }
        
        best_score = 0
        best_model_name = None
        
        for name, model in models_to_evaluate.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred)
            }
            
            logger.info(f"{name} metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model_name = name
                self.best_model = model
                self.best_metrics = metrics
        
        logger.info(f"Best model: {best_model_name}")
        
    def save_artifacts(self):
        """Save model artifacts."""
        logger.info("Saving model artifacts")
        
        # Create directory if it doesn't exist
        MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        model_path = MODEL_ARTIFACTS_DIR / "best_model.joblib"
        joblib.dump(self.best_model, model_path)
        logger.info(f"Saved best model to {model_path}")
        
        # Save model metrics
        metrics_path = MODEL_ARTIFACTS_DIR / "model_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'best_model': self.best_model.__class__.__name__,
                **{k: float(v) for k, v in self.best_metrics.items()}
            }, f, indent=4)
        logger.info(f"Saved model metrics to {metrics_path}")
        
        # Save feature names
        feature_names_path = MODEL_ARTIFACTS_DIR / "feature_names.txt"
        with open(feature_names_path, 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        logger.info(f"Saved feature names to {feature_names_path}")
        
    def main(self):
        """Main function to run the model trainer."""
        self.load_data()
        self.train_model()
        self.save_artifacts() 