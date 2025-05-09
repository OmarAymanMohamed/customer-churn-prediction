"""
Configuration settings for the customer churn prediction project.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR]:
    directory.mkdir(exist_ok=True)

# Dataset configuration
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATASET_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_data.csv"
METADATA_PATH = PROCESSED_DATA_DIR / "metadata.json"

# Model configuration
MODEL_PATHS = {
    'best_model': MODEL_ARTIFACTS_DIR / "best_model.joblib",
    'scaler': MODEL_ARTIFACTS_DIR / "scaler.joblib",
    'metrics': MODEL_ARTIFACTS_DIR / "model_metrics.json"
}

# Data settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature settings
NUMERICAL_FEATURES = [
    'TenureMonths',
    'MonthlyCharges',
    'TotalCharges',
    'AvgMonthlyCharges',
    'TenureYears',
    'ChargesPerYear',
    'ServiceCount',
    'ContractRiskScore',
    'PaymentRiskScore',
    'MonthlyChargesPerService',
    'TotalChargesPerService'
]

CATEGORICAL_FEATURES = [
    'Gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

TARGET_FEATURE = 'Churn'

# Model training configuration
CV_FOLDS = 5

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
}

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "pipeline.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': "Customer Churn Predictor",
    'page_icon': "🔄",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
} 