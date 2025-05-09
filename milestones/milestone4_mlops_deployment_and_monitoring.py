"""
Customer Churn Prediction and Analysis Project

Milestone 4: MLOps, Deployment, and Monitoring
Team Member: Doha Sayed

This script contains the implementation of Milestone 4 of our customer churn prediction project,
focusing on MLOps, deployment, and monitoring.
"""

import subprocess
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

import plotly.graph_objects as go

# Install required packages
required_packages = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly',
    'scikit-learn', 'xgboost', 'scipy', 'joblib', 'flask', 'mlflow'
]

for package in required_packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Data manipulation and analysis
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier

# Model loading and saving
import joblib
import json
import datetime

# MLOps tools
import mlflow

# Define paths and configurations
BASE_DIR = project_root
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODEL_ARTIFACTS_DIR]:
    directory.mkdir(exist_ok=True)

# Model paths
MODEL_PATHS = {
    'best_model': MODEL_ARTIFACTS_DIR / "best_model.joblib",
    'scaler': MODEL_ARTIFACTS_DIR / "scaler.joblib",
    'power_transformer': MODEL_ARTIFACTS_DIR / "power_transformer.joblib"
}

# Feature configurations
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

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

"""## Load Model for Deployment

Let's try to load the model we developed in Milestone 3. If it's not available, we'll create a simple model.
"""

# Create or ensure model_artifacts directory exists
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Try to load the model from Milestone 3
try:
    # Try to load the model
    model = joblib.load(MODEL_PATHS['best_model'])
    scaler = joblib.load(MODEL_PATHS['scaler'])

    # Load feature names
    with open(MODEL_PATHS['best_model'].parent / "feature_names.txt", 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    print("Successfully loaded model artifacts from Milestone 3.")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features: {len(feature_names)}")

except (FileNotFoundError, EOFError) as e:
    print(f"Could not load model artifacts: {str(e)}")
    print("Creating a simple model for demonstration purposes...")

    # Load feature-engineered data if available
    try:
        os.makedirs('milestone_outputs', exist_ok=True)
        df_model = pd.read_csv('milestone_outputs/feature_engineered_data.csv')
        print("Using feature-engineered data from previous milestone.")
    except FileNotFoundError:
        print("No feature-engineered data found. Creating synthetic data...")
        # Create a simple synthetic dataset
        np.random.seed(42)
        n_samples = 1000

        # Create synthetic data with key features for churn prediction
        df_model = pd.DataFrame({
            'customerID': [f'CUST-{i:04d}' for i in range(1, n_samples + 1)],
            'Tenure Months': np.random.randint(1, 73, size=n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, size=n_samples).round(2),
            'TotalCharges': np.random.uniform(100, 8000, size=n_samples).round(2),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples),
            'Churn Label': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.27, 0.73])
        })

    # Prepare data for modeling
    # Make sure we have a numeric target variable
    if 'Churn Value' in df_model.columns:
        target_col = 'Churn Value'
    elif 'Churn Label' in df_model.columns:
        df_model['Churn Value'] = df_model['Churn Label'].map({'Yes': 1, 'No': 0})
        target_col = 'Churn Value'
    elif 'Churn' in df_model.columns:
        if df_model['Churn'].dtype == object:  # String values
            df_model['Churn Value'] = df_model['Churn'].map({'Yes': 1, 'No': 0})
            target_col = 'Churn Value'
        else:  # Already numeric
            target_col = 'Churn'
    else:
        raise ValueError("No churn column found in the dataset.")

    # Drop ID columns and columns not used for modeling
    drop_cols = ['customerID', 'Churn Label']
    drop_cols = [col for col in drop_cols if col in df_model.columns and col != target_col]
    if 'Churn' in df_model.columns and target_col != 'Churn':
        drop_cols.append('Churn')

    X = df_model.drop(drop_cols + [target_col], axis=1)
    y = df_model[target_col]

    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the feature names
    feature_names = X.columns.tolist()
    with open(MODEL_PATHS['best_model'].parent / "feature_names.txt", 'w') as f:
        f.write('\n'.join(feature_names))

    # Save the model and scaler
    joblib.dump(model, MODEL_PATHS['best_model'])
    joblib.dump(scaler, MODEL_PATHS['scaler'])

    print("Simple model created and saved.")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features: {len(feature_names)}")

"""## 4.1 Create a Simple Prediction Function

Let's create a function to make predictions on new data.
"""

def create_engineered_features(df):
    """Create engineered features for prediction."""
    # Calculate average monthly charges
    df['AvgMonthlyCharges'] = df['TotalCharges'] / df['TenureMonths'].replace(0, 1)
    
    # Create tenure-based features
    df['TenureYears'] = df['TenureMonths'] / 12
    df['ChargesPerYear'] = df['TotalCharges'] / df['TenureYears'].replace(0, 1)
    
    # Create service usage score
    service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['ServiceCount'] = df[service_columns].apply(
        lambda x: sum(1 for item in x if item in ['Yes', 'DSL', 'Fiber optic']), axis=1
    )
    
    # Create contract risk score
    df['ContractRiskScore'] = df['Contract'].map({
        'Month-to-month': 3,
        'One year': 2,
        'Two year': 1
    })
    
    # Create payment risk score
    df['PaymentRiskScore'] = df['PaymentMethod'].map({
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
    })
    
    # Add interaction features
    df['MonthlyChargesPerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
    df['TotalChargesPerService'] = df['TotalCharges'] / (df['ServiceCount'] + 1)
    
    return df

def predict_churn(customer_data):
    """
    Predict customer churn using the trained model.
    
    Args:
        customer_data (dict): Customer information
        
    Returns:
        dict: Prediction results
    """
    try:
        # Load model and artifacts
        model = joblib.load(MODEL_PATHS['best_model'])
        scaler = joblib.load(MODEL_PATHS['scaler'])
        power_transformer = joblib.load(MODEL_PATHS['power_transformer'])
        
        # Create DataFrame from customer data
        df = pd.DataFrame([customer_data])
        
        # Rename column if necessary
        if 'Tenure Months' in df.columns:
            df['TenureMonths'] = df['Tenure Months']
            df = df.drop('Tenure Months', axis=1)
            
        # Create engineered features
        df = create_engineered_features(df)
        
        # Prepare numerical features
        numerical_features = NUMERICAL_FEATURES
        
        # Scale numerical features
        df[numerical_features] = power_transformer.transform(df[numerical_features])
        df[numerical_features] = scaler.transform(df[numerical_features])
        
        # Create DataFrame for encoded features
        with open(MODEL_PATHS['best_model'].parent / "feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        encoded_features = pd.DataFrame(0, index=df.index, columns=feature_names)
        
        # Set numerical features
        for feature in numerical_features:
            if feature in df.columns:
                encoded_features[feature] = df[feature]
        
        # Set categorical features
        categorical_mappings = {
            'Gender': ['Gender_Female', 'Gender_Male'],
            'SeniorCitizen': ['SeniorCitizen_No', 'SeniorCitizen_Yes'],
            'Partner': ['Partner_No', 'Partner_Yes'],
            'Dependents': ['Dependents_No', 'Dependents_Yes'],
            'PhoneService': ['PhoneService_No', 'PhoneService_Yes'],
            'MultipleLines': ['MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes'],
            'InternetService': ['InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No'],
            'OnlineSecurity': ['OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes'],
            'OnlineBackup': ['OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes'],
            'DeviceProtection': ['DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes'],
            'TechSupport': ['TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes'],
            'StreamingTV': ['StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes'],
            'StreamingMovies': ['StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes'],
            'Contract': ['Contract_Month-to-month', 'Contract_One year', 'Contract_Two year'],
            'PaperlessBilling': ['PaperlessBilling_No', 'PaperlessBilling_Yes'],
            'PaymentMethod': [
                'PaymentMethod_Bank transfer (automatic)',
                'PaymentMethod_Credit card (automatic)',
                'PaymentMethod_Electronic check',
                'PaymentMethod_Mailed check'
            ]
        }
        
        for feature, encoded_cols in categorical_mappings.items():
            if feature in df.columns:
                value = df[feature].iloc[0]
                for col in encoded_cols:
                    if col.endswith(str(value)):
                        encoded_features[col] = 1
                        break
        
        # Make prediction
        prediction = model.predict(encoded_features)[0]
        probability = model.predict_proba(encoded_features)[0][1]
        
        # Log prediction
        prediction_result = {
            'will_churn': bool(prediction),
            'churn_probability': float(probability),
            'timestamp': datetime.datetime.now().isoformat(),
            'model_version': model.__class__.__name__
        }
        
        return prediction_result
        
    except Exception as e:
        return {'error': f'Error making prediction: {str(e)}'}

def log_prediction(prediction_result):
    """
    Log prediction results.
    
    Args:
        prediction_result (dict): Prediction results to log
    """
    try:
        log_file = 'logs/predictions.json'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Load existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new prediction
        logs.append(prediction_result)
        
        # Save updated logs
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=4)
            
    except Exception as e:
        print(f"Error logging prediction: {str(e)}")

"""## 4.2 Test the Prediction Function

Let's test our prediction function with a sample customer.
"""

# Create a sample customer
sample_customer = {
    'Gender': 'Male',
    'SeniorCitizen': 'No',
    'Partner': 'Yes',
    'Dependents': 'No',
    'Tenure Months': 36,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'Yes',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.7,
    'TotalCharges': 3455.2
}

# Predict churn
prediction = predict_churn(sample_customer)

# Display prediction result
print("\nPrediction Result:")
for key, value in prediction.items():
    if key != 'timestamp':
        print(f"{key}: {value}")

def create_streamlit_app():
    """Create and run the Streamlit application."""
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("Customer Churn Predictor")
    st.markdown("""
        This application predicts the likelihood of a customer churning based on their service details and demographics.
        Fill out the form below to get a prediction.
    """)

    # Create input form
    with st.form("customer_info"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.checkbox("Senior Citizen")
            partner = st.checkbox("Has Partner")
            dependents = st.checkbox("Has Dependents")

            st.subheader("Service Information")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.checkbox("Paperless Billing")
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])

        with col2:
            st.subheader("Service Features")
            phone_service = st.checkbox("Phone Service")
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

            st.subheader("Charges")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

        # Submit button
        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Create customer data dictionary
        customer_data = {
            'Gender': gender,
            'SeniorCitizen': 'Yes' if senior_citizen else 'No',
            'Partner': 'Yes' if partner else 'No',
            'Dependents': 'Yes' if dependents else 'No',
            'TenureMonths': tenure,
            'PhoneService': 'Yes' if phone_service else 'No',
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': 'Yes' if paperless_billing else 'No',
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        # Make prediction
        result = predict_churn(customer_data)

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create a color-coded result box
        if result.get('will_churn'):
            st.error("⚠️ Customer likely to churn")
        else:
            st.success("✅ Customer likely to stay")

        # Display probability
        st.metric(
            "Churn Probability",
            f"{result.get('churn_probability', 0):.1%}",
            delta=None
        )

        # Display prediction details
        with st.expander("View Detailed Results"):
            st.json(result)

if __name__ == "__main__":
    create_streamlit_app()