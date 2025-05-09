"""
Customer Churn Prediction - Performance Metrics Report
This script analyzes and stores performance metrics for all models in the project.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime
import os
from pathlib import Path

def load_and_prepare_data():
    """Load and prepare data for model evaluation."""
    try:
        # Try to load feature-engineered data
        df = pd.read_csv('milestone_outputs/feature_engineered_data.csv')
        print("Successfully loaded feature-engineered data.")
    except FileNotFoundError:
        print("Feature-engineered data not found.")
        return None, None, None

    # Prepare target variable
    if 'Churn Value' in df.columns:
        target_col = 'Churn Value'
    elif 'Churn Label' in df.columns:
        df['Churn Value'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
        target_col = 'Churn Value'
    elif 'Churn' in df.columns:
        if df['Churn'].dtype == object:
            df['Churn Value'] = df['Churn'].map({'Yes': 1, 'No': 0})
            target_col = 'Churn Value'
        else:
            target_col = 'Churn'

    # Drop unnecessary columns
    drop_cols = ['customerID', 'Churn Label'] if 'customerID' in df.columns else []
    drop_cols = [col for col in drop_cols if col in df.columns and col != target_col]
    
    # Prepare features
    X = df.drop(drop_cols + [target_col], axis=1)
    y = df[target_col]

    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y, df.shape[0]

def evaluate_saved_model():
    """Evaluate the saved model's performance."""
    try:
        # Load model artifacts
        model = joblib.load('model_artifacts/churn_model.joblib')
        scaler = joblib.load('model_artifacts/scaler.joblib')
        
        # Load feature names
        with open('model_artifacts/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
            
        print("Successfully loaded model artifacts.")
        
        # Load and prepare data
        X, y, total_samples = load_and_prepare_data()
        if X is None:
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Model Type': type(model).__name__,
            'Total Samples': total_samples,
            'Training Samples': len(X_train),
            'Testing Samples': len(X_test),
            'Number of Features': len(feature_names),
            'Feature Names': feature_names,
            'Performance Metrics': {
                'Accuracy': float(accuracy_score(y_test, y_pred)),
                'Precision': float(precision_score(y_test, y_pred)),
                'Recall': float(recall_score(y_test, y_pred)),
                'F1 Score': float(f1_score(y_test, y_pred)),
                'ROC-AUC': float(roc_auc_score(y_test, y_pred_proba))
            },
            'Class Distribution': {
                'Train': {
                    'Not Churned': int((y_train == 0).sum()),
                    'Churned': int((y_train == 1).sum())
                },
                'Test': {
                    'Not Churned': int((y_test == 0).sum()),
                    'Churned': int((y_test == 1).sum())
                }
            }
        }
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            metrics['Feature Importance'] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(feature_names, model.coef_[0]))
            metrics['Feature Coefficients'] = dict(
                sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            )
            
        return metrics
        
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return None

def save_metrics_report(metrics):
    """Save metrics to a JSON file with timestamp."""
    if metrics is None:
        return
        
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Add timestamp
    metrics['Timestamp'] = datetime.now().isoformat()
    
    # Save as JSON
    report_path = Path('reports/performance_metrics_report.json')
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nPerformance metrics report saved to {report_path}")
    
    # Also save as markdown for better readability
    markdown_path = Path('reports/performance_metrics_report.md')
    with open(markdown_path, 'w') as f:
        f.write("# Customer Churn Prediction - Performance Metrics Report\n\n")
        f.write(f"Generated on: {metrics['Timestamp']}\n\n")
        
        f.write("## Model Information\n")
        f.write(f"- Model Type: {metrics['Model Type']}\n")
        f.write(f"- Total Samples: {metrics['Total Samples']}\n")
        f.write(f"- Training Samples: {metrics['Training Samples']}\n")
        f.write(f"- Testing Samples: {metrics['Testing Samples']}\n")
        f.write(f"- Number of Features: {metrics['Number of Features']}\n\n")
        
        f.write("## Performance Metrics\n")
        for metric, value in metrics['Performance Metrics'].items():
            f.write(f"- {metric}: {value:.4f}\n")
        f.write("\n")
        
        f.write("## Class Distribution\n")
        f.write("### Training Set\n")
        for label, count in metrics['Class Distribution']['Train'].items():
            f.write(f"- {label}: {count}\n")
        f.write("\n### Testing Set\n")
        for label, count in metrics['Class Distribution']['Test'].items():
            f.write(f"- {label}: {count}\n")
        f.write("\n")
        
        if 'Feature Importance' in metrics:
            f.write("## Feature Importance (Top 10)\n")
            for feature, importance in list(metrics['Feature Importance'].items())[:10]:
                f.write(f"- {feature}: {importance:.4f}\n")
        elif 'Feature Coefficients' in metrics:
            f.write("## Feature Coefficients (Top 10 by absolute value)\n")
            for feature, coef in list(metrics['Feature Coefficients'].items())[:10]:
                f.write(f"- {feature}: {coef:.4f}\n")
    
    print(f"Performance metrics report also saved as markdown to {markdown_path}")

def main():
    """Main function to generate performance metrics report."""
    print("Generating performance metrics report...")
    metrics = evaluate_saved_model()
    save_metrics_report(metrics)

if __name__ == "__main__":
    main() 