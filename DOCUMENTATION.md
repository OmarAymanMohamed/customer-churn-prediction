# Customer Churn Prediction and Analysis Project

## Project Overview
This project implements a machine learning solution to predict customer churn for a telecommunications company. Using advanced analytics and machine learning techniques, we've developed a system that can identify customers at risk of churning, enabling proactive retention strategies.

## Project Structure
```
customer-churn-prediction/
├── app.py                    # Streamlit web application
├── src/                     # Source code
│   ├── config/             # Configuration files
│   │   └── config.py      # Project settings and paths
│   └── utils/             # Utility functions
│       └── logger.py      # Logging configuration
├── model_artifacts/        # Trained models and artifacts
│   ├── best_model.joblib  # GradientBoostingClassifier
│   ├── scaler.joblib      # StandardScaler
│   ├── power_transformer.joblib  # Power transformer
│   └── model_metrics.json # Model performance metrics
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Milestone 1: Data Collection, Exploration, and Preprocessing

### Data Collection
- Source: IBM Telco Customer Churn dataset
- Features: Customer demographics, services, contract details, and charges
- Target Variable: Churn (Yes/No)

### Data Exploration
- Dataset Size: ~7,000 customer records
- Features: Mix of categorical and numerical variables
- Initial Churn Rate: ~26-27%

### Data Preprocessing
- Standardized column names
- Handled missing values in TotalCharges
- Converted categorical variables to appropriate formats
- Feature scaling using StandardScaler
- Power transformation for numerical features

### Key Findings
- Higher churn rate for:
  - Month-to-month contracts
  - Electronic check payments
  - Shorter tenure customers
  - Higher monthly charges
  - Customers without online security/backup

## Milestone 2: Feature Engineering and Analysis

### Feature Engineering
1. **Service-Based Features**:
   - ServiceCount: Total number of services
   - OnlineServices: Combined online security/backup flag
   - StreamingServices: Combined TV/Movies streaming flag

2. **Financial Features**:
   - AvgMonthlyCharges: TotalCharges/Tenure
   - ChargesPerYear: Annualized charges
   - MonthlyChargesPerService: Cost efficiency metric

3. **Risk Scores**:
   - ContractRiskScore: Based on contract type
   - PaymentRiskScore: Based on payment method

### Statistical Analysis
- Chi-square tests for categorical relationships
- Correlation analysis for numerical features
- Feature importance ranking

## Milestone 3: Model Development and Optimization

### Model Architecture
- Type: GradientBoostingClassifier
- Features: 55 engineered features
- Target: Binary churn prediction

### Model Performance
- Accuracy: 78.8%
- ROC AUC: 84.2%
- Precision: 60.2%
- Recall: 59.1%
- F1 Score: 59.6%

### Feature Importance
1. Contract Type
2. Tenure
3. Monthly Charges
4. Payment Method
5. Total Charges
6. Service Count

## Milestone 4: Deployment and Risk Analysis

### Web Application
- Framework: Streamlit
- Deployment: Streamlit Cloud
- URL: [Customer Churn Predictor](https://omaraymanmohamed-customer-churn-prediction-app-xxxxx.streamlit.app)

### Risk Analysis System
1. **Base Risk Factors**:
   - Month-to-month contract: 2.5x risk multiplier
   - Electronic check payment: 2.0x risk multiplier
   - Low tenure (< 24 months): 1.8x risk multiplier

2. **Additional Risk Factors**:
   - Multiple premium services: 1.5x multiplier
   - Streaming services: 1.3x multiplier
   - High total charges: 1.3x multiplier
   - Price sensitivity: 1.4x multiplier

### Prediction Pipeline
1. Data Collection:
   - Customer demographics
   - Service information
   - Contract details
   - Payment information

2. Feature Engineering:
   - Create engineered features
   - Apply scaling and transformations
   - Calculate risk scores

3. Prediction Generation:
   - Base probability from model
   - Risk factor adjustment
   - Final churn probability

4. Results Presentation:
   - Churn probability
   - Identified risk factors
   - Detailed analysis

## Model Monitoring and Maintenance

### Logging System
- Prediction logging for monitoring
- Error tracking and reporting
- Performance metrics tracking

### Model Updates
- Regular performance evaluation
- Retraining triggers based on:
  - Performance degradation
  - Data drift detection
  - New feature availability

### Future Improvements
1. Model Enhancements:
   - Ensemble methods
   - Deep learning approaches
   - Time series analysis

2. Feature Engineering:
   - Customer behavior patterns
   - Interaction effects
   - Temporal features

3. Application Features:
   - Batch prediction support
   - API integration
   - Custom risk factor configuration

## Running the Project

### Local Setup
```bash
# Clone repository
git clone https://github.com/OmarAymanMohamed/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Dependencies
- Python 3.9+
- streamlit==1.32.0
- pandas==2.2.0
- numpy==1.26.4
- scikit-learn==1.4.0
- joblib==1.3.2
- plotly==5.18.0
- matplotlib==3.8.3
- seaborn==0.13.2

## License
This project is licensed under the MIT License - see the LICENSE file for details. 