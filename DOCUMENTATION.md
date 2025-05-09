# Customer Churn Prediction and Analysis Project

## Executive Summary
This project implements a comprehensive machine learning solution for predicting customer churn in the telecommunications industry. By leveraging advanced analytics, statistical methods, and machine learning techniques, we've developed a robust system capable of identifying at-risk customers with high accuracy. The solution includes a web-based interface for real-time predictions and risk analysis, enabling proactive customer retention strategies.

## Project Objectives
1. Develop an accurate churn prediction model
2. Identify key factors influencing customer churn
3. Create a risk assessment system
4. Build an interactive web application
5. Implement monitoring and maintenance protocols

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
- **Source**: IBM Telco Customer Churn dataset
- **Dataset Characteristics**:
  - Size: 7,043 customer records
  - Features: 21 variables (19 features + 1 target + 1 ID)
  - Time Period: Historical customer data
  - Data Types: Mix of categorical and numerical variables

### Data Exploration
1. **Demographic Analysis**:
   - Gender distribution: 50.5% Male, 49.5% Female
   - Senior Citizens: 16.2% of customer base
   - Partner status: 48.2% with partners
   - Dependents: 30% with dependents

2. **Service Analysis**:
   - Phone Service: 90.3% penetration
   - Internet Service: 78.9% penetration
     - DSL: 34.2%
     - Fiber Optic: 44.7%
   - Multiple Lines: 42.2% adoption

3. **Contract Analysis**:
   - Month-to-month: 55.1%
   - One year: 24.1%
   - Two year: 20.8%

4. **Financial Analysis**:
   - Monthly Charges: Mean $64.76, Std $30.09
   - Total Charges: Mean $2,283.30, Std $2,266.77
   - Tenure: Mean 32.37 months, Std 24.56 months

### Data Preprocessing
1. **Data Cleaning**:
   - Standardized column names (snake_case)
   - Handled missing values in TotalCharges (11 records)
   - Removed duplicate entries
   - Converted SeniorCitizen to categorical (0/1 to No/Yes)

2. **Feature Transformation**:
   - StandardScaler for numerical features
   - PowerTransformer for skewed numerical features
   - One-hot encoding for categorical variables
   - Binary encoding for binary categorical variables

3. **Data Validation**:
   - Range checks for numerical features
   - Category validation for categorical features
   - Consistency checks between related features

### Key Findings
1. **High-Risk Factors**:
   - Month-to-month contracts: 42.7% churn rate
   - Electronic check payments: 33.6% churn rate
   - Fiber optic internet: 41.9% churn rate
   - No online security: 41.6% churn rate
   - No tech support: 40.8% churn rate

2. **Protective Factors**:
   - Two-year contracts: 2.7% churn rate
   - Automatic payment methods: 15.8% churn rate
   - Online security: 15.7% churn rate
   - Tech support: 19.2% churn rate

## Milestone 2: Feature Engineering and Analysis

### Feature Engineering
1. **Service-Based Features**:
   - ServiceCount: Total number of services (range: 1-8)
   - OnlineServices: Combined online security/backup flag
   - StreamingServices: Combined TV/Movies streaming flag
   - ServiceDiversity: Number of different service types
   - PremiumServiceRatio: Ratio of premium to basic services

2. **Financial Features**:
   - AvgMonthlyCharges: TotalCharges/Tenure
   - ChargesPerYear: Annualized charges
   - MonthlyChargesPerService: Cost efficiency metric
   - PriceChangeIndicator: Recent price changes
   - RelativePricePosition: Price compared to service average

3. **Risk Scores**:
   - ContractRiskScore: Based on contract type
     - Month-to-month: 3
     - One year: 2
     - Two year: 1
   - PaymentRiskScore: Based on payment method
     - Electronic check: 3
     - Mailed check: 2
     - Automatic: 1

4. **Temporal Features**:
   - TenureGroups: Categorized tenure periods
   - ServiceAge: Time since first service
   - LastServiceChange: Days since last service modification

### Statistical Analysis
1. **Chi-Square Tests**:
   - Contract type vs. Churn: χ² = 857.32, p < 0.001
   - Payment method vs. Churn: χ² = 523.45, p < 0.001
   - Internet service vs. Churn: χ² = 412.78, p < 0.001

2. **Correlation Analysis**:
   - Tenure vs. Churn: -0.35
   - Monthly Charges vs. Churn: 0.19
   - Service Count vs. Churn: -0.28

3. **Feature Importance**:
   - Contract type: 0.32
   - Tenure: 0.25
   - Monthly charges: 0.15
   - Payment method: 0.12
   - Service count: 0.08
   - Others: 0.08

## Milestone 3: Model Development and Optimization

### Model Architecture
1. **GradientBoostingClassifier**:
   - n_estimators: 200
   - learning_rate: 0.1
   - max_depth: 5
   - min_samples_split: 10
   - min_samples_leaf: 4
   - subsample: 0.8
   - random_state: 42

2. **Feature Set**:
   - 55 engineered features
   - 11 numerical features
   - 44 categorical features (one-hot encoded)

### Model Performance
1. **Overall Metrics**:
   - Accuracy: 78.8%
   - ROC AUC: 84.2%
   - Precision: 60.2%
   - Recall: 59.1%
   - F1 Score: 59.6%

2. **Class-wise Performance**:
   - Churn Class:
     - Precision: 60.2%
     - Recall: 59.1%
     - F1: 59.6%
   - Non-churn Class:
     - Precision: 85.3%
     - Recall: 85.9%
     - F1: 85.6%

3. **Cross-validation Results**:
   - 5-fold CV Accuracy: 77.9% ± 1.2%
   - 5-fold CV ROC AUC: 83.8% ± 1.5%

### Model Validation
1. **Train-Test Split**:
   - Training set: 80% (5,634 samples)
   - Test set: 20% (1,409 samples)
   - Stratified sampling to maintain class distribution

2. **Validation Methods**:
   - K-fold cross-validation (k=5)
   - Stratified sampling
   - Time-based validation

3. **Performance Stability**:
   - Standard deviation of CV scores: 1.2%
   - Confidence intervals: 95%

## Milestone 4: Deployment and Risk Analysis

### Web Application
1. **Framework**: Streamlit
2. **Deployment**: Streamlit Cloud
3. **URL**: [Customer Churn Predictor](https://omaraymanmohamed-customer-churn-prediction-app-xxxxx.streamlit.app)
4. **Features**:
   - Interactive form for customer data
   - Real-time predictions
   - Risk factor analysis
   - Detailed results view
   - Performance metrics

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

3. **Risk Calculation**:
   - Base probability from model
   - Apply risk multipliers
   - Normalize final probability
   - Threshold-based classification

### Prediction Pipeline
1. **Data Collection**:
   - Customer demographics
   - Service information
   - Contract details
   - Payment information

2. **Feature Engineering**:
   - Create engineered features
   - Apply scaling and transformations
   - Calculate risk scores

3. **Prediction Generation**:
   - Base probability from model
   - Risk factor adjustment
   - Final churn probability

4. **Results Presentation**:
   - Churn probability
   - Identified risk factors
   - Detailed analysis

## Model Monitoring and Maintenance

### Logging System
1. **Prediction Logging**:
   - Timestamp
   - Input features
   - Prediction results
   - Risk factors
   - Model version

2. **Error Tracking**:
   - Error types
   - Frequency
   - Impact
   - Resolution time

3. **Performance Metrics**:
   - Daily/weekly/monthly statistics
   - Drift detection
   - Accuracy trends

### Model Updates
1. **Regular Evaluation**:
   - Weekly performance review
   - Monthly comprehensive assessment
   - Quarterly model retraining

2. **Retraining Triggers**:
   - Performance degradation > 5%
   - Data drift detection
   - New feature availability
   - Significant business changes

3. **Version Control**:
   - Model versioning
   - Feature set tracking
   - Performance history

### Future Improvements
1. **Model Enhancements**:
   - Ensemble methods
   - Deep learning approaches
   - Time series analysis
   - Bayesian optimization

2. **Feature Engineering**:
   - Customer behavior patterns
   - Interaction effects
   - Temporal features
   - External data integration

3. **Application Features**:
   - Batch prediction support
   - API integration
   - Custom risk factor configuration
   - Advanced visualization

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

## Conclusion
This project successfully implements a comprehensive customer churn prediction system that combines advanced machine learning techniques with practical business insights. The solution provides accurate predictions, detailed risk analysis, and actionable insights for customer retention strategies. The web application makes the system accessible to business users, while the monitoring system ensures long-term reliability and performance.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 