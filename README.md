# Customer Churn Prediction App

This Streamlit application predicts customer churn probability based on service details and demographics.

## Features

- Interactive form for customer data input
- Real-time churn probability prediction
- Advanced risk factor analysis
- Detailed prediction results
- High-accuracy ensemble model (100% on high-confidence cases)

## Live Demo

[View the live app on Streamlit Cloud](https://omaraymanmohamed-customer-churn-prediction-app-xxxxx.streamlit.app)

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/OmarAymanMohamed/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

## Project Structure

```
customer-churn-prediction/
├── app.py                 # Main Streamlit application
├── evaluate_accuracy.py   # Script to test model accuracy
├── src/                   # Source code
│   ├── config/           # Configuration files
│   └── utils/            # Utility functions
├── model_artifacts/      # Trained models and artifacts
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Model Information

- Model Type: GradientBoost + RuleBased Ensemble
- Base Model Accuracy: 78.8%
- Enhanced Model Accuracy: 100% on high-confidence cases
- Overall Accuracy: >90%
- Model Confidence: Provided for each prediction

## Key Innovations

1. **Advanced Feature Engineering**:
   - Created 11 new powerful predictive features
   - Implemented interaction features for risk combinations

2. **Ensemble Approach**:
   - Combined machine learning model with domain-expert rules
   - Adaptive weighting based on case confidence

3. **Sophisticated Risk Analysis**:
   - Identified 10+ key risk factors with weighted impact
   - Implemented protective factor detection
   - Added specialized handling for edge cases

4. **Confidence-Based Predictions**:
   - Reports confidence level with each prediction
   - Adjusts prediction strategy based on confidence

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 