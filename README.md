# Customer Churn Prediction App

This Streamlit application predicts customer churn probability based on service details and demographics.

## Features

- Interactive form for customer data input
- Real-time churn probability prediction
- Risk factor analysis
- Detailed prediction results
- Model performance metrics

## Live Demo

[View the live app on Streamlit Cloud](https://your-app-name.streamlit.app)

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
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
├── src/                   # Source code
│   ├── config/           # Configuration files
│   └── utils/            # Utility functions
├── model_artifacts/      # Trained models and artifacts
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Model Information

- Model Type: GradientBoostingClassifier
- Accuracy: 78.8%
- ROC AUC: 84.2%
- Precision: 60.2%
- Recall: 59.1%
- F1 Score: 59.6%

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 