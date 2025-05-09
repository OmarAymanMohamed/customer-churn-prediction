import os
import json

# Create notebooks directory if it doesn't exist
os.makedirs("notebooks", exist_ok=True)

# Define the notebook content for each milestone
milestone1 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Milestone 1: Customer Churn Data Exploration and Preprocessing\n\n",
                      "This notebook covers the initial data exploration and preprocessing for the customer churn prediction project."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["# Install required packages\n",
                      "!pip install pandas numpy matplotlib seaborn scikit-learn"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Data Loading\n\n",
                      "Load the customer churn dataset for exploration."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["import pandas as pd\n",
                      "import numpy as np\n",
                      "import matplotlib.pyplot as plt\n",
                      "import seaborn as sns\n\n",
                      "# URL for the dataset\n",
                      "url = \"https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv\"\n",
                      "df = pd.read_csv(url)\n\n",
                      "# Display the first few rows\n",
                      "df.head()"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Key Components\n\n",
                      "1. Data loading\n",
                      "2. Exploratory data analysis\n",
                      "3. Data cleaning\n",
                      "4. Feature preparation\n",
                      "5. Dataset splitting"]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

milestone2 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Milestone 2: Feature Engineering\n\n",
                      "This notebook focuses on creating advanced features to improve churn prediction models."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["# Install required packages\n",
                      "!pip install pandas numpy matplotlib seaborn scikit-learn"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Feature Engineering Approach\n\n",
                      "We'll create several types of engineered features to improve our model performance:\n\n",
                      "1. Interaction features: Combine related variables to capture joint effects\n",
                      "2. Ratio features: Create meaningful ratios from numeric variables\n",
                      "3. Service bundle features: Group related services\n",
                      "4. Customer profile features: Create customer segments"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["import pandas as pd\n",
                      "import numpy as np\n\n",
                      "# Feature engineering function\n",
                      "def create_engineered_features(X):\n",
                      "    X_new = X.copy()\n",
                      "    \n",
                      "    # Create your engineered features here\n",
                      "    \n",
                      "    return X_new"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Key Components\n\n",
                      "1. Interaction features\n",
                      "2. Ratio features\n",
                      "3. Service bundle features\n",
                      "4. Customer profile features\n",
                      "5. Feature importance analysis"]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

milestone3 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Milestone 3: Model Development for Customer Churn Prediction\n\n",
                      "This notebook focuses on developing, evaluating, and optimizing machine learning models for predicting customer churn."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["# Install required packages\n",
                      "!pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Model Evaluation Function\n\n",
                      "Let's create a function to evaluate model performance consistently."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["import pandas as pd\n",
                      "import numpy as np\n",
                      "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n\n",
                      "def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):\n",
                      "    \"\"\"Evaluate model performance on multiple metrics\"\"\"\n",
                      "    # Model evaluation code here\n",
                      "    pass"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Key Components\n\n",
                      "1. Baseline model training\n",
                      "2. Model comparison\n",
                      "3. Hyperparameter optimization\n",
                      "4. Feature importance analysis\n",
                      "5. Model interpretation with SHAP"]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

milestone4 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Milestone 4: Deployment and Monitoring\n\n",
                      "This notebook focuses on deploying the customer churn prediction model with a web interface."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["# Install required packages\n",
                      "!pip install pandas numpy streamlit joblib scikit-learn"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Create Streamlit Web Application\n\n",
                      "Now we'll create a Streamlit app for model deployment."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["# Create app.py file\n",
                      "app_code = '''\n",
                      "import streamlit as st\n",
                      "import pandas as pd\n",
                      "import numpy as np\n",
                      "import joblib\n",
                      "\n",
                      "# Title and description\n",
                      "st.title(\"Customer Churn Predictor\")\n",
                      "st.write(\"This application predicts the likelihood of a customer churning.\")\n",
                      "'''\n",
                      "\n",
                      "# Write app code to file\n",
                      "with open('app.py', 'w') as f:\n",
                      "    f.write(app_code)\n",
                      "\n",
                      "print(\"Created app.py with Streamlit web application code\")"]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Key Components\n\n",
                      "1. Model serialization\n",
                      "2. Web application development with Streamlit\n",
                      "3. Deployment instructions\n",
                      "4. Prediction logging\n",
                      "5. Monitoring setup"]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write notebooks to files
with open("notebooks/Milestone1_Data_Exploration_Preprocessing.ipynb", "w") as f:
    json.dump(milestone1, f, indent=1)

with open("notebooks/Milestone2_Feature_Engineering.ipynb", "w") as f:
    json.dump(milestone2, f, indent=1)

with open("notebooks/Milestone3_Model_Development.ipynb", "w") as f:
    json.dump(milestone3, f, indent=1)

with open("notebooks/Milestone4_Deployment.ipynb", "w") as f:
    json.dump(milestone4, f, indent=1)

print("Created all 4 milestone notebooks in the notebooks directory") 