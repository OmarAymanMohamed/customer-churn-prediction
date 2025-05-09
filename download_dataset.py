"""
Download the Telco Customer Churn dataset using kagglehub.
"""

import kagglehub
import os
import pandas as pd

def download_dataset():
    """Download the dataset and move it to the project directory."""
    print("Downloading dataset...")
    
    # Download the dataset
    dataset_path = kagglehub.dataset_download(
        "yeanzc/telco-customer-churn-ibm-dataset"
    )
    
    print("Extracting files...")
    print(f"Path to dataset files: {dataset_path}")
    
    # Find the Excel file in the downloaded directory
    excel_file = None
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.xlsx'):
                excel_file = os.path.join(root, file)
                break
        if excel_file:
            break
    
    if excel_file is None:
        raise Exception("Could not find Excel file in downloaded dataset")
    
    print(f"Found Excel file: {excel_file}")
    
    # Read the Excel file and convert to CSV
    print("Converting Excel to CSV...")
    df = pd.read_excel(excel_file)
    
    # Save as CSV in the project directory
    output_path = os.path.join(os.getcwd(), "Telco_customer_churn.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved as CSV at: {output_path}")

if __name__ == "__main__":
    download_dataset() 