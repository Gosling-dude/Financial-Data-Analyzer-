import pandas as pd

def load_data(filepath):
    """Load the financial dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the dataset, handling missing values and feature scaling."""
    df.fillna(method='ffill', inplace=True)
    # Add more preprocessing steps as needed
    return df
