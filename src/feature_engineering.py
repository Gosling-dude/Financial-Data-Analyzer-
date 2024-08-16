import numpy as np

def add_technical_indicators(df):
    """Add technical indicators like moving averages to the dataset."""
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    # Add more indicators as needed
    return df

def create_features(df):
    """Create features for machine learning models."""
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    return df.dropna()
