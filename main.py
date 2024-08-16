from src.data_processing import load_data, preprocess_data
from src.feature_engineering import add_technical_indicators, create_features
from src.model import train_model
from src.visualization import plot_data, plot_predictions

def main():
    # Load and preprocess data
    df = load_data('data/stock_data.csv')
    print(f"Data after loading: {df.shape}")  # Debugging output

    df = preprocess_data(df)
    print(f"Data after preprocessing: {df.shape}")  # Debugging output

    # Feature engineering
    df = add_technical_indicators(df)
    df = create_features(df)
    print(f"Data after feature engineering: {df.shape}")  # Debugging output

    # Check if df is empty before proceeding
    if df.empty:
        print("No data available after preprocessing and feature engineering.")
        return

    # Train model
    model = train_model(df)

    # Visualize data
    plot_data(df)

if __name__ == '__main__':
    main()
