import matplotlib.pyplot as plt

def plot_data(df):
    """Plot the closing prices and moving averages."""
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Closing Price')
    plt.plot(df['Date'], df['SMA_50'], label='50-Day SMA')
    plt.plot(df['Date'], df['SMA_200'], label='200-Day SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def plot_predictions(y_test, y_pred):
    """Plot the actual vs predicted stock prices."""
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.values, label='Actual Prices')
    plt.plot(y_pred, label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
