import pandas as pd
import numpy as np
import yfinance as yf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_stock_data(ticker, start_date, end_date, interval='1d'):
    """
    Load stock price data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Interval for data ('1d', '1wk', '1mo', etc.).

    Returns:
        pd.DataFrame: DataFrame with stock price data including technical indicators.
    """
    try:
        logger.info(f"Fetching stock data for {ticker} from {start_date} to {end_date} with {interval} interval")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        if data.empty:
            logger.warning(f"No data found for {ticker} in the given date range.")
            return None

        # Reset index to make 'Date' a column
        data.reset_index(inplace=True)

        # Add technical indicators
        data = add_technical_indicators(data)

        logger.info(f"Successfully loaded stock data. Data shape: {data.shape}")
        return data

    except Exception as e:
        logger.error(f"Failed to load stock data: {str(e)}")
        return None

def add_technical_indicators(df):
    """
    Add technical indicators such as SMA, EMA, RSI, MACD, and Bollinger Bands.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    df = df.copy()
    
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = -np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (2 * rolling_std)

    # Fill NaN values appropriately
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    return df

def create_lagged_features(df, columns, lags):
    """
    Create lagged features for specified columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of columns to create lags for.
        lags (list): List of lag values.

    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    df_copy = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in dataframe.")
            continue

        for lag in lags:
            df_copy[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Drop initial NaN values resulting from shift
    df_copy.dropna(inplace=True)

    return df_copy

if __name__ == "__main__":
    logging.info("Starting stock data retrieval process...")
    
    # Example Usage
    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    interval = "1d"

    df = load_stock_data(ticker, start_date, end_date, interval)

    if df is not None:
        print("Sample Data:\n", df.head())

        # Create lagged features for 'Close' price
        lagged_df = create_lagged_features(df, columns=['Close'], lags=[1, 3, 5])
        print("\nSample Data with Lagged Features:\n", lagged_df.head())
    else:
        print(f"No data found for {ticker}.")