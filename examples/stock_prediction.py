import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Optional
from datetime import datetime, timedelta
from time_forecaster.core.model_loader import ModelLoader
from time_forecaster.core.predictor import TimeSeriesPredictor
from time_forecaster.core.preprocessor import TimeSeriesPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker_symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker_symbol: Stock ticker symbol
        period: Time period to fetch data for
        
    Returns:
        DataFrame with stock data or None if fetch fails
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            logger.error(f"No data found for {ticker_symbol}")
            return None
            
        # Fill missing values using ffill and bfill
        df = df.ffill()  # Forward fill
        df = df.bfill()  # Backward fill for any remaining NaNs at the start
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker_symbol}: {str(e)}")
        return None

def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        prices: Series of prices
        periods: RSI period
        
    Returns:
        Series with RSI values
    """
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=prices.index)

def prepare_model_data(
    df: pd.DataFrame,
    target_cols: list,
    sequence_length: int,
    train_test_split: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for model training and testing
    
    Args:
        df: Input DataFrame
        target_cols: Target columns to predict
        sequence_length: Length of input sequences
        train_test_split: Ratio for train/test split
        
    Returns:
        Tuple of (train_data, test_data)
    """
    try:
        # Ensure we have enough data
        if len(df) < sequence_length:
            raise ValueError(f"Not enough data points. Need at least {sequence_length}")
            
        # Split point
        split_idx = int(len(df) * train_test_split)
        
        # Split data
        train_data = df[:split_idx]
        test_data = df[split_idx:]
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error preparing model data: {str(e)}")
        raise

def main():
    try:
        # Initialize components
        predictor = TimeSeriesPredictor()
        preprocessor = TimeSeriesPreprocessor()
        
        # Load model
        model_name = "EleutherAI/gpt-neo-125m"
        logger.info(f"Loading model from {model_name}...")
        try:
            model_loader = ModelLoader()
            model, tokenizer = model_loader.load_model(model_name)
            predictor.set_model(model, tokenizer)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return

        # Fetch stock data
        ticker_symbol = "AAPL"
        logger.info(f"Fetching data for {ticker_symbol}...")
        df = fetch_stock_data(ticker_symbol)  # Using default period of "1y"
        if df is None:
            return
            
        # Select target columns for prediction
        target_cols = ['Close', 'Volume']
        
        # Preprocess data
        logger.info("Preprocessing data...")
        try:
            preprocessed_data = preprocessor.preprocess(
                df,
                target_cols=target_cols,
                sequence_length=30,  # 30 days of history
                train_test_split=0.8
            )
            train_data = preprocessed_data['X_train']
            test_data = preprocessed_data['X_test']
        except Exception as e:
            logger.error(f"Failed to preprocess data: {str(e)}")
            return
        
        # Generate predictions
        logger.info("Generating predictions...")
        prediction_days = 7  # Predict next 7 days
        try:
            predictions = predictor.predict(
                test_data,
                num_steps=prediction_days,
                target_cols=target_cols,
                sequence_length=30
            )
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            return
        
        # Evaluate model
        logger.info("Evaluating predictions...")
        try:
            metrics = predictor.evaluate(
                preprocessed_data['y_test'][-prediction_days:],
                predictions.values,
                target_cols
            )
            
            # Print evaluation metrics
            print("\nEvaluation Metrics:")
            for col in target_cols:
                print(f"\n{col}:")
                print(f"RMSE: {metrics[col]['RMSE']:.2f}")
                print(f"MAE: {metrics[col]['MAE']:.2f}")
                print(f"R2: {metrics[col]['R2']:.2f}")
        except Exception as e:
            logger.error(f"Failed to evaluate predictions: {str(e)}")
            return
        
        # Plot predictions
        logger.info("Plotting predictions...")
        try:
            # Create figure with error handling
            plt.ion()  # Turn on interactive mode
            fig = plt.figure(figsize=(15, 10))
            
            # Create date range for actual values
            actual_dates = pd.date_range(
                start=df.index[-1] - pd.Timedelta(days=prediction_days),
                periods=prediction_days
            )
            
            # Plot Close price predictions
            plt.subplot(2, 1, 1)
            plt.title(f"{ticker_symbol} Stock Price Prediction")
            plt.plot(actual_dates, preprocessed_data['y_test'][-prediction_days:, 0], 
                     label='Actual', color='blue')
            plt.plot(predictions.index, predictions['Close'].values, 
                     label='Predicted', color='red', linestyle='--')
            plt.legend()
            plt.grid(True)
            
            # Plot Volume predictions
            plt.subplot(2, 1, 2)
            plt.title(f"{ticker_symbol} Volume Prediction")
            plt.plot(actual_dates, preprocessed_data['y_test'][-prediction_days:, 1], 
                     label='Actual', color='blue')
            plt.plot(predictions.index, predictions['Volume'].values, 
                     label='Predicted', color='red', linestyle='--')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save the plot instead of showing it
            plt.savefig('stock_predictions.png')
            logger.info("Plot saved as 'stock_predictions.png'")
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to plot predictions: {str(e)}")
            logger.info("Skipping plot generation due to error")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")

if __name__ == "__main__":
    main() 