import numpy as np
import pandas as pd

def calculate_returns(prices, periods=[1, 5, 20]):
    """
    Calculate returns over different periods.

    Args:
        prices (pd.Series or pd.DataFrame): Time series of prices.
        periods (list, optional): List of periods for return calculation.

    Returns:
        pd.DataFrame: DataFrame containing calculated returns.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise ValueError("Input prices must be a pandas Series or single-column DataFrame")

    prices = prices.squeeze()  # Convert to Series if it's a single-column DataFrame
    returns = pd.DataFrame(index=prices.index)

    for period in periods:
        returns[f'return_{period}d'] = prices.pct_change(periods=period)

    return returns.dropna()

def calculate_volatility(returns, window=20):
    """
    Calculate rolling volatility.

    Args:
        returns (pd.Series): Series of returns.
        window (int, optional): Rolling window size.

    Returns:
        pd.Series: Rolling volatility.
    """
    if returns.isna().all():
        return pd.Series(np.nan, index=returns.index)

    return returns.rolling(window=window, min_periods=1).std() * np.sqrt(window)

def calculate_sharpe_ratio(returns):
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns (pd.Series): Daily returns.

    Returns:
        float: Sharpe ratio (NaN if std is zero).
    """
    if returns.isna().all():
        return np.nan

    risk_free_rate = 0.0  # Assuming risk-free rate is 0%
    mean_return = returns.mean() * 252  # Annualized return
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    
    if volatility == 0:
        return np.nan  # Avoid division by zero
    
    return mean_return / volatility

def calculate_drawdowns(prices):
    """
    Calculate drawdowns and maximum drawdown.

    Args:
        prices (pd.Series): Time series of prices.

    Returns:
        tuple: (pd.Series of drawdowns, float max drawdown)
    """
    prices = prices.dropna()
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

def calculate_win_rate(predictions, actuals, threshold=0):
    """
    Calculate win rate based on predicted and actual returns.

    Args:
        predictions (pd.Series): Predicted returns.
        actuals (pd.Series): Actual returns.
        threshold (float, optional): Minimum return to count as a win.

    Returns:
        float: Win rate.
    """
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have the same length.")

    predictions = predictions.dropna()
    actuals = actuals.loc[predictions.index]  # Ensure alignment

    correct = ((predictions > threshold) & (actuals > 0)) | ((predictions < -threshold) & (actuals < -threshold))
    win_rate = correct.mean()

    return win_rate if not np.isnan(win_rate) else np.nan