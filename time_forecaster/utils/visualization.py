import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_stock_data(df, title="Stock Price with Technical Indicators", figsize=(12, 8)):
    """
    Plot stock price data with technical indicators including SMA and Bollinger Bands.

    Args:
        df (pd.DataFrame): DataFrame with stock price data.
        title (str, optional): Title of the plot. Defaults to "Stock Price with Technical Indicators".
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure object.
    """
    if 'Close' not in df.columns:
        logger.error("DataFrame must contain 'Close' column for plotting price data.")
        raise ValueError("DataFrame must contain 'Close' column for plotting.")

    df = df.copy()
    df['Date'] = pd.to_datetime(df.index)  # Ensure 'Date' is in datetime format

    fig, axes = plt.subplots(nrows=2, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot stock price with moving averages and Bollinger bands
    axes[0].plot(df['Date'], df['Close'], label='Close Price', color='blue', linewidth=1.5)
    
    if 'SMA_20' in df.columns:
        ax = axes[0]
        ax.plot(df['Date'], df['SMA_20'], label='SMA (20)', linestyle='--', color='orange')

    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        ax.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.3, label='Bollinger Bands')
    
    axes[0].set_ylabel('Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Volume plot
    if 'Volume' in df.columns:
        ax = axes[1]
        ax.bar(df['Date'], df['Volume'], color='gray', alpha=0.4)
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.show()
    return fig

def plot_feature_importance(feature_names, importances, title="Feature Importance"):
    """
    Plot feature importance.

    Args:
        feature_names (list): List of feature names.
        importances (list): List of importance values.
        title (str): Plot title.

    Returns:
        plt.Figure: The generated plot figure.
    """
    if len(feature_names) != len(importances):
        logger.error("Feature names and importance values must have the same length.")
        raise ValueError("Feature names and importances must have the same length.")

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=True)  # Sort for better visualization

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df, y='Feature', x='Importance', ax=ax, palette="Blues_r")

    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    plt.show()
    return fig

def plot_correlation_matrix(df, title="Feature Correlation Matrix"):
    """
    Plot a correlation matrix for numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        title (str): Title of the plot.

    Returns:
        plt.Figure: The generated plot figure.
    """
    df = df.copy()
    if df.empty:
        logger.warning("The input DataFrame is empty.")
        return None

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title(title)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()
    return fig