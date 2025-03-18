# Time Forecaster

A powerful Python library for time series forecasting using state-of-the-art language models. This library provides a simple interface to load, preprocess, predict, and evaluate time series data using pre-trained models from HuggingFace.

## Features

- Easy-to-use interface for time series forecasting
- Support for multiple target variables
- Integration with HuggingFace models
- Comprehensive data preprocessing
- Advanced evaluation metrics and visualization
- Support for both HuggingFace and local models
- Automatic device management (CPU/GPU)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/time_forecaster.git
cd time_forecaster
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

Here's a simple example of how to use the library:

```python
from time_forecaster import TimeForecaster
import pandas as pd
import yfinance as yf

# Initialize the forecaster
forecaster = TimeForecaster()

# Load a model from HuggingFace
model_name = "EleutherAI/gpt-neo-125m"
forecaster.load_model("stock_model", model_name)

# Fetch some stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(period="1y")

# Preprocess the data
target_cols = ['Close', 'Volume']
preprocessed_data = forecaster.preprocess_data(
    df, 
    target_cols=target_cols,
    sequence_length=30,
    train_test_split=0.8
)

# Generate predictions
predictions = forecaster.predict(
    "stock_model",
    df,
    num_steps=7,
    target_cols=target_cols
)

# Evaluate the predictions
metrics = forecaster.evaluate(
    "stock_model",
    preprocessed_data['y_test'][-7:],
    predictions.values,
    target_cols
)

# Plot the results
forecaster.plot_predictions(
    "stock_model",
    preprocessed_data['y_test'][-7:],
    predictions.values,
    target_cols,
    dates=predictions.index
)
```

## Detailed Documentation

### TimeForecaster Class

The main interface for the library. Provides methods for model management, data preprocessing, prediction, and evaluation.

#### Methods

##### `__init__(cache_dir=None)`
Initialize the TimeForecaster instance.
- `cache_dir`: Optional directory for caching downloaded models

##### `load_model(model_name, model_path, model_type="hf", **kwargs)`
Load a model from HuggingFace or local directory.
- `model_name`: Name to reference this model
- `model_path`: Path to the model (HuggingFace model name or local path)
- `model_type`: "hf" for HuggingFace or "local" for local path
- `**kwargs`: Additional arguments for model loading

##### `preprocess_data(df, target_cols, sequence_length=60, train_test_split=0.8)`
Preprocess time series data for model input.
- `df`: Input dataframe with time series data
- `target_cols`: List of target columns to predict
- `sequence_length`: Length of input sequences
- `train_test_split`: Ratio of training data

##### `predict(model_name, input_df, num_steps, target_cols, sequence_length=60)`
Generate predictions using the specified model.
- `model_name`: Name of the model to use
- `input_df`: Input dataframe with time series data
- `num_steps`: Number of steps to predict into the future
- `target_cols`: List of target columns to predict
- `sequence_length`: Length of input sequences

##### `evaluate(model_name, true_values, predictions, target_cols)`
Evaluate model predictions.
- `model_name`: Name of the model being evaluated
- `true_values`: Ground truth values
- `predictions`: Predicted values
- `target_cols`: Names of target columns

##### `compare_models(top_n=None, metric='RMSE')`
Compare all evaluated models.
- `top_n`: Number of top models to return
- `metric`: Metric to sort by

##### `plot_predictions(model_name, true_values, predictions, target_cols, dates=None)`
Plot predictions against true values.
- `model_name`: Name of the model
- `true_values`: Ground truth values
- `predictions`: Predicted values
- `target_cols`: Names of target columns
- `dates`: Optional datetime index for x-axis

### Core Components

#### ModelLoader
Handles loading of pre-trained models from HuggingFace or local directories.
- Supports model caching
- Automatic device management
- Tokenizer handling

#### TimeSeriesPreprocessor
Preprocesses time series data for model input.
- Handles missing values
- Creates sequences
- Scales data
- Splits data into train/test sets

#### TimeSeriesPredictor
Generates predictions using loaded models.
- Handles sequence formatting
- Manages model inference
- Processes model outputs

#### ModelEvaluator
Evaluates model predictions and generates visualizations.
- Calculates various metrics (RMSE, MAE, R2)
- Creates prediction plots
- Compares model performance

## Examples

### Stock Price Prediction
See `examples/stock_prediction.py` for a complete example of predicting stock prices and volumes.

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- Transformers >= 4.11.0
- Pandas >= 1.3.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Scikit-learn >= 0.24.0
- yfinance >= 0.1.63
- accelerate >= 0.26.0

## Contributing

We welcome contributions to Time Forecaster! Here's how you can help:

### Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/your-username/time_forecaster.git
cd time_forecaster
```

3. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

4. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function arguments and return values
- Add docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Write clear, descriptive variable names

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Run tests with:
```bash
pytest tests/
```

### Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if you're changing functionality
3. Add tests for new features
4. Ensure the test suite passes
5. Update the version number in setup.py
6. Create a Pull Request with a clear description of your changes

### Code Review Process

1. All PRs require at least one review
2. Address review comments promptly
3. Keep commits focused and atomic
4. Rebase your branch on main if needed

### Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Your environment details (Python version, OS, etc.)

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Time Forecaster

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

- HuggingFace for providing the transformer models
- The PyTorch team for the deep learning framework
- The pandas team for data manipulation tools
- The scikit-learn team for evaluation metrics 