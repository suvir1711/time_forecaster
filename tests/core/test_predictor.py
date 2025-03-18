import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from core.predictor import (
    TimeSeriesPredictor,
    PredictionConfig,
    ModelNotSetError,
    PredictionError
)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    data = {
        'value1': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
        'value2': np.cos(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def mock_model():
    """Create a mock model"""
    model = Mock()
    model.generate.return_value = torch.tensor([[1, 2, 3]])  # Mock output tokens
    return model

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer"""
    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.decode.return_value = "Given the following time series data: [...]\nPredict the next values: 0.5, 0.6"
    return tokenizer

@pytest.fixture
def predictor(mock_model, mock_tokenizer):
    """Create a predictor instance with mock model and tokenizer"""
    config = PredictionConfig(use_tqdm=False)  # Disable tqdm for testing
    return TimeSeriesPredictor(mock_model, mock_tokenizer, config=config)

def test_init_default():
    """Test initialization with default values"""
    predictor = TimeSeriesPredictor()
    assert predictor.model is None
    assert predictor.tokenizer is None
    assert predictor.device in ['cuda', 'cpu']
    assert isinstance(predictor.config, PredictionConfig)

def test_init_custom_config():
    """Test initialization with custom config"""
    config = PredictionConfig(temperature=0.5, max_new_tokens=100)
    predictor = TimeSeriesPredictor(config=config)
    assert predictor.config.temperature == 0.5
    assert predictor.config.max_new_tokens == 100

def test_set_model():
    """Test setting model and tokenizer"""
    predictor = TimeSeriesPredictor()
    model = Mock()
    tokenizer = Mock()
    predictor.set_model(model, tokenizer)
    assert predictor.model == model
    assert predictor.tokenizer == tokenizer

def test_predict_without_model():
    """Test prediction without setting model"""
    predictor = TimeSeriesPredictor()
    with pytest.raises(ModelNotSetError):
        predictor.predict(pd.DataFrame(), 1, ['value'])

def test_predict_empty_df(predictor):
    """Test prediction with empty DataFrame"""
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        predictor.predict(pd.DataFrame(), 1, ['value'])

def test_predict_missing_columns(predictor, sample_df):
    """Test prediction with missing target columns"""
    with pytest.raises(ValueError, match="Missing target columns"):
        predictor.predict(sample_df, 1, ['nonexistent'])

def test_predict_invalid_steps(predictor, sample_df):
    """Test prediction with invalid number of steps"""
    with pytest.raises(ValueError, match="Number of prediction steps must be positive"):
        predictor.predict(sample_df, 0, ['value1'])

def test_predict_invalid_sequence_length(predictor, sample_df):
    """Test prediction with invalid sequence length"""
    with pytest.raises(ValueError, match="Sequence length must be positive"):
        predictor.predict(sample_df, 1, ['value1'], sequence_length=0)

def test_predict_insufficient_data(predictor):
    """Test prediction with insufficient data"""
    df = pd.DataFrame({'value1': [1, 2, 3]})
    with pytest.raises(ValueError, match="Input dataframe must have at least"):
        predictor.predict(df, 1, ['value1'], sequence_length=5)

def test_successful_prediction(predictor, sample_df):
    """Test successful prediction"""
    result = predictor.predict(sample_df, 3, ['value1', 'value2'])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ['value1', 'value2']
    assert isinstance(result.index, pd.DatetimeIndex)

def test_format_input_for_model(predictor):
    """Test input formatting"""
    sequence = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = predictor._format_input_for_model(sequence)
    assert isinstance(result, dict)
    assert "input_ids" in result

def test_get_model_prediction(predictor):
    """Test model prediction generation"""
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    result = predictor._get_model_prediction(inputs)
    assert isinstance(result, str)
    assert "Predict the next values" in result

def test_extract_prediction_values(predictor):
    """Test prediction value extraction"""
    output = "Given the following time series data: [...]\nPredict the next values: 0.5, 0.6"
    values = predictor._extract_prediction_values(output, ['value1', 'value2'])
    assert isinstance(values, np.ndarray)
    assert len(values) == 2
    assert values[0] == 0.5
    assert values[1] == 0.6

def test_extract_prediction_values_error_handling(predictor):
    """Test error handling in prediction value extraction"""
    output = "Invalid output format"
    values = predictor._extract_prediction_values(output, ['value1', 'value2'])
    assert isinstance(values, np.ndarray)
    assert len(values) == 2
    assert all(v == 0 for v in values)

def test_create_prediction_dataframe(predictor, sample_df):
    """Test prediction DataFrame creation"""
    predictions = [np.array([0.5, 0.6]), np.array([0.7, 0.8])]
    result = predictor._create_prediction_dataframe(
        sample_df, predictions, 2, ['value1', 'value2']
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ['value1', 'value2']
    assert isinstance(result.index, pd.DatetimeIndex)

@patch('torch.cuda.is_available')
def test_device_selection(mock_cuda_available):
    """Test device selection logic"""
    mock_cuda_available.return_value = True
    predictor = TimeSeriesPredictor()
    assert predictor.device == 'cuda'

    mock_cuda_available.return_value = False
    predictor = TimeSeriesPredictor()
    assert predictor.device == 'cpu'

def test_prediction_with_custom_config():
    """Test prediction with custom configuration"""
    config = PredictionConfig(
        temperature=0.5,
        max_new_tokens=100,
        num_return_sequences=2,
        batch_size=32
    )
    predictor = TimeSeriesPredictor(config=config)
    assert predictor.config.temperature == 0.5
    assert predictor.config.max_new_tokens == 100
    assert predictor.config.num_return_sequences == 2
    assert predictor.config.batch_size == 32 