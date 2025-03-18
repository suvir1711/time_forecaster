import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from core.evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    EvaluatorError,
    ValidationError
)

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    true_values = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
        [5.0, 10.0]
    ])
    predictions = true_values + np.random.normal(0, 0.1, true_values.shape)
    target_cols = ['value1', 'value2']
    return true_values, predictions, target_cols

@pytest.fixture
def evaluator():
    """Create an evaluator instance"""
    config = EvaluationConfig(
        plot_style="seaborn",
        figure_size=(8, 4),
        confidence_interval=0.95
    )
    return ModelEvaluator(config)

def test_init_default():
    """Test initialization with default config"""
    evaluator = ModelEvaluator()
    assert evaluator.config.plot_style == "seaborn"
    assert evaluator.config.figure_size == (12, 6)
    assert evaluator.config.confidence_interval == 0.95

def test_init_custom_config():
    """Test initialization with custom config"""
    config = EvaluationConfig(plot_style="classic", figure_size=(10, 5))
    evaluator = ModelEvaluator(config)
    assert evaluator.config.plot_style == "classic"
    assert evaluator.config.figure_size == (10, 5)

def test_validate_evaluation_inputs_shape_mismatch(evaluator):
    """Test validation with shape mismatch"""
    true_values = np.array([[1, 2], [3, 4]])
    predictions = np.array([[1, 2]])
    with pytest.raises(ValidationError, match="Shape mismatch"):
        evaluator._validate_evaluation_inputs(true_values, predictions, ['col1', 'col2'])

def test_validate_evaluation_inputs_column_mismatch(evaluator):
    """Test validation with column count mismatch"""
    true_values = np.array([[1, 2], [3, 4]])
    predictions = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValidationError, match="Number of columns"):
        evaluator._validate_evaluation_inputs(true_values, predictions, ['col1'])

def test_calculate_metrics(evaluator):
    """Test metric calculation"""
    true_values = np.array([1.0, 2.0, 3.0])
    predictions = np.array([1.1, 2.1, 3.1])
    metrics = evaluator._calculate_metrics(true_values, predictions)
    
    assert 'RMSE' in metrics
    assert 'MAE' in metrics
    assert 'R2' in metrics
    assert metrics['RMSE'] > 0
    assert metrics['MAE'] > 0
    assert metrics['R2'] <= 1.0

def test_evaluate_model(evaluator, sample_data):
    """Test model evaluation"""
    true_values, predictions, target_cols = sample_data
    results = evaluator.evaluate_model('test_model', true_values, predictions, target_cols)
    
    assert isinstance(results, dict)
    assert all(col in results for col in target_cols)
    assert all(metric in results[target_cols[0]] for metric in ['RMSE', 'MAE', 'R2'])

def test_compare_models_empty(evaluator):
    """Test model comparison with no evaluations"""
    df = evaluator.compare_models()
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ["Model", "Target", "RMSE", "MAE", "R2"]

def test_compare_models(evaluator, sample_data):
    """Test model comparison with evaluations"""
    true_values, predictions, target_cols = sample_data
    
    # Evaluate two models
    evaluator.evaluate_model('model1', true_values, predictions, target_cols)
    evaluator.evaluate_model('model2', true_values * 1.1, predictions, target_cols)
    
    df = evaluator.compare_models()
    assert len(df) == 4  # 2 models * 2 targets
    assert 'Model' in df.columns
    assert 'Target' in df.columns
    assert all(metric in df.columns for metric in ['RMSE', 'MAE', 'R2'])

def test_compare_models_top_n(evaluator, sample_data):
    """Test model comparison with top_n parameter"""
    true_values, predictions, target_cols = sample_data
    
    # Evaluate three models
    evaluator.evaluate_model('model1', true_values, predictions, target_cols)
    evaluator.evaluate_model('model2', true_values * 1.1, predictions, target_cols)
    evaluator.evaluate_model('model3', true_values * 1.2, predictions, target_cols)
    
    df = evaluator.compare_models(top_n=2)
    assert len(df) == 2

def test_plot_predictions(evaluator, sample_data):
    """Test prediction plotting"""
    true_values, predictions, target_cols = sample_data
    fig = evaluator.plot_predictions('test_model', true_values, predictions, target_cols)
    
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == len(target_cols)

def test_plot_predictions_with_dates(evaluator, sample_data):
    """Test prediction plotting with datetime index"""
    true_values, predictions, target_cols = sample_data
    dates = pd.date_range(start='2023-01-01', periods=len(true_values), freq='D')
    
    fig = evaluator.plot_predictions('test_model', true_values, predictions, target_cols, dates)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == len(target_cols)

def test_get_metric_summary_invalid_model(evaluator):
    """Test metric summary with invalid model"""
    with pytest.raises(ValueError, match="No evaluation results found"):
        evaluator.get_metric_summary('nonexistent_model')

def test_get_metric_summary(evaluator, sample_data):
    """Test metric summary"""
    true_values, predictions, target_cols = sample_data
    evaluator.evaluate_model('test_model', true_values, predictions, target_cols)
    
    summary = evaluator.get_metric_summary('test_model')
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == len(target_cols)
    assert all(col in summary.columns for col in ['Target', 'RMSE', 'MAE', 'R2'])

def test_plot_metric_comparison(evaluator, sample_data):
    """Test metric comparison plotting"""
    true_values, predictions, target_cols = sample_data
    
    # Evaluate multiple models
    evaluator.evaluate_model('model1', true_values, predictions, target_cols)
    evaluator.evaluate_model('model2', true_values * 1.1, predictions, target_cols)
    
    fig = evaluator.plot_metric_comparison(metric='RMSE')
    assert isinstance(fig, plt.Figure)

def test_add_confidence_interval(evaluator):
    """Test confidence interval addition"""
    fig, ax = plt.subplots()
    x = np.arange(5)
    true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = true_values + np.random.normal(0, 0.1, true_values.shape)
    
    evaluator._add_confidence_interval(ax, x, true_values, predictions)
    assert len(ax.collections) > 0  # Check if fill_between was called

@pytest.mark.parametrize("metric", ["RMSE", "MAE", "R2"])
def test_different_metrics_comparison(evaluator, sample_data, metric):
    """Test model comparison with different metrics"""
    true_values, predictions, target_cols = sample_data
    evaluator.evaluate_model('test_model', true_values, predictions, target_cols)
    
    df = evaluator.compare_models(metric=metric)
    assert metric in df.columns
    assert not df[metric].isna().any()

def test_evaluation_with_perfect_predictions(evaluator):
    """Test evaluation with perfect predictions"""
    true_values = np.array([[1.0, 2.0], [3.0, 4.0]])
    predictions = true_values.copy()  # Perfect predictions
    target_cols = ['value1', 'value2']
    
    results = evaluator.evaluate_model('perfect_model', true_values, predictions, target_cols)
    assert all(results[col]['RMSE'] == 0 for col in target_cols)
    assert all(results[col]['R2'] == 1 for col in target_cols) 