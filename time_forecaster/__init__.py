from .core.model_loader import ModelLoader
from .core.preprocessor import TimeSeriesPreprocessor
from .core.predictor import TimeSeriesPredictor
from .core.evaluator import ModelEvaluator

import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class TimeForecaster:
    """Main interface for the Time Series Forecasting library"""
    
    def __init__(self, cache_dir=None):
        self.model_loader = ModelLoader(cache_dir=cache_dir)
        self.preprocessor = TimeSeriesPreprocessor()
        self.predictor = TimeSeriesPredictor()
        self.evaluator = ModelEvaluator()
        self.models = {}
        
    def load_model(self, model_name, model_path, model_type="hf", **kwargs):
        """
        Load a model from HuggingFace or local directory
        
        Args:
            model_name (str): Name to reference this model
            model_path (str): Path to the model
            model_type (str): "hf" for HuggingFace or "local" for local path
            **kwargs: Additional arguments for model loading
        """
        model, tokenizer = self.model_loader.load_model(model_path, model_type, **kwargs)
        
        model_data = {
            'model': model,
            'path': model_path,
            'type': model_type
        }
        if tokenizer is not None:
            model_data['tokenizer'] = tokenizer
        
        self.models[model_name] = model_data
        return model, tokenizer
    
    def preprocess_data(self, df, target_cols, sequence_length=60, train_test_split=0.8):
        """
        Preprocess time series data for model input
        
        Args:
            df (pd.DataFrame): Input dataframe with time series data
            target_cols (list): List of target columns to predict
            sequence_length (int): Length of input sequences
            train_test_split (float): Ratio of training data
        """
        if df.empty:
            raise ValueError("Input dataframe is empty.")
        if not all(col in df.columns for col in target_cols):
            raise ValueError(f"Some target columns {target_cols} are missing in the dataframe.")

        return self.preprocessor.preprocess(
            df, target_cols, sequence_length, train_test_split
        )
    
    def predict(self, model_name, input_df, num_steps, target_cols, sequence_length=60):
        """
        Generate predictions using the specified model
        
        Args:
            model_name (str): Name of the model to use
            input_df (pd.DataFrame): Input dataframe with time series data
            num_steps (int): Number of steps to predict into the future
            target_cols (list): List of target columns to predict
            sequence_length (int): Length of input sequences
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model_info = self.models[model_name]
        if not model_info.get('model'):
            raise ValueError(f"Model {model_name} is not properly loaded.")

        tokenizer = model_info.get('tokenizer', None)
        self.predictor.set_model(model_info['model'], tokenizer)
        
        return self.predictor.predict(
            input_df, num_steps, target_cols, sequence_length
        )
    
    def evaluate(self, model_name, true_values, predictions, target_cols):
        """
        Evaluate model predictions
        
        Args:
            model_name (str): Name of the model being evaluated
            true_values (np.array): Ground truth values
            predictions (np.array): Predicted values
            target_cols (list): Names of target columns
        """
        if true_values.shape != predictions.shape:
            raise ValueError(f"Shape mismatch: true_values {true_values.shape}, predictions {predictions.shape}")
        
        return self.evaluator.evaluate_model(
            model_name, true_values, predictions, target_cols
        )
    
    def compare_models(self, top_n=None, metric='RMSE'):
        """Compare all evaluated models"""
        if not self.evaluator.evaluation_results:
            return pd.DataFrame(columns=["Model", "RMSE", "MAE", "R2"])
        
        return self.evaluator.compare_models(top_n, metric)
    
    def plot_predictions(self, model_name, true_values, predictions, target_cols, dates=None):
        """Plot predictions against true values"""
        if true_values.size == 0 or predictions.size == 0:
            raise ValueError("Cannot plot empty predictions or true values.")
        
        return self.evaluator.plot_predictions(
            model_name, true_values, predictions, target_cols, dates
        )