import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluatorError(Exception):
    """Base exception for evaluator errors"""
    pass

class ValidationError(EvaluatorError):
    """Exception raised for validation failures"""
    pass

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    rmse: float
    mae: float
    r2: float
    model_name: str
    target_col: str

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    plot_style: str = "default"
    figure_size: Tuple[int, int] = (12, 6)
    confidence_interval: float = 0.95
    metrics: List[str] = field(default_factory=lambda: ["RMSE", "MAE", "R2"])

class ModelEvaluator:
    """Evaluator for time series models"""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluator
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self.evaluation_results: Dict[str, Dict[str, EvaluationMetrics]] = {}
        plt.style.use(self.config.plot_style)

    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.DataFrame],
        y_pred: Union[np.ndarray, pd.DataFrame],
        target_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model predictions
        
        Args:
            y_true: True values
            y_pred: Predicted values
            target_cols: List of target column names
            
        Returns:
            Dictionary of metrics for each target column
        """
        try:
            # Convert inputs to numpy arrays if they aren't already
            if isinstance(y_true, pd.DataFrame):
                y_true = y_true.values
            if isinstance(y_pred, pd.DataFrame):
                y_pred = y_pred.values

            # Ensure inputs have the same shape
            if y_true.shape != y_pred.shape:
                raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}")

            metrics = {}
            for i, col in enumerate(target_cols):
                metrics[col] = {
                    'RMSE': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                    'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                    'R2': r2_score(y_true[:, i], y_pred[:, i])
                }

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def evaluate_model(
        self,
        model_name: str,
        true_values: np.ndarray,
        predictions: np.ndarray,
        target_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model predictions
        
        Args:
            model_name: Name of the model being evaluated
            true_values: Ground truth values
            predictions: Predicted values
            target_cols: Names of target columns
            
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            ValidationError: If inputs are invalid
        """
        try:
            self._validate_evaluation_inputs(true_values, predictions, target_cols)
            
            results = {}
            for i, col in enumerate(target_cols):
                true_col = true_values[:, i]
                pred_col = predictions[:, i]
                
                metrics = self._calculate_metrics(true_col, pred_col)
                self.evaluation_results.setdefault(model_name, {})[col] = EvaluationMetrics(
                    rmse=metrics['RMSE'],
                    mae=metrics['MAE'],
                    r2=metrics['R2'],
                    model_name=model_name,
                    target_col=col
                )
                results[col] = metrics
                
            logger.info(f"Evaluation complete for model {model_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise EvaluatorError(f"Evaluation failed: {str(e)}") from e

    def _validate_evaluation_inputs(
        self,
        true_values: np.ndarray,
        predictions: np.ndarray,
        target_cols: List[str]
    ) -> None:
        """Validate evaluation inputs"""
        if true_values.shape != predictions.shape:
            raise ValidationError(
                f"Shape mismatch: true_values {true_values.shape}, "
                f"predictions {predictions.shape}"
            )
            
        if true_values.shape[1] != len(target_cols):
            raise ValidationError(
                f"Number of columns ({true_values.shape[1]}) doesn't match "
                f"number of target columns ({len(target_cols)})"
            )

    def _calculate_metrics(
        self,
        true_values: np.ndarray,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            true_values: Ground truth values
            predictions: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        return {
            'RMSE': np.sqrt(mean_squared_error(true_values, predictions)),
            'MAE': mean_absolute_error(true_values, predictions),
            'R2': r2_score(true_values, predictions)
        }

    def compare_models(
        self,
        top_n: Optional[int] = None,
        metric: str = 'RMSE'
    ) -> pd.DataFrame:
        """
        Compare all evaluated models
        
        Args:
            top_n: Number of top models to return
            metric: Metric to sort by
            
        Returns:
            DataFrame with model comparisons
        """
        if not self.evaluation_results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame(columns=["Model", "Target", "RMSE", "MAE", "R2"])

        try:
            comparison_data = []
            for model_name, targets in self.evaluation_results.items():
                for target_col, metrics in targets.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Target': target_col,
                        'RMSE': metrics.rmse,
                        'MAE': metrics.mae,
                        'R2': metrics.r2
                    })

            df = pd.DataFrame(comparison_data)
            df = df.sort_values(metric)
            
            if top_n:
                df = df.head(top_n)
                
            return df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise EvaluatorError("Failed to compare models") from e

    def plot_predictions(
        self,
        model_name: str,
        true_values: np.ndarray,
        predictions: np.ndarray,
        target_cols: List[str],
        dates: Optional[pd.DatetimeIndex] = None
    ) -> plt.Figure:
        """
        Plot predictions against true values
        
        Args:
            model_name: Name of the model
            true_values: Ground truth values
            predictions: Predicted values
            target_cols: Names of target columns
            dates: Optional datetime index for x-axis
            
        Returns:
            Matplotlib figure
        """
        try:
            self._validate_evaluation_inputs(true_values, predictions, target_cols)
            
            n_cols = len(target_cols)
            fig, axes = plt.subplots(
                n_cols, 1,
                figsize=self.config.figure_size,
                sharex=True
            )
            
            if n_cols == 1:
                axes = [axes]
            
            x = dates if dates is not None else np.arange(len(true_values))
            
            for i, (ax, col) in enumerate(zip(axes, target_cols)):
                ax.plot(x, true_values[:, i], label='True', color='blue', alpha=0.6)
                ax.plot(x, predictions[:, i], label='Predicted', color='red', alpha=0.6)
                
                # Add confidence interval if configured
                if self.config.confidence_interval:
                    self._add_confidence_interval(ax, x, true_values[:, i], predictions[:, i])
                
                ax.set_title(f'{col} - {model_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            raise EvaluatorError("Failed to plot predictions") from e

    def _add_confidence_interval(
        self,
        ax: plt.Axes,
        x: Union[np.ndarray, pd.DatetimeIndex],
        true_values: np.ndarray,
        predictions: np.ndarray
    ) -> None:
        """Add confidence interval to plot"""
        residuals = predictions - true_values
        std = np.std(residuals)
        ci = std * 1.96  # 95% confidence interval
        
        ax.fill_between(
            x,
            predictions - ci,
            predictions + ci,
            color='red',
            alpha=0.1,
            label=f'{int(self.config.confidence_interval*100)}% CI'
        )

    def get_metric_summary(self, model_name: str) -> pd.DataFrame:
        """
        Get summary of metrics for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with metric summary
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for model {model_name}")
            
        summary_data = []
        for target_col, metrics in self.evaluation_results[model_name].items():
            summary_data.append({
                'Target': target_col,
                'RMSE': metrics.rmse,
                'MAE': metrics.mae,
                'R2': metrics.r2
            })
            
        return pd.DataFrame(summary_data)

    def plot_metric_comparison(
        self,
        metric: str = 'RMSE',
        top_n: Optional[int] = None
    ) -> plt.Figure:
        """
        Plot comparison of models based on a metric
        
        Args:
            metric: Metric to compare
            top_n: Number of top models to include
            
        Returns:
            Matplotlib figure
        """
        try:
            comparison_df = self.compare_models(top_n=top_n, metric=metric)
            
            plt.figure(figsize=self.config.figure_size)
            sns.barplot(
                data=comparison_df,
                x='Model',
                y=metric,
                hue='Target'
            )
            plt.xticks(rotation=45)
            plt.title(f'Model Comparison - {metric}')
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting metric comparison: {str(e)}")
            raise EvaluatorError("Failed to plot metric comparison") from e