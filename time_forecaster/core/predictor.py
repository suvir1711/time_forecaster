import pandas as pd
import numpy as np
import torch
import logging
import re
import time
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
from time_forecaster.core.evaluator import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictorError(Exception):
    """Base exception for predictor errors"""
    pass

class ModelNotSetError(PredictorError):
    """Exception raised when model is not set"""
    pass

class PredictionError(PredictorError):
    """Exception raised when prediction fails"""
    pass

class PredictionTimeoutError(PredictorError):
    """Exception raised when prediction times out"""
    pass

@dataclass
class PredictionConfig:
    """Configuration for prediction"""
    temperature: float = 0.1
    max_new_tokens: int = 50
    num_return_sequences: int = 1
    batch_size: int = 16
    use_tqdm: bool = True
    timeout_seconds: int = 30  # Default timeout of 30 seconds

class TimeSeriesPredictor:
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        config: Optional[PredictionConfig] = None
    ):
        """
        Initialize predictor
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer for the model
            device: Device to run predictions on
            config: Prediction configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"  # Force CPU usage
        self.config = config or PredictionConfig()
        self.evaluator = ModelEvaluator()

    def set_model(self, model: Any, tokenizer: Any) -> None:
        """
        Set the model and tokenizer for prediction
        
        Args:
            model: Pre-trained model
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        logger.info("Model and tokenizer set successfully")

    def predict(
        self,
        input_data: Union[pd.DataFrame, np.ndarray],
        num_steps: int,
        target_cols: List[str],
        sequence_length: int = 60
    ) -> pd.DataFrame:
        """
        Generate predictions for time series data
        
        Args:
            input_data: Input data as DataFrame or NumPy array
            num_steps: Number of steps to predict into the future
            target_cols: List of target columns to predict
            sequence_length: Length of input sequences
            
        Returns:
            DataFrame with predictions
            
        Raises:
            ModelNotSetError: If model or tokenizer is not set
            ValueError: If input validation fails
            PredictionError: If prediction generation fails
            PredictionTimeoutError: If prediction takes too long
        """
        try:
            self._validate_prediction_inputs(input_data, num_steps, target_cols, sequence_length)
            
            logger.info(f"Generating predictions for {num_steps} steps ahead")
            
            # Get the last sequence_length rows
            if isinstance(input_data, pd.DataFrame):
                input_sequence = input_data.iloc[-sequence_length:].values
                last_date = input_data.index[-1] if isinstance(input_data.index, pd.DatetimeIndex) else pd.Timestamp.now()
            else:
                # If input_data is already a sequence, use the last sequence
                if len(input_data.shape) == 3:  # (n_samples, sequence_length, n_features)
                    input_sequence = input_data[-1]  # Take the last sequence
                else:
                    input_sequence = input_data[-sequence_length:]
                last_date = pd.Timestamp.now()
            
            predictions = []
            current_sequence = input_sequence.copy()
            start_time = time.time()

            iterator = tqdm(range(num_steps), desc="Predicting") if self.config.use_tqdm else range(num_steps)
            
            for step in iterator:
                # Check for timeout
                if time.time() - start_time > self.config.timeout_seconds:
                    raise PredictionTimeoutError(
                        f"Prediction timed out after {self.config.timeout_seconds} seconds"
                    )
                
                try:
                    formatted_input = self._format_input_for_model(current_sequence)
                    with torch.no_grad():
                        output = self._get_model_prediction(formatted_input)
                    pred_values = self._extract_prediction_values(output, target_cols)
                    predictions.append(pred_values)

                    # Roll sequence forward
                    current_sequence = np.roll(current_sequence, -1, axis=0)
                    current_sequence[-1, :len(target_cols)] = pred_values

                except Exception as e:
                    logger.error(f"Error during prediction step {step}: {str(e)}")
                    raise PredictionError(f"Failed at prediction step {step}") from e

            # Create prediction DataFrame with dates
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_steps)
            pred_df = pd.DataFrame(predictions, index=pred_dates, columns=target_cols)
            
            logger.info(f"Generated predictions dataframe with shape {pred_df.shape}")
            return pred_df

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError("Failed to generate predictions") from e

    def _validate_prediction_inputs(
        self,
        input_data: Union[pd.DataFrame, np.ndarray],
        num_steps: int,
        target_cols: List[str],
        sequence_length: int
    ) -> None:
        """Validate prediction inputs"""
        if self.model is None or self.tokenizer is None:
            raise ModelNotSetError("Model and tokenizer must be set before prediction")

        # Check if input is empty
        if isinstance(input_data, pd.DataFrame):
            if input_data.empty:
                raise ValueError("Input DataFrame is empty")
            if not all(col in input_data.columns for col in target_cols):
                missing_cols = [col for col in target_cols if col not in input_data.columns]
                raise ValueError(f"Missing target columns: {missing_cols}")
        else:
            if input_data.size == 0:
                raise ValueError("Input array is empty")
            if input_data.shape[1] < len(target_cols):
                raise ValueError(f"Input array has fewer columns than target columns")

        if num_steps < 1:
            raise ValueError("Number of prediction steps must be positive")

        if sequence_length < 1:
            raise ValueError("Sequence length must be positive")

        input_length = len(input_data) if isinstance(input_data, pd.DataFrame) else input_data.shape[0]
        if input_length < sequence_length:
            raise ValueError(f"Input must have at least {sequence_length} rows")

    def _format_input_for_model(self, sequence: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Format the input sequence for the model
        
        Args:
            sequence: Input sequence
            
        Returns:
            Formatted input for the model
        """
        try:
            # Truncate sequence to last few rows to keep input size manageable
            max_rows = 10  # Only use last 10 rows to keep input size small
            sequence = sequence[-max_rows:]
            
            # Format sequence with fewer decimal places
            sequence_str = np.array2string(sequence, precision=2, separator=',', suppress_small=True)
            prompt = f"Given the following time series data: {sequence_str}\nPredict the next values:"

            # Tokenize and move to correct device
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Maximum context length for GPT models
                padding=True,
                add_special_tokens=True
            )
            
            # Move inputs to the same device as the model
            return {k: v.to(self.device) for k, v in inputs.items()}
            
        except Exception as e:
            logger.error(f"Error formatting input: {str(e)}")
            raise PredictionError("Failed to format input for model") from e

    def _get_model_prediction(self, inputs: Dict[str, torch.Tensor]) -> str:
        """
        Generate prediction from the model
        
        Args:
            inputs: Formatted model inputs
            
        Returns:
            Model output as string
        """
        try:
            # Ensure model is in eval mode
            self.model.eval()
            
            # Generate with proper configuration
            output = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,  # Enable sampling
                temperature=self.config.temperature,
                num_return_sequences=self.config.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise PredictionError("Failed to generate model prediction") from e

    def _extract_prediction_values(self, output: str, target_cols: List[str]) -> np.ndarray:
        """
        Extract predicted values from model output
        
        Args:
            output: Model output string
            target_cols: Target column names
            
        Returns:
            Array of predicted values
        """
        try:
            prediction_part = output.split("Predict the next values:")[1].strip()
            values = re.findall(r"[-+]?\d*\.\d+|\d+", prediction_part)
            values = [float(v) for v in values][:len(target_cols)]

            # Ensure correct length
            while len(values) < len(target_cols):
                values.append(0.0)

            return np.array(values)
        except Exception as e:
            logger.error(f"Error extracting prediction values: {str(e)}")
            logger.warning("Returning zeros as fallback")
            return np.zeros(len(target_cols))

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
        return self.evaluator.evaluate(y_true, y_pred, target_cols)