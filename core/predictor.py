import pandas as pd
import numpy as np
import torch
import logging
import re
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

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

@dataclass
class PredictionConfig:
    """Configuration for prediction"""
    temperature: float = 0.1
    max_new_tokens: int = 50
    num_return_sequences: int = 1
    batch_size: int = 16
    use_tqdm: bool = True

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
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or PredictionConfig()

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
        input_df: pd.DataFrame,
        num_steps: int,
        target_cols: List[str],
        sequence_length: int = 60
    ) -> pd.DataFrame:
        """
        Generate predictions for time series data
        
        Args:
            input_df: Input dataframe with time series data
            num_steps: Number of steps to predict into the future
            target_cols: List of target columns to predict
            sequence_length: Length of input sequences
            
        Returns:
            DataFrame with predictions
            
        Raises:
            ModelNotSetError: If model or tokenizer is not set
            ValueError: If input validation fails
            PredictionError: If prediction generation fails
        """
        try:
            self._validate_prediction_inputs(input_df, num_steps, target_cols, sequence_length)
            
            logger.info(f"Generating predictions for {num_steps} steps ahead")
            input_sequence = input_df.iloc[-sequence_length:].copy()
            predictions = []
            current_sequence = input_sequence.values

            iterator = tqdm(range(num_steps), desc="Predicting") if self.config.use_tqdm else range(num_steps)
            
            for step in iterator:
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

            pred_df = self._create_prediction_dataframe(input_df, predictions, num_steps, target_cols)
            logger.info(f"Generated predictions dataframe with shape {pred_df.shape}")
            return pred_df

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError("Failed to generate predictions") from e

    def _validate_prediction_inputs(
        self,
        input_df: pd.DataFrame,
        num_steps: int,
        target_cols: List[str],
        sequence_length: int
    ) -> None:
        """Validate prediction inputs"""
        if self.model is None or self.tokenizer is None:
            raise ModelNotSetError("Model and tokenizer must be set before prediction")

        if input_df.empty:
            raise ValueError("Input DataFrame is empty")

        if not all(col in input_df.columns for col in target_cols):
            missing_cols = [col for col in target_cols if col not in input_df.columns]
            raise ValueError(f"Missing target columns: {missing_cols}")

        if num_steps < 1:
            raise ValueError("Number of prediction steps must be positive")

        if sequence_length < 1:
            raise ValueError("Sequence length must be positive")

        if len(input_df) < sequence_length:
            raise ValueError(f"Input dataframe must have at least {sequence_length} rows")

    def _format_input_for_model(self, sequence: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Format the input sequence for the model
        
        Args:
            sequence: Input sequence
            
        Returns:
            Formatted input for the model
        """
        sequence_str = str(sequence.tolist())
        prompt = f"Given the following time series data: {sequence_str}\nPredict the next values:"

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
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
            output = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                num_return_sequences=self.config.num_return_sequences
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

    def _create_prediction_dataframe(
        self,
        input_df: pd.DataFrame,
        predictions: List[np.ndarray],
        num_steps: int,
        target_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create DataFrame from predictions
        
        Args:
            input_df: Original input DataFrame
            predictions: List of predictions
            num_steps: Number of prediction steps
            target_cols: Target column names
            
        Returns:
            DataFrame with predictions
        """
        try:
            last_date = input_df.index[-1] if isinstance(input_df.index, pd.DatetimeIndex) else pd.Timestamp.now()
            pred_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_steps)
            return pd.DataFrame(predictions, index=pred_dates, columns=target_cols)
        except Exception as e:
            logger.error(f"Error creating prediction DataFrame: {str(e)}")
            raise PredictionError("Failed to create prediction DataFrame") from e