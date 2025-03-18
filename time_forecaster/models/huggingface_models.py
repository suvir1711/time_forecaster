import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re
import pandas as pd

logger = logging.getLogger(__name__)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesHFModel:
    """Base class for time series forecasting using Hugging Face models."""
    
    def __init__(self, model_name=None, device=None, cache_dir=None):
        self.model_name = model_name
        self.device = device or get_device()
        self.model = None
        self.tokenizer = None
        if model_name:
            self.load_model(model_name, cache_dir)
    
    def load_model(self, model_name, cache_dir=None, **kwargs):
        try:
            logging.info(f"Loading model and tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                **kwargs
            ).to(self.device)
            logging.info(f"Successfully loaded model: {model_name}")
            return self.model, self.tokenizer
        except Exception as e:
            logging.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def format_input(self, time_series_data, target_cols=None):
        """Format time series data into text format for transformer models."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Load a model first.")

        if isinstance(time_series_data, pd.DataFrame):
            target_cols = target_cols or time_series_data.columns.tolist()
            data_str = "\n".join(
                [" ".join([f"{col}: {row[col]:.4f}" for col in target_cols]) for _, row in time_series_data.iterrows()]
            )
        else:
            data_str = " ".join([str(x) for x in time_series_data.flatten()])
        
        prompt = f"Predict the next values given the following time series data:\n{data_str}\nNext values:"
        inputs = self.tokenizer( 
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.device)
        return inputs
    
    def predict(self, time_series_data, num_steps=1, target_cols=None, temperature=0.1, top_k=50, top_p=0.9):
        """Generate predictions for time series data."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before making predictions.")
        
        predictions = []
        current_data = time_series_data.copy()
        
        for _ in range(num_steps):
            inputs = self.format_input(current_data, target_cols)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=target_cols,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.9,
                    num_return_sequences=1
                )
                
            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            prediction = self._extract_prediction(output_text, len(target_cols) if target_cols else 1)
            
            predictions.append(prediction)
            
            if isinstance(current_data, pd.DataFrame):
                current_data = pd.concat(
                    [current_data.iloc[1:], pd.DataFrame([prediction], columns=target_cols)],
                    ignore_index=True
                )
            else:
                current_data = torch.cat((current_data[1:], torch.tensor(prediction).unsqueeze(0)), dim=0)
        
        return torch.stack(predictions).cpu().numpy()
    
    def _extract_prediction(self, output_text, num_values):
        """Extract numeric predictions from the model output."""
        match = re.findall(r"[-+]?\d*\.\d+|\d+", output_text)
        values = [float(v) for v in match[:num_values]]
        while len(values) < num_values:
            values.append(0.0)
        return np.array(values)

class TimeSeriesTimesFM(TimeSeriesHFModel):
    """Time series forecasting using Google's TimesFM model."""
    
    def __init__(self, model_name="google/timesfm", cache_dir=None):
        super().__init__()
        self.load_model(model_name, cache_dir, trust_remote_code=True)
    
class MiraiTimeSeriesModel(TimeSeriesHFModel):
    """Time series forecasting using Salesforce's Mirai model."""
    
    def __init__(self, model_name="Salesforce/mirai", cache_dir=None):
        super().__init__()
        self.load_model(model_name, cache_dir)
    
# Utility function to determine device
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Example usage
if __name__ == "__main__":
    model = MiraiTimeSeriesModel()  # Load Salesforce Mirai model
    example_data = pd.DataFrame({
        "Time": [1, 2, 3, 4, 5],
        "Value": [100.5, 102.3, 104.2, 110.5, 112.8]
    })
    pred = model.predict(example_data, num_steps=2, target_cols=["value"])
    print("Predicted values:", pred)
