import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.loaded_models = {}

    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """
        Load a pre-trained model and tokenizer
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading model from HuggingFace: {model_name}")
            
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with CPU device
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",  # Force CPU usage
                torch_dtype=torch.float32  # Use float32 for better compatibility
            )
            
            # Set model to eval mode
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise ModelLoadError(f"Failed to load model {model_name}") from e

    def load_model_from_local(self, model_path, **kwargs):
        """
        Load a model from local directory
        
        Args:
            model_path (str): Path to the model on local directory
            **kwargs: Additional arguments for model loading
            
        Returns:
            tuple: (model, tokenizer) for the loaded model
        """
        model_key = f"local_{model_path}"

        if model_key in self.loaded_models:
            logger.info(f"Model {model_path} already loaded, reusing instance")
            return self.loaded_models[model_key]

        try:
            logger.info(f"Loading model from local path: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Local model path not found: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                **kwargs
            )
            model.to("cpu")

            self.loaded_models[model_key] = (model, tokenizer)
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
            raise RuntimeError(f"Failed to load model: {model_path}") from e