import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_path='./models/registry.json'):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self):
        """Load the model registry from a JSON file."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as file:
                    return json.load(file)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading model registry: {e}")
        return {'models': {}, 'benchmarks': {}}

    def _save_registry(self):
        """Save the current registry to a file."""
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, 'w') as file:
                json.dump(self.registry, file, indent=4)
            logger.info(f"Successfully saved model registry to {self.registry_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving registry to {self.registry_path}: {str(e)}")
            return False

    def register_model(self, model_name, model_path, model_type="hf"):
        """
        Register a new model in the registry.
        
        Args:
            model_name (str): Name of the model.
            model_path (str): Path where the model is stored.
            model_type (str): Type of model (default: "hf").
        """
        if model_name in self.registry.get('models', {}):
            logger.warning(f"Model '{model_name}' is already registered.")

        self.registry['models'][model_name] = {
            'path': model_path,
            'model_type': model_type,
            'registered_at': datetime.now().isoformat(),
            'metrics': {}
        }

        logger.info(f"Registered model: {model_name}")
        self.save()

    def get_model_info(self, model_name):
        """Retrieve details of a registered model."""
        model_info = self.registry.get('models', {}).get(model_name)
        if not model_info:
            logger.warning(f"Model '{model_name}' not found in registry.")
        return model_info

    def register_benchmark(self, benchmark_name, models, dataset, metadata=None):
        """
        Register a benchmark in the registry.

        Args:
            benchmark_name (str): Name of the benchmark.
            models (list): List of model names used in the benchmark.
            dataset (str): Dataset used for benchmarking.
            metadata (dict, optional): Additional metadata.
        """
        if 'benchmarks' not in self.registry:
            self.registry['benchmarks'] = {}

        if metadata is None:
            metadata = {}

        self.registry['benchmarks'][benchmark_name] = {
            'models': models,
            'dataset': dataset,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata
        }

        logger.info(f"Registered benchmark: {benchmark_name}")
        self.save()

    def get_benchmark(self, benchmark_name):
        """Retrieve details of a registered benchmark."""
        return self.registry.get('benchmarks', {}).get(benchmark_name, None)

    def list_models(self):
        """Get a list of all registered models."""
        return list(self.registry.get('models', {}).keys())

    def list_benchmarks(self):
        """Get a list of all registered benchmarks."""
        return list(self.registry.get('benchmarks', {}).keys())

    def update_model_metrics(self, model_name, metrics):
        """
        Update the metrics for a registered model.
        
        Args:
            model_name (str): Name of the model.
            metrics (dict): Dictionary of new metrics.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if model_name not in self.registry.get('models', {}):
            logger.warning(f"Model '{model_name}' not found in registry.")
            return False

        self.registry['models'][model_name].setdefault('metrics', {}).update(metrics)

        logger.info(f"Updated metrics for model: {model_name}")
        return self.save()