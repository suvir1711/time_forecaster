import os
import yaml
import logging

class ConfigLoader:
    def __init__(self, config_path=None):
        """
        Initialize ConfigLoader with default config and optional config file.
        
        Args:
            config_path (str, optional): Path to the configuration YAML file.
        """
        self.config_path = config_path
        self.config = self._load_default_config()

        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)

    def _load_default_config(self):
        """Load default configuration values."""
        return {
            'config_path': self.config_path,
            'data': {
                'config_dir': './data/cache',
                'default_models': ['Open', 'High', 'Close'],
                'train_test_split': 0.8,
                'default_features': ['Close', 'Volume', 'RSI', 'MACD']
            },
            'models': {
                'cache_dir': './models/cache',
                'default_models': {
                    'gpt2-stock': 'distilgpt2',
                    'falcon-stock': 'falcon-7b'
                }
            },
            'parameters': {
                'epochs': 10,
                'learning_rate': 0.0001,
                'batch_size': 32
            },
            'visualization': {
                'figsize': (12, 8),
                'style': 'seaborn'
            }
        }

    def _load_from_file(self, config_path):
        """Load and update configuration from a YAML file."""
        try:
            with open(config_path, 'r') as file:
                file_config = yaml.safe_load(file)
                if isinstance(file_config, dict):
                    self._update_nested_dict(self.config, file_config)
                else:
                    raise ValueError(f"Invalid format in config file: {config_path}")

        except Exception as e:
            logging.error(f"Error loading configuration from {config_path}: {e}")

    def _update_nested_dict(self, d, u):
        """Recursively update a nested dictionary with values from another dictionary."""
        if not isinstance(u, dict): 
            return

        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v

    def save_config(self, config_path=None):
        """Save the current configuration to a YAML file."""
        if config_path:
            self.config_path = config_path  # Update path if provided

        if not self.config_path:
            logging.error("No config path specified. Cannot save configuration.")
            return False

        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as file:
                yaml.safe_dump(self.config, file, default_flow_style=False)
            return True

        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return False

    def get(self, section=None):
        """Retrieve a specific section or the full configuration."""
        if section is None:
            return self.config
        return self.config.get(section, {})

    def set(self, section, key, value):
        """Update or add a key-value pair to a section."""
        if section not in self.config:
            self.config[section] = {}
        if not isinstance(self.config[section], dict):
            logging.error(f"Cannot set key in {section}, not a dictionary")
            return False

        self.config[section][key] = value
        return True