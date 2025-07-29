"""
Configuration loading utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Handles loading and parsing of YAML configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_sections = ['model', 'training', 'logging']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model config
        model_config = config['model']
        if 'loss_type' not in model_config:
            raise ValueError("Missing 'loss_type' in model configuration")
        if model_config['loss_type'] not in ['dense', 'global', 'dense_global']:
            raise ValueError(f"Invalid loss_type: {model_config['loss_type']}")
        
        # Validate training config
        training_config = config['training']
        required_training_fields = ['batch_size', 'learning_rate', 'num_epochs']
        for field in required_training_fields:
            if field not in training_config:
                raise ValueError(f"Missing '{field}' in training configuration")