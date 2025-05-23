# utils/config_loader.py

import yaml

def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to config.yml

    Returns:
        dict: Parsed configuration as dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
