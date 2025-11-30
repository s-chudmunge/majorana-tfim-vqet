"""Configuration loader for TFIM simulations."""
import yaml
import os


def load_config(config_path=None):
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default config.

    Returns
    -------
    config : dict
        Configuration parameters
    """
    if config_path is None:
        # Use default config
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, '..', 'config', 'default_params.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_param(config, *keys, default=None):
    """
    Get nested parameter from config dict.

    Example: get_param(config, 'system', 'n_sites') returns config['system']['n_sites']
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
