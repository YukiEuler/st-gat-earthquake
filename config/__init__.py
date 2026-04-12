"""Configuration module for ST-GAT earthquake prediction.


Exports:
	- CONFIG: Main configuration dictionary
	- DEVICE: PyTorch device (cuda or cpu)
	- print_config(): Function to print configuration
	- CONFIG_EVENT: Event-based configuration dictionary
	- print_event_config(): Function to print event configuration
"""

from .base import CONFIG, DEVICE, print_config
from .config_event import CONFIG_EVENT, print_event_config

__all__ = ['CONFIG', 'DEVICE', 'print_config', 'CONFIG_EVENT', 'print_event_config']
