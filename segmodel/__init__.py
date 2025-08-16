"""
Segmentation model package for verse/chorus classification.

A clean, modular implementation with anti-collapse features.
"""

__version__ = "1.0.0"
__author__ = "BLSTM Baseline Implementation"

# Main components
from . import data
from . import features  
from . import models
from . import losses
from . import train
from . import utils

__all__ = [
    'data',
    'features',
    'models', 
    'losses',
    'train',
    'utils'
]
