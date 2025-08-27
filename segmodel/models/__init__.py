"""Model architectures for sequence labeling."""

from .blstm_tagger import (
    BLSTMTagger,
    create_model
)
from .cnn_tagger import CNNTagger

__all__ = [
    'BLSTMTagger',
    'CNNTagger',
    'create_model'
]
