"""Model architectures for sequence labeling."""

from .blstm_tagger import (
    BLSTMTagger,
    create_model
)

__all__ = [
    'BLSTMTagger',
    'create_model'
]
