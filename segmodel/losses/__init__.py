"""Loss functions and training metrics."""

from .cross_entropy import (
    CrossEntropyWithLabelSmoothing,
    batch_guardrails,
    sequence_f1_score
)

from .boundary_aware_cross_entropy import (
    BoundaryAwareCrossEntropy,
    create_loss_function as create_boundary_aware_loss_function,
)

# Maintain backward compatibility
create_loss_function = create_boundary_aware_loss_function

__all__ = [
    'CrossEntropyWithLabelSmoothing',  # Legacy support
    'BoundaryAwareCrossEntropy',       # New roadmap-based loss
    'create_loss_function',            # Factory (now points to boundary-aware)
    'create_boundary_aware_loss_function',  # Explicit new loss factory
    'batch_guardrails',
    'sequence_f1_score'
]
