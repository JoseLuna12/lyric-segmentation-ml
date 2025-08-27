"""Evaluation metrics for lyrics segmentation."""

from .boundary_metrics import (
    compute_boundary_metrics,
    compute_segment_metrics,
    compute_transition_metrics,
    format_boundary_metrics_report,
    BoundaryMetrics,
    SegmentMetrics,
    TransitionMetrics
)
from .segmentation_metrics import (
    compute_segmentation_metrics,
    compute_window_diff,
    compute_pk_metric,
    format_segmentation_metrics_report,
    SegmentationMetrics
)
from .calibration_metrics import (
    ece,
    reliability_curve,
    ReliabilityDiagram,
    plot_reliability_diagram
)

__all__ = [
    'compute_boundary_metrics',
    'compute_segment_metrics',
    'compute_transition_metrics',
    'format_boundary_metrics_report',
    'BoundaryMetrics',
    'SegmentMetrics',
    'TransitionMetrics',
    'compute_segmentation_metrics',
    'compute_window_diff',
    'compute_pk_metric',
    'format_segmentation_metrics_report',
    'SegmentationMetrics',
    # Calibration metrics
    'ece',
    'reliability_curve',
    'ReliabilityDiagram',
    'plot_reliability_diagram'
]
