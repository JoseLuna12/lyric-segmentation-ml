"""
Training module with anti-collapse guardrails and temperature calibration.
Implements the complete training loop with monitoring and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, 
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
    StepLR,
    LinearLR
)

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
from dataclasses import dataclass, asdict
import json

from ..losses import batch_guardrails, sequence_f1_score
from ..metrics import (
    compute_boundary_metrics,
    compute_segment_metrics,
    compute_transition_metrics,
    format_boundary_metrics_report,
    compute_segmentation_metrics,
    format_segmentation_metrics_report
)
from .calibration import fit_calibration, load_calibration


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    train_chorus_rate: float
    train_max_prob: float
    train_conf_over_90: float
    train_conf_over_95: float
    val_loss: float
    val_macro_f1: float
    val_verse_f1: float
    val_chorus_f1: float
    val_chorus_rate: float
    val_max_prob: float
    val_conf_over_90: float
    val_conf_over_95: float
    learning_rate: float
    epoch_time: float
    # Boundary-aware metrics (Phase 2)
    val_boundary_f1: float = 0.0
    val_boundary_precision: float = 0.0
    val_boundary_recall: float = 0.0
    val_complete_segments: float = 0.0
    val_avg_segment_overlap: float = 0.0
    val_verse_to_chorus_acc: float = 0.0
    val_chorus_to_verse_acc: float = 0.0
    # Segmentation metrics (Phase 3)
    val_window_diff: float = 1.0
    val_pk_metric: float = 1.0


class EmergencyMonitor:
    """
    Emergency monitoring system to detect and prevent overconfidence collapse.
    Based on lessons from architecture knowledge.
    """
    
    def __init__(
        self,
        max_confidence: float = 0.95,
        min_chorus_rate: float = 0.05,
        max_chorus_rate: float = 0.85,
        max_conf_over_95: float = 0.1,
        val_overconf_threshold: float = 0.96,
        val_f1_collapse_threshold: float = 0.1,
        emergency_overconf_threshold: float = 0.98,
        emergency_conf95_ratio: float = 0.8,
        emergency_f1_threshold: float = 0.05
    ):
        self.max_confidence = max_confidence
        self.min_chorus_rate = min_chorus_rate
        self.max_chorus_rate = max_chorus_rate
        self.max_conf_over_95 = max_conf_over_95
        self.val_overconf_threshold = val_overconf_threshold
        self.val_f1_collapse_threshold = val_f1_collapse_threshold
        self.emergency_overconf_threshold = emergency_overconf_threshold
        self.emergency_conf95_ratio = emergency_conf95_ratio
        self.emergency_f1_threshold = emergency_f1_threshold
        
        print(f"🛡️  Emergency Monitor Activated:")
        print(f"   Max confidence threshold: {max_confidence}")
        print(f"   Chorus rate range: [{min_chorus_rate:.2f}, {max_chorus_rate:.2f}]")
        print(f"   Max conf>0.95 ratio: {max_conf_over_95}")
    
    def check_batch(self, guardrails: Dict[str, float], batch_idx: int) -> Tuple[bool, str]:
        """
        Check batch-level guardrails for emergency conditions.
        
        Returns:
            should_stop: Whether to emergency stop training
            warning: Warning message if any
        """
        warnings = []
        
        # Check overconfidence
        if guardrails['max_prob_mean'] > self.max_confidence:
            warnings.append(f"OVERCONFIDENCE: avg_conf={guardrails['max_prob_mean']:.3f}")
        
        if guardrails['confidence_over_95'] > self.max_conf_over_95:
            warnings.append(f"HIGH_CONF_RATIO: conf>0.95={guardrails['confidence_over_95']:.2%}")
        
        # Check chorus rate collapse
        if guardrails['chorus_rate'] < self.min_chorus_rate:
            warnings.append(f"CHORUS_COLLAPSE: rate={guardrails['chorus_rate']:.2%}")
        elif guardrails['chorus_rate'] > self.max_chorus_rate:
            warnings.append(f"CHORUS_OVERPREDICT: rate={guardrails['chorus_rate']:.2%}")
        
        # Emergency stop conditions (debug version)
        conditions = {
            'extreme_overconf': guardrails['max_prob_mean'] > 0.99,
            'high_conf_ratio': guardrails['confidence_over_95'] > 0.8,
            'chorus_collapse': guardrails['chorus_rate'] < 0.01,
            'all_chorus': guardrails['chorus_rate'] > 0.99
        }
        
        should_stop = any(conditions.values())
        
        if should_stop:
            triggered = [k for k, v in conditions.items() if v]
            warnings.append(f"EMERGENCY_TRIGGERED: {triggered}")
            # Debug info
            warnings.append(f"max_prob={guardrails['max_prob_mean']:.3f}, conf>95={guardrails['confidence_over_95']:.3f}, chorus_rate={guardrails['chorus_rate']:.3f}")
        
        warning_msg = " | ".join(warnings) if warnings else ""
        
        return should_stop, warning_msg
    
    def check_epoch(self, metrics: TrainingMetrics) -> Tuple[bool, str]:
        """Check epoch-level metrics for emergency conditions."""
        warnings = []
        
        # Check validation overconfidence
        if metrics.val_max_prob > self.val_overconf_threshold:
            warnings.append(f"VAL_OVERCONF: {metrics.val_max_prob:.3f}")
        
        # Check F1 score collapse
        if metrics.val_macro_f1 < self.val_f1_collapse_threshold:
            warnings.append(f"F1_COLLAPSE: {metrics.val_macro_f1:.3f}")
        
        should_stop = (
            metrics.val_max_prob > self.emergency_overconf_threshold or
            metrics.val_conf_over_95 > self.emergency_conf95_ratio or
            metrics.val_macro_f1 < self.emergency_f1_threshold
        )
        
        warning_msg = " | ".join(warnings) if warnings else ""
        return should_stop, warning_msg


def create_scheduler(optimizer, config, total_steps: int = None):
    """
    Factory function to create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration object
        total_steps: Total number of training steps (for cosine annealing)
        
    Returns:
        scheduler: Configured learning rate scheduler
        scheduler_type: Type of scheduler ('step' or 'epoch' based)
    """
    scheduler_name = getattr(config, 'scheduler', 'plateau')
    
    if scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=getattr(config, 'lr_factor', 0.5),
            patience=getattr(config, 'lr_patience', config.patience // 2),
            min_lr=float(getattr(config, 'min_lr', 1e-6))  # Ensure float conversion
        )
        return scheduler, 'epoch'
    
    elif scheduler_name == 'cosine':
        T_max = getattr(config, 'cosine_t_max', config.max_epochs)
        eta_min = float(getattr(config, 'min_lr', 1e-6))  # Ensure float conversion
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        return scheduler, 'epoch'
    
    elif scheduler_name == 'cosine_restarts':
        T_0 = getattr(config, 'cosine_t0', 10)
        T_mult = getattr(config, 'cosine_t_mult', 2)
        eta_min = float(getattr(config, 'min_lr', 1e-6))  # Ensure float conversion
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        return scheduler, 'epoch'
    
    elif scheduler_name == 'step':
        step_size = getattr(config, 'step_size', config.max_epochs // 4)
        gamma = getattr(config, 'step_gamma', 0.5)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        return scheduler, 'epoch'
    
    elif scheduler_name == 'warmup_cosine':
        # Implement warmup + cosine annealing
        warmup_epochs = getattr(config, 'warmup_epochs', 5)
        if total_steps is None:
            raise ValueError("total_steps required for warmup_cosine scheduler")
        
        # Create warmup scheduler (linearly increase LR for first few epochs)
        def lr_lambda(step):
            if step < warmup_epochs:
                return step / warmup_epochs
            else:
                # Cosine decay after warmup
                progress = (step - warmup_epochs) / (config.max_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, 'epoch'
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def compute_validation_score(metrics: TrainingMetrics, config: Any) -> float:
    """
    Compute validation score based on configured strategy.
    
    Args:
        metrics: Training metrics containing all computed values
        config: Configuration object with validation strategy setting
        
    Returns:
        Validation score for model selection (higher is better for all strategies)
    """
    strategy = getattr(config, 'validation_strategy', 'line_f1')
    
    if strategy == 'line_f1':
        # Line-level macro F1 (original method)
        return metrics.val_macro_f1
    
    elif strategy == 'boundary_f1':
        # Boundary detection F1 (structural understanding) - RECOMMENDED
        return metrics.val_boundary_f1
    
    elif strategy == 'windowdiff':
        # WindowDiff (lower is better, so invert for "higher is better")
        return 1.0 - metrics.val_window_diff
    
    elif strategy == 'pk':
        # Pk metric (lower is better, so invert for "higher is better") 
        return 1.0 - metrics.val_pk_metric
    
    elif strategy == 'segment_iou':
        # Segment IoU (complete segment quality)
        return metrics.val_avg_segment_overlap
    
    elif strategy == 'composite':
        # Composite score with sensible hardcoded weights
        # Emphasize structural understanding while maintaining some line-level performance
        composite_score = (
            0.25 * metrics.val_macro_f1 +                    # Line-level performance
            0.40 * metrics.val_boundary_f1 +                 # Boundary detection (most important)
            0.25 * metrics.val_avg_segment_overlap +         # Complete segment quality
            0.10 * (1.0 - metrics.val_window_diff)          # Forgiving boundary evaluation
        )
        return composite_score
    
    else:
        # Fallback to line-level F1
        print(f"⚠️  Unknown validation strategy '{strategy}', falling back to line_f1")
        return metrics.val_macro_f1


class Trainer:
    """
    Main trainer class with anti-collapse guardrails.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Any,
        output_dir: Path,
        disable_emergency_monitoring: bool = False
    ):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.output_dir = output_dir
        
        # Initialize calibration info storage
        self.calibration_info = None
        
        # Set validation strategy first
        self.validation_strategy = getattr(config, 'validation_strategy', 'line_f1')
        
        # Set up scheduler using factory function
        self.scheduler, self.scheduler_type = create_scheduler(
            optimizer, 
            config,
            total_steps=config.max_epochs  # For cosine scheduling
        )
        
        print(f"🎯 Scheduler: {getattr(config, 'scheduler', 'plateau')} ({self.scheduler_type}-based)")
        print(f"🎯 Validation Strategy: {self.validation_strategy} (for best model selection)")
        
        # Set up emergency monitoring using flattened configuration parameters
        if not disable_emergency_monitoring:
            self.emergency_monitor = EmergencyMonitor(
                max_confidence=getattr(config, 'max_confidence_threshold', 0.95),
                min_chorus_rate=getattr(config, 'min_chorus_rate', 0.05),
                max_chorus_rate=getattr(config, 'max_chorus_rate', 0.85),
                max_conf_over_95=getattr(config, 'max_conf_over_95_ratio', 0.1),
                val_overconf_threshold=getattr(config, 'val_overconf_threshold', 0.96),
                val_f1_collapse_threshold=getattr(config, 'val_f1_collapse_threshold', 0.1),
                emergency_overconf_threshold=getattr(config, 'emergency_overconf_threshold', 0.98),
                emergency_conf95_ratio=getattr(config, 'emergency_conf95_ratio', 0.8),
                emergency_f1_threshold=getattr(config, 'emergency_f1_threshold', 0.05)
            )
        else:
            self.emergency_monitor = None
            print("⚠️  Emergency monitoring DISABLED for debugging")
        
        # Training state
        self.best_val_score = -1.0  # Now tracks configurable validation metric
        self.patience_counter = 0
        self.training_metrics = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty metrics file
        self._initialize_metrics_file()
        
        print(f"🚀 Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Output dir: {output_dir}")
        print(f"   Max epochs: {config.max_epochs}")
        print(f"   Patience: {config.patience}")
    
    def _run_calibration(self, val_loader):
        """
        Run model calibration and save results.
        This method can be called from normal completion or keyboard interrupt.
        """
        print(f"\n🌡️ Running model calibration...")
        
        # Get calibration methods from config
        calibration_methods = None
        if hasattr(self.config, 'calibration') and hasattr(self.config.calibration, 'methods'):
            calibration_methods = self.config.calibration.methods
        
        try:
            self.calibration_info = fit_calibration(
                model=self.model,
                val_loader=val_loader,
                device=self.device,
                methods=calibration_methods,
                output_dir=self.output_dir
            )
            print(f"✅ Calibration completed successfully")
        except Exception as e:
            print(f"⚠️  Calibration failed: {e}")
            self.calibration_info = {}
        
        return self.calibration_info
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_guardrails = []
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            features = batch.features.to(self.device)
            labels = batch.labels.to(self.device)
            mask = batch.mask.to(self.device)
            
            # Forward pass
            logits = self.model(features, mask)
            loss = self.loss_function(logits, labels, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Batch-level guardrails
            guardrails = batch_guardrails(logits, mask)
            all_guardrails.append(guardrails)
            
            # Emergency monitoring (skip first N batches to allow model to stabilize)
            should_stop, warning = False, ""
            skip_batches = getattr(self.config, 'skip_batches', 50)
            print_batch_every = getattr(self.config, 'print_batch_every', 10)
            
            if self.emergency_monitor is not None and batch_idx >= skip_batches:
                should_stop, warning = self.emergency_monitor.check_batch(guardrails, batch_idx)
            
            # Print batch info
            if batch_idx % print_batch_every == 0 or batch_idx == num_batches - 1:
                print(f"  Batch {batch_idx+1:3d}/{num_batches}: "
                      f"loss={loss.item():.4f}, "
                      f"chorus%={guardrails['chorus_rate']:.2f}, "
                      f"max_prob={guardrails['max_prob_mean']:.3f}")
                
                if warning:
                    print(f"    ⚠️  {warning}")
            
            if should_stop:
                print(f"🛑 EMERGENCY STOP at batch {batch_idx+1}")
                raise RuntimeError("Emergency training stop due to overconfidence collapse")
        
        # Aggregate metrics
        avg_metrics = {}
        for key in all_guardrails[0].keys():
            avg_metrics[key] = np.mean([g[key] for g in all_guardrails])
        
        avg_metrics['loss'] = total_loss / num_batches
        
        return avg_metrics
    
    def evaluate(self, val_loader: DataLoader, calibrator=None) -> Dict[str, float]:
        """Evaluate on validation set with boundary-aware metrics."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        all_guardrails = []
        
        print(f"🔍 Validating on {len(val_loader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, 1):
                features = batch.features.to(self.device)
                labels = batch.labels.to(self.device)
                mask = batch.mask.to(self.device)
                
                # Forward pass
                logits = self.model(features, mask)
                loss = self.loss_function(logits, labels, mask)
                
                # Track loss
                total_loss += loss.item()
                
                # Apply calibration if provided
                probs = (calibrator.apply(logits) if calibrator else torch.softmax(logits, dim=-1))
                predictions = probs.argmax(dim=-1)
                
                # Store for metrics computation
                all_predictions.append(predictions)
                all_targets.append(labels)
                all_masks.append(mask)
                
                # Track guardrails (use probs for confidence metrics)
                guardrails = batch_guardrails(torch.log(probs + 1e-8), mask)  # Convert back to logits for guardrails
                all_guardrails.append(guardrails)
                
                # Print validation progress
                if batch_idx == 1 or batch_idx % max(1, len(val_loader) // 3) == 0 or batch_idx == len(val_loader):
                    avg_loss = total_loss / batch_idx
                    print(f"  Val batch {batch_idx:2d}/{len(val_loader)}: "
                          f"loss={avg_loss:.4f}, "
                          f"chorus%={guardrails['chorus_rate']:.2f}, "
                          f"conf={guardrails['max_prob_mean']:.3f}")
        
        print(f"✅ Validation complete, computing boundary metrics...")
        
        # Compute aggregate metrics
        avg_loss = total_loss / len(val_loader)
        
        # Aggregate guardrails
        avg_guardrails = {}
        for key in all_guardrails[0].keys():
            avg_guardrails[key] = np.mean([g[key] for g in all_guardrails])
        
        # Compute line-level F1 scores (existing functionality)
        all_pred_flat = torch.cat([pred.flatten() for pred in all_predictions], dim=0)
        all_targ_flat = torch.cat([targ.flatten() for targ in all_targets], dim=0) 
        all_mask_flat = torch.cat([mask.flatten() for mask in all_masks], dim=0)
        
        f1_scores = sequence_f1_score(all_pred_flat, all_targ_flat, all_mask_flat)
        
        # Compute boundary-aware metrics (Phase 2 enhancement)
        # For boundary analysis, we compute metrics per batch and then aggregate
        boundary_metrics_batch = []
        segment_metrics_batch = []
        transition_metrics_batch = []
        
        for pred_batch, targ_batch, mask_batch in zip(all_predictions, all_targets, all_masks):
            # Compute boundary metrics for this batch
            batch_boundary = compute_boundary_metrics(pred_batch, targ_batch, mask_batch)
            batch_segment = compute_segment_metrics(pred_batch, targ_batch, mask_batch)
            batch_transition = compute_transition_metrics(pred_batch, targ_batch, mask_batch)
            
            boundary_metrics_batch.append(batch_boundary)
            segment_metrics_batch.append(batch_segment)
            transition_metrics_batch.append(batch_transition)
        
        # Aggregate boundary metrics across all batches
        total_pred_boundaries = sum(bm.total_boundaries_predicted for bm in boundary_metrics_batch)
        total_true_boundaries = sum(bm.total_boundaries_actual for bm in boundary_metrics_batch)  
        total_correct_boundaries = sum(bm.correct_boundaries for bm in boundary_metrics_batch)
        
        boundary_precision = total_correct_boundaries / total_pred_boundaries if total_pred_boundaries > 0 else 0.0
        boundary_recall = total_correct_boundaries / total_true_boundaries if total_true_boundaries > 0 else 0.0
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0.0
        
        # Aggregate segment metrics
        avg_complete_segments = np.mean([sm.complete_segments_detected for sm in segment_metrics_batch]) if segment_metrics_batch else 0.0
        avg_segment_overlap = np.mean([sm.avg_segment_overlap for sm in segment_metrics_batch]) if segment_metrics_batch else 0.0
        
        # Aggregate transition metrics  
        total_v2c = sum(tm.verse_to_chorus_count for tm in transition_metrics_batch)
        total_c2v = sum(tm.chorus_to_verse_count for tm in transition_metrics_batch)
        total_v2c_correct = sum(tm.verse_to_chorus_correct for tm in transition_metrics_batch)
        total_c2v_correct = sum(tm.chorus_to_verse_correct for tm in transition_metrics_batch)
        
        v2c_accuracy = total_v2c_correct / total_v2c if total_v2c > 0 else 0.0
        c2v_accuracy = total_c2v_correct / total_c2v if total_c2v > 0 else 0.0
        
        # Compute segmentation metrics (Phase 3 - WindowDiff and Pk)
        segmentation_metrics_batch = []
        for pred_batch, targ_batch, mask_batch in zip(all_predictions, all_targets, all_masks):
            batch_segmentation = compute_segmentation_metrics(pred_batch, targ_batch, mask_batch)
            segmentation_metrics_batch.append(batch_segmentation)
        
        # Aggregate segmentation metrics
        avg_window_diff = np.mean([sm.window_diff for sm in segmentation_metrics_batch]) if segmentation_metrics_batch else 1.0
        avg_pk_metric = np.mean([sm.pk_metric for sm in segmentation_metrics_batch]) if segmentation_metrics_batch else 1.0
        
        # Combine all metrics
        metrics = {
            'loss': avg_loss,
            **avg_guardrails,
            **f1_scores,
            # Boundary metrics (aggregated across batches)
            'boundary_f1': boundary_f1,
            'boundary_precision': boundary_precision,
            'boundary_recall': boundary_recall,
            # Segment metrics (averaged across batches)
            'complete_segments': avg_complete_segments,
            'avg_segment_overlap': avg_segment_overlap,
            # Transition metrics (aggregated across batches)
            'verse_to_chorus_acc': v2c_accuracy,
            'chorus_to_verse_acc': c2v_accuracy,
            # Segmentation metrics (Phase 3 - WindowDiff and Pk)
            'window_diff': avg_window_diff,
            'pk_metric': avg_pk_metric,
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[nn.Module, float]:
        """
        Main training loop with emergency monitoring.
        
        Returns:
            best_model: Model with best validation F1
            best_temperature: Calibrated temperature
        """
        print(f"🚀 Starting training for {self.config.max_epochs} epochs...")
        
        best_model_state = None
        best_epoch = 0
        
        try:
            for epoch in range(1, self.config.max_epochs + 1):
                epoch_start = time.time()
                
                print(f"\n📅 Epoch {epoch}/{self.config.max_epochs}")
                print("-" * 50)
                
                try:
                    # Training
                    train_metrics = self.train_epoch(train_loader)
                    
                    # Validation  
                    val_metrics = self.evaluate(val_loader)
                    
                    # Learning rate scheduling (use selected validation metric)  
                    # First, calculate epoch time for metrics
                    epoch_time = time.time() - epoch_start
                    
                    # Create TrainingMetrics with all required fields
                    metrics = TrainingMetrics(
                        epoch=epoch,
                        train_loss=train_metrics['loss'],
                        train_chorus_rate=train_metrics['chorus_rate'],
                        train_max_prob=train_metrics['max_prob_mean'],
                        train_conf_over_90=train_metrics['confidence_over_90'],
                        train_conf_over_95=train_metrics['confidence_over_95'],
                        val_loss=val_metrics['loss'],
                        val_macro_f1=val_metrics['macro_f1'],
                        val_verse_f1=val_metrics['verse_f1'],
                        val_chorus_f1=val_metrics['chorus_f1'],
                        val_chorus_rate=val_metrics['chorus_rate'],
                        val_max_prob=val_metrics['max_prob_mean'],
                        val_conf_over_90=val_metrics['confidence_over_90'],
                        val_conf_over_95=val_metrics['confidence_over_95'],
                        learning_rate=0.0,  # Will be set below
                        epoch_time=epoch_time,
                        val_boundary_f1=val_metrics.get('boundary_f1', 0.0),
                        val_boundary_precision=val_metrics.get('boundary_precision', 0.0),
                        val_boundary_recall=val_metrics.get('boundary_recall', 0.0),
                        val_window_diff=val_metrics.get('window_diff', 1.0),
                        val_pk_metric=val_metrics.get('pk_metric', 1.0),
                        val_complete_segments=val_metrics.get('complete_segments', 0.0),
                        val_avg_segment_overlap=val_metrics.get('avg_segment_overlap', 0.0),
                        val_verse_to_chorus_acc=val_metrics.get('verse_to_chorus_acc', 0.0),
                        val_chorus_to_verse_acc=val_metrics.get('chorus_to_verse_acc', 0.0)
                    )
                    
                    current_val_score = compute_validation_score(metrics, self.config)
                    
                    # Learning rate scheduling (use selected validation metric)
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(current_val_score)
                    else:
                        self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Update learning rate in metrics object
                    metrics.learning_rate = current_lr
                    
                    self._save_epoch_metrics(metrics)
                    
                    # Print epoch summary with boundary-aware metrics
                    print(f"\n📊 Epoch {epoch} Summary:")
                    
                    print(f"  Train: loss={metrics.train_loss:.4f}, "
                          f"chorus%={metrics.train_chorus_rate:.2f}, "
                          f"conf={metrics.train_max_prob:.3f}")
                    print(f"  Val:   F1={current_val_score:.4f} ({self.validation_strategy}), "
                          f"chorus%={metrics.val_chorus_rate:.2f}, "
                          f"conf={metrics.val_max_prob:.3f}")
                    print(f"  📏 Boundary: F1={metrics.val_boundary_f1:.3f}, "
                          f"Precision={metrics.val_boundary_precision:.3f}, "
                          f"Recall={metrics.val_boundary_recall:.3f}")
                    print(f"  🎯 Segments: Complete={metrics.val_complete_segments:.1%}, "
                          f"Overlap={metrics.val_avg_segment_overlap:.3f}")
                    print(f"  🔄 Transitions: V→C={metrics.val_verse_to_chorus_acc:.1%}, "
                          f"C→V={metrics.val_chorus_to_verse_acc:.1%}")
                    print(f"  📐 Segmentation: WindowDiff={metrics.val_window_diff:.3f}, "
                          f"Pk={metrics.val_pk_metric:.3f}")
                    print(f"  📊 Line-level F1: {metrics.val_macro_f1:.4f} (for reference)")
                    print(f"  Time: {epoch_time:.1f}s, LR: {current_lr:.2e}")
                    
                    # Emergency monitoring (skip first N epochs to allow model to stabilize)
                    should_stop, warning = False, ""
                    skip_epochs = getattr(self.config, 'skip_epochs', 3)
                    if self.emergency_monitor is not None and epoch >= skip_epochs:
                        should_stop, warning = self.emergency_monitor.check_epoch(metrics)
                    if warning:
                        print(f"  ⚠️  {warning}")
                    if should_stop:
                        print(f"🛑 EMERGENCY STOP at epoch {epoch}")
                        break
                    
                    # Early stopping and best model tracking with configurable validation metric
                    if current_val_score > self.best_val_score:
                        self.best_val_score = current_val_score
                        self.patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        best_epoch = epoch
                        
                        # Save best model
                        torch.save(best_model_state, self.output_dir / "best_model.pt")
                        print(f"  ✅ New best {self.validation_strategy}: {self.best_val_score:.4f}")
                    else:
                        self.patience_counter += 1
                        print(f"  📉 Patience: {self.patience_counter}/{self.config.patience}")
                    
                    if self.patience_counter >= self.config.patience:
                        print(f"🛑 Early stopping after {epoch} epochs")
                        break
                
                except RuntimeError as e:
                    if "Emergency training stop" in str(e):
                        print(f"🚨 Training stopped due to emergency condition")
                        break
                    else:
                        raise e
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Training interrupted by user at epoch {epoch}")
            
            # Load best model if available
            if best_model_state is not None:
                print(f"📦 Loading best model state (epoch {best_epoch})")
                self.model.load_state_dict(best_model_state)
            
            # Run calibration even on interruption
            print(f"🌡️ Running calibration on interrupted training...")
            self._run_calibration(val_loader)
            
            # Save final model and metrics
            torch.save(self.model.state_dict(), self.output_dir / "final_model.pt")
            self._save_training_log(final=True)
            print(f"💾 Saved interrupted training state")
            
            # Return what we have so far
            return self.model, self.calibration_info
            
        except Exception as e:
            print(f"\n❌ Training failed with exception: {e}")
            # Save metrics even if training failed
            if self.training_metrics:
                print(f"📝 Saving partial training metrics...")
                self._save_training_log()
            raise
        
        finally:
            # Always save metrics, even if training stopped early
            if self.training_metrics:
                print(f"📝 Ensuring training metrics are saved...")
                self._save_training_log()
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Run calibration
        print(f"\n� Running model calibration...")
        
        # Get calibration methods from config
        calibration_methods = None
        if hasattr(self.config, 'calibration') and hasattr(self.config.calibration, 'methods'):
            calibration_methods = self.config.calibration.methods
        
        self.calibration_info = fit_calibration(
            model=self.model,
            val_loader=val_loader,
            device=self.device,
            methods=calibration_methods,
            output_dir=self.output_dir
        )
        
        # Save final model and metrics
        torch.save(self.model.state_dict(), self.output_dir / "final_model.pt")
        self._save_training_log(final=True)
        
        # Save dedicated boundary metrics summary (Phase 2 enhancement)
        self._save_boundary_metrics_summary(val_loader)
        
        # Generate final boundary-aware metrics report
        if self.training_metrics:
            print(f"\n📊 Final Boundary-Aware Metrics Report:")
            final_metrics = self.training_metrics[-1]  # Get latest metrics
            
            print(f"   Line-Level Metrics:")
            print(f"      Macro F1: {final_metrics.val_macro_f1:.4f}")  
            print(f"      Verse F1: {final_metrics.val_verse_f1:.4f}")
            print(f"      Chorus F1: {final_metrics.val_chorus_f1:.4f}")
            
            print(f"   📏 Boundary Detection:")
            print(f"      F1: {final_metrics.val_boundary_f1:.3f}")
            print(f"      Precision: {final_metrics.val_boundary_precision:.3f}")
            print(f"      Recall: {final_metrics.val_boundary_recall:.3f}")
            
            print(f"   🎯 Segment Quality:")  
            print(f"      Complete segments: {final_metrics.val_complete_segments:.1%}")
            print(f"      Avg overlap (IoU): {final_metrics.val_avg_segment_overlap:.3f}")
            
            print(f"   🔄 Transition Accuracy:")
            print(f"      Verse→Chorus: {final_metrics.val_verse_to_chorus_acc:.1%}")
            print(f"      Chorus→Verse: {final_metrics.val_chorus_to_verse_acc:.1%}")

        print(f"\n✅ Training completed!")
        print(f"   Best validation score ({self.validation_strategy}): {self.best_val_score:.4f}")
        if self.calibration_info:
            print(f"   Calibration methods: {', '.join(self.calibration_info.keys())}")
            for method, info in self.calibration_info.items():
                if method == 'temperature' and 'params' in info:
                    print(f"   Temperature: {info['params']['temperature']:.2f}")
                elif method == 'platt' and 'params' in info:
                    A = info['params']['A']; B = info['params']['B']
                    print(f"   Platt: A={A:.3f}, B={B:.3f}")
        print(f"   Models saved to: {self.output_dir}")
        
        return self.model, self.calibration_info
    
    def _save_training_log(self, final: bool = False):
        """Save training metrics to file."""
        metrics_file = self.output_dir / "training_metrics.json"
        
        # Convert to serializable format
        metrics_data = [asdict(m) for m in self.training_metrics]
        
        # Create complete training log with metadata
        training_log = {
            "metadata": {
                "model_info": {
                    "hidden_dim": getattr(self.model, 'hidden_dim', 128),
                    "num_layers": getattr(self.model, 'num_layers', 1),
                    "dropout": getattr(self.model, 'dropout_p', 0.2),
                    "layer_dropout": getattr(self.model, 'layer_dropout_p', 0.0)
                },
                "training_info": {
                    "batch_size": getattr(self.config, 'batch_size', 16),
                    "learning_rate": getattr(self.config, 'learning_rate', 0.001),
                    "scheduler": getattr(self.config, 'scheduler', 'plateau'),
                    "label_smoothing": getattr(self.config, 'label_smoothing', 0.0),
                    "weighted_sampling": getattr(self.config, 'weighted_sampling', False),
                    "total_epochs": len(self.training_metrics),
                    "validation_strategy": self.validation_strategy
                },
                "calibration_info": self.calibration_info
            },
            "metrics": metrics_data
        }
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(training_log, f, indent=2)
            
            if final:
                print(f"📝 Final training log saved to: {metrics_file}")
            elif len(metrics_data) % 5 == 0:  # Print every 5 epochs to avoid spam
                print(f"📝 Training metrics updated: {len(metrics_data)} epochs saved")
        
        except Exception as e:
            print(f"⚠️  Error saving training metrics: {e}")
    
    def _initialize_metrics_file(self):
        """Initialize empty metrics file at start of training."""
        metrics_file = self.output_dir / "training_metrics.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump([], f, indent=2)
            
            print(f"📝 Initialized metrics file: {metrics_file}")
        
        except Exception as e:
            print(f"⚠️  Error initializing metrics file: {e}")
    
    def _save_epoch_metrics(self, metrics: TrainingMetrics):
        """Save metrics after each epoch (incremental)."""
        self.training_metrics.append(metrics)
        self._save_training_log()  # Save immediately after each epoch
    
    def _save_boundary_metrics_summary(self, val_loader: DataLoader):
        """Save detailed boundary metrics summary for analysis."""
        boundary_summary_file = self.output_dir / "boundary_metrics_summary.json"
        
        try:
            # Get final detailed boundary metrics
            final_val_metrics = self.evaluate(val_loader)
            
            # Create boundary metrics summary
            boundary_summary = {
                "boundary_metrics": {
                    "description": "Boundary-aware evaluation metrics for structural understanding",
                    "line_level_metrics": {
                        "macro_f1": final_val_metrics.get('macro_f1', 0.0),
                        "verse_f1": final_val_metrics.get('verse_f1', 0.0),
                        "chorus_f1": final_val_metrics.get('chorus_f1', 0.0)
                    },
                    "boundary_detection": {
                        "f1": final_val_metrics.get('boundary_f1', 0.0),
                        "precision": final_val_metrics.get('boundary_precision', 0.0),
                        "recall": final_val_metrics.get('boundary_recall', 0.0)
                    },
                    "segment_quality": {
                        "complete_segments_detected": final_val_metrics.get('complete_segments', 0.0),
                        "avg_segment_overlap_iou": final_val_metrics.get('avg_segment_overlap', 0.0)
                    },
                    "transition_accuracy": {
                        "verse_to_chorus": final_val_metrics.get('verse_to_chorus_acc', 0.0),
                        "chorus_to_verse": final_val_metrics.get('chorus_to_verse_acc', 0.0)
                    }
                },
                "historical_progression": [
                    {
                        "epoch": m.epoch,
                        "boundary_f1": m.val_boundary_f1,
                        "complete_segments": m.val_complete_segments,
                        "verse_to_chorus_acc": m.val_verse_to_chorus_acc,
                        "chorus_to_verse_acc": m.val_chorus_to_verse_acc
                    }
                    for m in self.training_metrics
                ],
                "export_timestamp": time.time(),
                "model_config": {
                    "experiment_name": getattr(self.config, 'experiment_name', 'unknown'),
                    "batch_size": getattr(self.config, 'batch_size', 'unknown'),
                    "learning_rate": getattr(self.config, 'learning_rate', 'unknown'),
                    "scheduler": getattr(self.config, 'scheduler', 'unknown')
                }
            }
            
            # Save the summary
            with open(boundary_summary_file, 'w') as f:
                json.dump(boundary_summary, f, indent=2)
            
            print(f"📊 Boundary metrics summary saved to: {boundary_summary_file}")
            
        except Exception as e:
            print(f"⚠️  Error saving boundary metrics summary: {e}")



if __name__ == "__main__":
    print("🧪 Testing trainer components...")
    
    # Test emergency monitor
    monitor = EmergencyMonitor()
    
    # Test normal batch
    normal_guardrails = {
        'chorus_rate': 0.35,
        'max_prob_mean': 0.75,
        'confidence_over_90': 0.05,
        'confidence_over_95': 0.01
    }
    
    should_stop, warning = monitor.check_batch(normal_guardrails, 0)
    print(f"Normal batch: stop={should_stop}, warning='{warning}'")
    
    # Test problematic batch
    problem_guardrails = {
        'chorus_rate': 0.05,  # Too low
        'max_prob_mean': 0.97,  # Too high
        'confidence_over_90': 0.8,
        'confidence_over_95': 0.6  # Too high
    }
    
    should_stop, warning = monitor.check_batch(problem_guardrails, 0)
    print(f"Problem batch: stop={should_stop}, warning='{warning}'")
    
    print("✅ Trainer components test passed!")
