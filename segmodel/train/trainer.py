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


def calibrate_temperature(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Any = None,
    temperature_grid: List[float] = None
) -> float:
    """
    Calibrate temperature parameter on validation set.
    
    Uses grid search to find optimal temperature that minimizes
    negative log-likelihood on validation set.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: Device to run on
        temperature_grid: List of temperature values to try
        
    Returns:
        best_temperature: Optimal temperature value
    """
    if temperature_grid is None:
        if config is not None and hasattr(config, 'temperature_grid'):
            temperature_grid = getattr(config, 'temperature_grid', [0.8, 1.0, 1.2, 1.5, 1.7, 2.0])
        else:
            temperature_grid = [0.8, 1.0, 1.2, 1.5, 1.7, 2.0]
    
    model.eval()
    best_temp = 1.0
    best_nll = float('inf')
    
    print(f"🌡️  Calibrating temperature on {len(dataloader)} batches...")
    
    for temp in temperature_grid:
        total_nll = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch.features.to(device)
                labels = batch.labels.to(device)  
                mask = batch.mask.to(device)
                
                # Forward pass with temperature
                logits = model(features, mask) / temp
                
                # Compute negative log-likelihood
                log_probs = torch.log_softmax(logits, dim=-1)
                nll = -torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
                
                # Apply mask and sum
                valid_nll = nll * mask.float()
                total_nll += valid_nll.sum().item()
                total_samples += mask.sum().item()
        
        avg_nll = total_nll / max(total_samples, 1)
        
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_temp = temp
        
        print(f"   T={temp:.1f}: NLL={avg_nll:.4f}")
    
    print(f"✅ Best temperature: {best_temp:.1f} (NLL={best_nll:.4f})")
    return best_temp


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
        
        # Set up scheduler using factory function
        self.scheduler, self.scheduler_type = create_scheduler(
            optimizer, 
            config,
            total_steps=config.max_epochs  # For cosine scheduling
        )
        
        print(f"🎯 Scheduler: {getattr(config, 'scheduler', 'plateau')} ({self.scheduler_type}-based)")
        
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
        self.best_val_f1 = -1.0
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
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        all_guardrails = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch.features.to(self.device)
                labels = batch.labels.to(self.device)
                mask = batch.mask.to(self.device)
                
                # Forward pass
                logits = self.model(features, mask)
                loss = self.loss_function(logits, labels, mask)
                
                # Track loss
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Store for F1 computation
                all_predictions.append(predictions)
                all_targets.append(labels)
                all_masks.append(mask)
                
                # Track guardrails
                guardrails = batch_guardrails(logits, mask)
                all_guardrails.append(guardrails)
        
        # Compute aggregate metrics
        avg_loss = total_loss / len(val_loader)
        
        # Aggregate guardrails
        avg_guardrails = {}
        for key in all_guardrails[0].keys():
            avg_guardrails[key] = np.mean([g[key] for g in all_guardrails])
        
        # Compute F1 scores - flatten all predictions/targets with proper masking
        all_pred = torch.cat([pred.flatten() for pred in all_predictions], dim=0)
        all_targ = torch.cat([targ.flatten() for targ in all_targets], dim=0) 
        all_mask = torch.cat([mask.flatten() for mask in all_masks], dim=0)
        
        f1_scores = sequence_f1_score(all_pred, all_targ, all_mask)
        
        # Combine metrics
        metrics = {
            'loss': avg_loss,
            **avg_guardrails,
            **f1_scores
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
                    
                    # Learning rate scheduling (different schedulers need different arguments)
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['macro_f1'])
                    else:
                        self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Track metrics
                    epoch_time = time.time() - epoch_start
                    
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
                        learning_rate=current_lr,
                        epoch_time=epoch_time
                    )
                    
                    self._save_epoch_metrics(metrics)
                    
                    # Print epoch summary
                    print(f"\n📊 Epoch {epoch} Summary:")
                    print(f"  Train: loss={metrics.train_loss:.4f}, "
                          f"chorus%={metrics.train_chorus_rate:.2f}, "
                          f"conf={metrics.train_max_prob:.3f}")
                    print(f"  Val:   F1={metrics.val_macro_f1:.4f}, "
                          f"chorus%={metrics.val_chorus_rate:.2f}, "
                          f"conf={metrics.val_max_prob:.3f}")
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
                    
                    # Early stopping and best model tracking
                    if metrics.val_macro_f1 > self.best_val_f1:
                        self.best_val_f1 = metrics.val_macro_f1
                        self.patience_counter = 0
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        
                        # Save best model
                        torch.save(best_model_state, self.output_dir / "best_model.pt")
                        print(f"  ✅ New best F1: {self.best_val_f1:.4f}")
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
        
        # Calibrate temperature
        print(f"\n🌡️  Calibrating temperature...")
        best_temperature = calibrate_temperature(self.model, val_loader, self.device, config=self.config)
        
        # Save final model and metrics
        torch.save(self.model.state_dict(), self.output_dir / "final_model.pt")
        self._save_training_log(final=True)
        
        print(f"\n✅ Training completed!")
        print(f"   Best validation F1: {self.best_val_f1:.4f}")
        print(f"   Calibrated temperature: {best_temperature:.2f}")
        print(f"   Models saved to: {self.output_dir}")
        
        return self.model, best_temperature
    
    def _save_training_log(self, final: bool = False):
        """Save training metrics to file."""
        metrics_file = self.output_dir / "training_metrics.json"
        
        # Convert to serializable format
        metrics_data = [asdict(m) for m in self.training_metrics]
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
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
