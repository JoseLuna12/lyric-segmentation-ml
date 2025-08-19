"""
Clean implementation of model calibration for BiLSTM segmentation.

This module implements temperature scaling, Platt scaling, and isotonic regression
for calibrating model confidence scores in a clean, robust manner.
"""

import torch
import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.optimize import minimize_scalar, minimize, Bounds

from ..metrics import ece


@dataclass
class CalibrationResult:
    """Results from calibration fitting."""
    method: str
    params: Dict[str, float]
    ece_before: float
    ece_after: float
    improvement: float


class TemperatureCalibrator:
    """
    Temperature scaling calibrator.
    
    Applies a single temperature parameter T to logits: p = softmax(logits / T)
    - T > 1: reduces confidence (more conservative)
    - T < 1: increases confidence (more aggressive)
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Find optimal temperature by minimizing NLL.
        
        Args:
            logits: Model logits (N, num_classes) - already filtered
            labels: True labels (N,) - already filtered
            
        Returns:
            ece_after: ECE after calibration
        """
        def temperature_nll(T):
            if T <= 0:
                return float('inf')
            
            with torch.no_grad():
                scaled_logits = logits / T
                log_probs = torch.log_softmax(scaled_logits, dim=-1)
                nll = -torch.gather(log_probs, 1, labels.unsqueeze(-1)).squeeze(-1)
                return nll.mean().item()
        
        # Optimize temperature
        result = minimize_scalar(temperature_nll, bounds=(0.1, 5.0), method='bounded')
        self.temperature = result.x
        self.is_fitted = True
        
        # Compute ECE after calibration
        with torch.no_grad():
            calibrated_probs = self.apply(logits)
            calibrated_probs_np = calibrated_probs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            ece_after = ece(calibrated_probs_np, labels_np, mask=None)
        
        return ece_after
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if not self.is_fitted:
            return torch.softmax(logits, dim=-1)
        
        with torch.no_grad():
            scaled_logits = logits / self.temperature
            return torch.softmax(scaled_logits, dim=-1)
    
    def get_params(self) -> Dict[str, float]:
        """Get calibration parameters."""
        return {"temperature": self.temperature}


class PlattCalibrator:
    """
    Platt scaling calibrator.
    
    Applies sigmoid(A * margin + B) where margin is the difference between 
    top1 and top2 logits for better calibration quality.
    """
    
    def __init__(self):
        self.A = 1.0
        self.B = 0.0
        self.is_fitted = False
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Find optimal Platt parameters using logit margins.
        
        Args:
            logits: Model logits (N, num_classes) - already filtered  
            labels: True labels (N,) - already filtered
            
        Returns:
            ece_after: ECE after calibration
        """
        with torch.no_grad():
            # logits: (N, C)
            top2 = torch.topk(logits, k=2, dim=1).values  # (N, 2)
            margins = (top2[:, 0] - top2[:, 1]).cpu().numpy()  # (N,)
            preds = logits.argmax(dim=1)
            correct = (preds == labels).float().cpu().numpy()

        def nll(params):
            A, B = params
            z = np.clip(A * margins + B, -50, 50)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-15, 1 - 1e-15)
            data_loss = -np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p))
            # Add tiny L2 regularization to prevent extreme parameters
            reg = 1e-3 * (A*A + B*B)
            return data_loss + reg

        # Try multiple starting points to avoid local minima
        bounds = Bounds(lb=[0.0, -5.0], ub=[5.0, 5.0])  # A>=0 for monotonicity
        
        best_result = None
        best_loss = float('inf')
        best_start = 0
        
        # Analyze margins to inform starting points
        margin_mean = np.mean(margins)
        margin_std = np.std(margins)
        
        # Smart starting points based on data characteristics
        smart_A = 1.0 / max(margin_std, 0.1)  # Scale based on margin spread
        starting_points = [
            [1.0, 0.0],                        # Traditional starting point
            [smart_A, 0.0],                    # Data-adaptive scaling
            [0.5, 0.0],                        # Conservative scaling
            [1.5, 0.0],                        # Aggressive scaling
            [1.0, -margin_mean * 0.1],         # Bias correction
        ]
        
        print(f"    Platt optimization: trying {len(starting_points)} starting points...")
        print(f"    Margin stats: mean={margin_mean:.3f}, std={margin_std:.3f}, smart_A={smart_A:.3f}")
        
        for i, x0 in enumerate(starting_points):
            try:
                result = minimize(nll, x0=x0, method='L-BFGS-B', bounds=bounds)
                if result.success and result.fun < best_loss:
                    best_result = result
                    best_loss = result.fun
                    best_start = i
            except:
                continue
        
        # Fallback to unconstrained if bounded optimization fails
        if best_result is None:
            print(f"    Bounded optimization failed, trying BFGS...")
            best_result = minimize(nll, x0=[1.0, 0.0], method='BFGS')
            
        # Final fallback
        if not best_result.success:
            print(f"    BFGS failed, trying Nelder-Mead...")
            best_result = minimize(nll, x0=[1.0, 0.0], method='Nelder-Mead')
        
        # Print debugging info
        print(f"    Initial NLL (A=1,B=0): {nll([1.0, 0.0]):.6f}")
        print(f"    Best NLL: {best_result.fun:.6f} (from starting point {best_start})")
        print(f"    Optimized params: A={best_result.x[0]:.4f}, B={best_result.x[1]:.4f}")
        
        self.A, self.B = float(best_result.x[0]), float(best_result.x[1])
        self.is_fitted = True

        # ECE after calibration
        with torch.no_grad():
            calibrated = self.apply(logits)  # (N, C)
            ece_after = ece(calibrated, labels, mask=None)
        return ece_after
    
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling to logits using margins."""
        if not self.is_fitted:
            return torch.softmax(logits, dim=-1)

        with torch.no_grad():
            # Handle both 2D (N, C) and 3D (B, S, C) logits
            original_shape = logits.shape
            if len(original_shape) == 3:
                # Flatten 3D to 2D for calibration, then reshape back
                batch_size, seq_len, num_classes = original_shape
                logits_flat = logits.view(-1, num_classes)  # (B*S, C)
            else:
                logits_flat = logits
            
            # base probs for non-top classes will share leftover mass
            preds = logits_flat.argmax(dim=1)              # (N,)
            top2 = torch.topk(logits_flat, k=2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]         # (N,)
            z = self.A * margins + self.B
            p_top = torch.sigmoid(z)                  # calibrated top1 prob, (N,)

            N, C = logits_flat.shape
            out = torch.zeros_like(logits_flat)

            # scatter top1 prob
            out[torch.arange(N), preds] = p_top

            # distribute remaining mass proportionally (shape-preserving)
            if C > 1:
                orig_probs = torch.softmax(logits_flat, dim=-1)  # Use flattened logits
                # scale non-top probabilities to fit in remaining mass
                non_top_orig = orig_probs.clone()
                non_top_orig[torch.arange(N), preds] = 0  # zero out top class
                non_top_sum = non_top_orig.sum(dim=1, keepdim=True)  # (N, 1)
                
                # avoid division by zero
                valid_mask = non_top_sum.squeeze() > 1e-8
                remaining_mass = (1.0 - p_top).unsqueeze(1)  # (N, 1)
                
                # shape-preserving scaling
                if valid_mask.any():
                    scaling = remaining_mass / non_top_sum.clamp(min=1e-8)
                    out += non_top_orig * scaling
                else:
                    # fallback: uniform if original probs are degenerate
                    out += remaining_mass / C
                
                # reassign calibrated top probability
                out[torch.arange(N), preds] = p_top
            
            # Reshape back to original shape if needed
            if len(original_shape) == 3:
                out = out.view(original_shape)
            
            return out
    
    def get_params(self) -> Dict[str, float]:
        """Get calibration parameters."""
        return {"A": self.A, "B": self.B}


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for top-1 probabilities.
    Learns g(p_top) ~ P(correct | p_top). Applies g to top-1 prob and
    rescales non-top probs proportionally (shape-preserving).
    """
    
    def __init__(self, out_of_bounds: str = "clip", min_samples: int = 100):
        # out_of_bounds: 'clip' replicates sklearn>=1.0 behavior for extrapolation
        self.out_of_bounds = out_of_bounds
        self.min_samples = min_samples
        self.is_fitted = False
        self._iso = None

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Args:
            logits: (N, C)
            labels: (N,)
        Returns:
            ece_after (float)
        """
        with torch.no_grad():
            base = torch.softmax(logits, dim=-1)      # (N,C)
            preds = base.argmax(dim=1)                # (N,)
            correct = (preds == labels).float()       # (N,)
            p_top = base.gather(1, preds.unsqueeze(1)).squeeze(1)  # (N,)

        # Need enough distinct points for isotonic to be meaningful
        p_np = p_top.detach().cpu().numpy()
        y_np = correct.detach().cpu().numpy()

        print(f"    Isotonic calibration: {len(p_np)} samples, {np.unique(p_np).size} unique confidences")
        
        # Fallback: if too few samples or degenerate p's, skip training
        # More adaptive thresholds for different dataset sizes
        min_unique_values = max(3, min(10, len(np.unique(p_np)) // 3))  # At least 3, aim for 1/3 of unique values
        if len(p_np) < self.min_samples or np.unique(p_np).size < min_unique_values:
            print(f"    Insufficient data for isotonic regression (need {self.min_samples}+ samples, {min_unique_values}+ unique values)")
            print(f"    Got {len(p_np)} samples, {np.unique(p_np).size} unique values")
            print(f"    Using identity mapping as fallback")
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds=self.out_of_bounds)
            # Fit a trivial mapping 0..1 -> 0..1 to keep code paths unified
            iso.fit([0.0, 1.0], [0.0, 1.0])
            self._iso = iso
            self.is_fitted = True
        else:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds=self.out_of_bounds)
            # Fit on (p_top, correctness)
            print(f"    Fitting isotonic regression on confidence range [{p_np.min():.3f}, {p_np.max():.3f}]")
            iso.fit(p_np, y_np)
            self._iso = iso
            self.is_fitted = True
            print(f"    Fitted with {getattr(iso, 'X_thresholds_', np.array([])).size} knots")

        # ECE after calibration
        with torch.no_grad():
            calibrated = self.apply(logits)  # (N, C)
            ece_after = ece(calibrated, labels, mask=None)
        return float(ece_after)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply isotonic mapping to top-1 prob and preserve non-top shape."""
        with torch.no_grad():
            # Handle both 2D (N, C) and 3D (B, S, C) logits
            original_shape = logits.shape
            if len(original_shape) == 3:
                # Flatten 3D to 2D for calibration, then reshape back
                batch_size, seq_len, num_classes = original_shape
                logits_flat = logits.view(-1, num_classes)  # (B*S, C)
            else:
                logits_flat = logits
            
            base = torch.softmax(logits_flat, dim=-1)  # (N,C)
            if not self.is_fitted or self._iso is None:
                if len(original_shape) == 3:
                    base = base.view(original_shape)
                return base

            N, C = base.shape
            preds = base.argmax(dim=1)  # (N,)
            base_top = base.gather(1, preds.unsqueeze(1)).squeeze(1)  # (N,)

            p_top_np = base_top.detach().cpu().numpy()
            p_top_cal_np = self._iso.predict(p_top_np)
            p_top_cal = torch.from_numpy(np.clip(p_top_cal_np, 0.0, 1.0)).to(base.device).to(base.dtype)  # (N,)

            # Shape-preserving rescale of non-top classes
            denom = (1.0 - base_top).clamp_min(1e-12)        # (N,)
            scale = (1.0 - p_top_cal) / denom                # (N,)

            out = base * scale.unsqueeze(1)                  # shrink all
            out[torch.arange(N), preds] = p_top_cal          # then set top class
            # Numerical guard
            out = torch.clamp(out, 0.0, 1.0)
            out = out / out.sum(dim=1, keepdim=True).clamp_min(1e-12)
            
            # Reshape back to original shape if needed
            if len(original_shape) == 3:
                out = out.view(original_shape)
            
            return out

    def get_params(self) -> Dict[str, float]:
        return {"knots": int(getattr(self._iso, "X_thresholds_", np.array([])).size)
                        if self._iso is not None else 0}


def collect_validation_data(model, val_loader, device):
    """
    Collect logits and labels from validation set.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        logits: Flattened logits (N, num_classes)
        labels: Flattened labels (N,)
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch.features.to(device)
            labels_batch = batch.labels.to(device)
            mask = batch.mask.to(device)
            
            # Forward pass
            logits_batch = model(features, mask)
            
            # Extract valid positions only
            # Note: This assumes masks are True prefix then padding (typical case).
            # For non-contiguous masks, use: m = mask[b].bool(); 
            # all_logits.append(logits_batch[b, m]); all_labels.append(labels_batch[b, m])
            batch_size, seq_len = logits_batch.size(0), logits_batch.size(1)
            for b in range(batch_size):
                valid_positions = mask[b].sum().item()
                if valid_positions > 0:
                    all_logits.append(logits_batch[b, :valid_positions])
                    all_labels.append(labels_batch[b, :valid_positions])
    
    if all_logits:
        logits = torch.cat(all_logits, dim=0)  # (N, num_classes)
        labels = torch.cat(all_labels, dim=0)  # (N,)
        return logits, labels
    else:
        return None, None


def fit_calibration(model, val_loader, device, methods=None, output_dir=None):
    """
    Fit calibration methods and select the best one.
    
    Args:
        model: Trained model
        val_loader: Validation data loader  
        device: Device to run on
        methods: List of methods to try ['temperature', 'platt', 'isotonic']
        output_dir: Directory to save results
        
    Returns:
        best_method: Name of best method
        best_calibrator: Best calibrator instance
        results: List of CalibrationResult objects
    """
    if methods is None:
        methods = ['temperature', 'platt']
    
    print(f"🌡️ Fitting {len(methods)} calibration methods...")
    
    # Collect validation data
    logits, labels = collect_validation_data(model, val_loader, device)
    if logits is None:
        print("❌ No validation data found for calibration")
        return {}
    
    # Compute uncalibrated ECE
    with torch.no_grad():
        uncalibrated_probs = torch.softmax(logits, dim=-1)
        # Convert to numpy for ECE calculation
        probs_np = uncalibrated_probs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # For ECE calculation, we need to pass the data correctly
        # The ECE function handles 2D probs (N, num_classes) and 1D labels (N,)
        ece_before = ece(probs_np, labels_np, mask=None)
    
    print(f"📊 Uncalibrated ECE: {ece_before:.4f}")
    
    results = []
    calibrators = {}
    
    # Try each method
    for method in methods:
        print(f"🔍 Fitting {method} calibration...")
        
        try:
            if method == 'temperature':
                calibrator = TemperatureCalibrator()
            elif method == 'platt':
                calibrator = PlattCalibrator()
            elif method == 'isotonic':
                # Adaptive min_samples based on dataset size
                # More aggressive adaptation for larger datasets
                adaptive_min_samples = max(20, min(50, len(logits) // 10))  # 10% of data, min 20, max 50
                print(f"    Using adaptive min_samples={adaptive_min_samples} for dataset size {len(logits)}")
                calibrator = IsotonicCalibrator(out_of_bounds="clip", min_samples=adaptive_min_samples)
            else:
                print(f"❌ Unknown method: {method}")
                continue
            
            ece_after = calibrator.fit(logits, labels)
            improvement = ece_before - ece_after
            
            result = CalibrationResult(
                method=method,
                params=calibrator.get_params(),
                ece_before=ece_before,
                ece_after=ece_after,
                improvement=improvement
            )
            
            results.append(result)
            calibrators[method] = calibrator
            
            print(f"✅ {method}: ECE {ece_before:.4f} → {ece_after:.4f} (Δ{improvement:+.4f})")
            
        except Exception as e:
            print(f"❌ Error fitting {method}: {e}")
    
    if not results:
        print("❌ No calibration methods succeeded")
        return {}
    
    # Select best method (lowest ECE after calibration)
    # Exclude isotonic from being the "best" method since it's not available for inference
    available_results = [r for r in results if r.method in ['temperature', 'platt']]
    if available_results:
        best_result = min(available_results, key=lambda r: r.ece_after)
        best_method = best_result.method
        best_calibrator = calibrators[best_method]
        
        # Check if isotonic was actually better
        isotonic_results = [r for r in results if r.method == 'isotonic']
        if isotonic_results:
            isotonic_result = isotonic_results[0]
            if isotonic_result.ece_after < best_result.ece_after:
                print(f"🏆 Best method: isotonic (ECE: {isotonic_result.ece_after:.4f}) - not available for inference")
                print(f"🏆 Best available method: {best_method} (ECE: {best_result.ece_after:.4f})")
            else:
                print(f"🏆 Best method: {best_method} (ECE: {best_result.ece_after:.4f})")
        else:
            print(f"🏆 Best method: {best_method} (ECE: {best_result.ece_after:.4f})")
    else:
        # Fallback to overall best if no temperature/platt available
        best_result = min(results, key=lambda r: r.ece_after)
        best_method = best_result.method
        best_calibrator = calibrators[best_method]
        print(f"🏆 Best method: {best_method} (ECE: {best_result.ece_after:.4f})")
    
    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save calibration parameters
        calib_file = output_path / "calibration.json"
        calib_data = {
            "method": best_method,
            "params": best_result.params,
            "ece_before": best_result.ece_before,
            "ece_after": best_result.ece_after,
            "improvement": best_result.improvement,
            "all_results": [asdict(r) for r in results]
        }
        
        with open(calib_file, 'w') as f:
            json.dump(calib_data, f, indent=2)
        
        print(f"💾 Saved calibration to {calib_file}")
    
    # Return structured calibration info (keep nested params)
    calibration_info = {}
    for result in results:
        calibration_info[result.method] = {
            'ece_before': result.ece_before,
            'ece_after': result.ece_after,
            'improvement': result.improvement,
            'params': result.params  # Keep nested
        }
    
    return calibration_info


def load_calibration(calibration_path):
    """
    Load calibration from saved file.
    
    Args:
        calibration_path: Path to calibration.json
        
    Returns:
        method: Calibration method
        calibrator: Calibrator instance
    """
    try:
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        
        method = data['method']
        params = data['params']
        
        if method == 'temperature':
            calibrator = TemperatureCalibrator()
            calibrator.temperature = params['temperature']
            calibrator.is_fitted = True
        elif method == 'platt':
            calibrator = PlattCalibrator() 
            calibrator.A = params['A']
            calibrator.B = params['B']
            calibrator.is_fitted = True
        elif method == 'isotonic':
            # Note: Full isotonic model serialization would require saving the fitted sklearn model
            # For now, we'll indicate that isotonic needs to be re-fitted
            print("⚠️ Isotonic calibration models cannot be fully serialized - using identity mapping")
            calibrator = IsotonicCalibrator()
            # Mark as fitted but with empty model (will behave as identity)
            calibrator.is_fitted = True
            calibrator._iso = None  # Force identity behavior
        else:
            return 'none', None
        
        return method, calibrator
        
    except Exception as e:
        print(f"❌ Error loading calibration: {e}")
        return 'none', None
