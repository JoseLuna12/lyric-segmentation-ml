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
            # base probs for non-top classes will share leftover mass
            preds = logits.argmax(dim=1)              # (N,)
            top2 = torch.topk(logits, k=2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]         # (N,)
            z = self.A * margins + self.B
            p_top = torch.sigmoid(z)                  # calibrated top1 prob, (N,)

            N, C = logits.shape
            out = torch.zeros_like(logits)

            # scatter top1 prob
            out[torch.arange(N), preds] = p_top

            # distribute remaining mass proportionally (shape-preserving)
            if C > 1:
                orig_probs = torch.softmax(logits, dim=-1)  # original shape
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
            return out
    
    def get_params(self) -> Dict[str, float]:
        """Get calibration parameters."""
        return {"A": self.A, "B": self.B}


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
        methods: List of methods to try ['temperature', 'platt']
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
        else:
            return 'none', None
        
        return method, calibrator
        
    except Exception as e:
        print(f"❌ Error loading calibration: {e}")
        return 'none', None
