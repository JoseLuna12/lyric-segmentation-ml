#!/usr/bin/env python3
"""
Validation script for trained models.
Calculates comprehensive metrics for verse/chorus structure predictions.
"""

import argparse
import torch
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Sequence

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from segmodel.features import FeatureExtractor
from segmodel.models import BLSTMTagger
from segmodel.utils import load_training_config
from segmodel.utils.prediction_config import (
    PredictionConfig, 
    load_prediction_config, 
    auto_detect_prediction_config,
    create_prediction_config_from_training_config,
    create_prediction_config_from_training_session,
    get_feature_extractor_from_config
)


def load_model(model_path: str, device: torch.device, training_config_path: str = None) -> BLSTMTagger:
    """Load trained model from file."""
    print(f"📦 Loading model from {model_path}...")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer model dimensions from state dict
    # LSTM layer: 'lstm.weight_ih_l0' has shape (4*hidden_size, input_size)
    # Classifier: 'classifier.weight' has shape (num_classes, 2*hidden_size) for bidirectional
    
    lstm_weight = state_dict['lstm.weight_ih_l0']
    classifier_weight = state_dict['classifier.weight']
    
    input_size = lstm_weight.shape[1]
    lstm_4h = lstm_weight.shape[0]  # 4 * hidden_size
    hidden_size = lstm_4h // 4      # Extract hidden_size
    num_classes = classifier_weight.shape[0]
    # Count LSTM layers by looking for weight keys
    forward_layer_keys = [key for key in state_dict.keys() 
                         if key.startswith('lstm.weight_ih_l') and '_reverse' not in key]
    num_layers = len(forward_layer_keys)
    
    # Detect attention parameters from state dict and training config
    attention_enabled = any(key.startswith('attention.') for key in state_dict.keys())
    attention_type = 'self'  # Default attention type
    attention_heads = 8  # Default, will be overridden by config
    attention_dropout = 0.1  # Default, will be overridden by config
    positional_encoding = False  # Default, will be overridden by config
    attention_dim = None  # Will be determined from training config or saved weights
    boundary_temperature = 2.0  # Default boundary temperature
    
    # Flag to track if config was successfully loaded
    config_loaded = False
    
    # Try to read attention config from training config file first
    if training_config_path and os.path.exists(training_config_path):
        try:
            print(f"📋 Loading attention config from training config: {training_config_path}")
            
            # Load YAML directly since snapshot has flattened structure
            import yaml
            with open(training_config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Handle both flattened (snapshot) and nested (original config) structures
            def get_config_value(key, config_dict):
                # Check flattened structure first (training snapshots)
                if key in config_dict:
                    return config_dict[key]
                # Check nested structure (original training configs)
                if 'model' in config_dict and key in config_dict['model']:
                    return config_dict['model'][key]
                return None
            
            # Extract attention parameters from config (handle both structures)
            val = get_config_value('attention_enabled', config_dict)
            if val is not None:
                attention_enabled = val
            val = get_config_value('attention_type', config_dict)
            if val is not None:
                attention_type = val
            val = get_config_value('attention_heads', config_dict)
            if val is not None:
                attention_heads = val
            val = get_config_value('attention_dropout', config_dict)
            if val is not None:
                attention_dropout = val
            val = get_config_value('attention_dim', config_dict)
            if val is not None:
                attention_dim = val
            val = get_config_value('positional_encoding', config_dict)
            if val is not None:
                positional_encoding = val
            val = get_config_value('boundary_temperature', config_dict)
            if val is not None:
                boundary_temperature = val
                
            config_loaded = True
            print(f"🎯 Loaded attention config from training config:")
            print(f"   Attention enabled: {attention_enabled}")
            print(f"   Attention type: {attention_type}")
            print(f"   Attention heads: {attention_heads}")
            print(f"   Attention dimension: {attention_dim}")
            print(f"   Attention dropout: {attention_dropout}")
            print(f"   Positional encoding: {positional_encoding}")
            if attention_type == 'boundary_aware':
                print(f"   Boundary temperature: {boundary_temperature}")
            
        except Exception as e:
            print(f"⚠️  Could not load training config, falling back to state dict detection: {e}")
    
    # Fallback: detect from state dict if training config not available or failed
    if not config_loaded and attention_dim is None and attention_enabled:
        # Try to infer attention configuration from state dict
        try:
            # Detect attention dimension from the saved weights
            if 'attention.attention.w_q.weight' in state_dict:
                saved_attention_dim = state_dict['attention.attention.w_q.weight'].shape[0]
                attention_dim = saved_attention_dim
                
                # NOTE: Can't reliably infer attention_heads from weights, keep default
                print(f"   ⚠️  Could not read from config, using default attention_heads: {attention_heads}")
            
            # Check if positional encoding exists
            positional_encoding = 'attention.attention.positional_encoding.pe' in state_dict
            
            print(f"🎯 Detected attention mechanism from state dict:")
            print(f"   Attention enabled: {attention_enabled}")
            print(f"   Attention heads: {attention_heads} (default - could not read from config)")
            print(f"   Attention dimension: {attention_dim} (from state dict)")
            print(f"   Positional encoding: {positional_encoding}")
        except Exception as e:
            print(f"⚠️  Could not fully determine attention config, using defaults: {e}")
            attention_dim = None  # Fall back to None to use LSTM output dimension
    
    print(f"🔧 Detected model architecture:")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Classes: {num_classes}")
    print(f"   Layers: {num_layers}")
    if attention_enabled:
        print(f"   🎯 Attention: enabled ({attention_heads} heads)")
    else:
        print(f"   🎯 Attention: disabled")
    
    # Create model with detected parameters
    model = BLSTMTagger(
        feat_dim=input_size,
        hidden_dim=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=0.0,  # No dropout during inference
        # Attention parameters
        attention_enabled=attention_enabled,
        attention_type=attention_type,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        attention_dim=attention_dim,
        positional_encoding=positional_encoding,
        window_size=7,  # Default window size
        boundary_temperature=boundary_temperature
    )
    
    # Load weights
    try:
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully (including attention weights)")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        print("🔧 This might be due to architecture mismatch or missing attention parameters")
        raise
    
    model.to(device)
    model.eval()
    
    return model


from segmodel.data.dataset import SongsDataset

def load_dataset(file_path: str) -> SongsDataset:
    """Load data using the SongsDataset class."""
    return SongsDataset(file_path)


def extract_lines_and_labels(dataset: SongsDataset, max_lines: int = 1000) -> Tuple[List[str], List[str]]:
    """
    Extract lines and labels from a SongsDataset with improved handling.
    
    Args:
        dataset: The SongsDataset to extract from
        max_lines: Maximum number of lines to extract (to prevent positional encoding errors)
    
    Returns:
        Tuple of (lines, labels)
    """
    lines = []
    labels = []
    label_map = {0: "verse", 1: "chorus"}
    
    # Count total lines first
    total_lines = sum(len(dataset[i][1]) for i in range(len(dataset)))
    if total_lines > max_lines:
        print(f"⚠️  Warning: Dataset contains {total_lines} lines, truncating to {max_lines}")
        print(f"    This is needed due to model's positional encoding limit")
        
        # IMPROVEMENT: Calculate class distribution before truncation
        class_counts = {0: 0, 1: 0}
        for i in range(len(dataset)):
            _, _, song_labels = dataset[i]
            for label_idx in song_labels:
                class_counts[label_idx] += 1
        
        verse_ratio = class_counts[0] / total_lines
        chorus_ratio = class_counts[1] / total_lines
        print(f"    Original distribution: Verse={verse_ratio:.2%}, Chorus={chorus_ratio:.2%}")
    
    # Extract lines and labels from all songs in the dataset
    line_count = 0
    for i in range(len(dataset)):
        song_id, song_lines, song_labels = dataset[i]
        
        # Add each line and its corresponding label
        for line, label_idx in zip(song_lines, song_labels):
            # IMPROVEMENT: Better validation of label_idx
            if label_idx not in label_map:
                print(f"⚠️  Invalid label index {label_idx} found, skipping line")
                continue
                
            lines.append(line.strip() if isinstance(line, str) else str(line))
            labels.append(label_map[label_idx])
            
            line_count += 1
            if line_count >= max_lines:
                print(f"⚠️  Reached maximum line limit ({max_lines}), truncating dataset")
                
                # IMPROVEMENT: Show final distribution after truncation
                final_verse_count = sum(1 for l in labels if l == "verse")
                final_chorus_count = sum(1 for l in labels if l == "chorus")
                final_total = len(labels)
                print(f"    Final distribution: Verse={final_verse_count/final_total:.2%}, Chorus={final_chorus_count/final_total:.2%}")
                return lines, labels
    
    return lines, labels

def extract_song_sequences(dataset: SongsDataset) -> List[Tuple[List[str], List[int]]]:
    """
    Return a list of (song_lines, song_label_indices) sequences.
    Filters out ignore labels (e.g., -100) and preserves per-song boundaries.
    """
    sequences: List[Tuple[List[str], List[int]]] = []
    valid_label_indices = {0, 1}  # verse=0, chorus=1

    for i in range(len(dataset)):
        song_id, song_lines, song_labels = dataset[i]
        lines: List[str] = []
        labels: List[int] = []
        for line, lab in zip(song_lines, song_labels):
            if lab in valid_label_indices:
                lines.append(line.strip() if isinstance(line, str) else str(line))
                labels.append(int(lab))
        if lines:
            sequences.append((lines, labels))
    return sequences


def predict_with_model(
    lines: List[str],
    model: BLSTMTagger,
    feature_extractor,
    device: torch.device,
    calibration_method: str = "none",
    temperature: float = 1.0,
    platt_A: float = 1.0,
    platt_B: float = 0.0,
    batch_size: int = 128  # Process in smaller batches for memory efficiency
) -> Tuple[List[int], List[float]]:
    """
    Make predictions with model using the specified calibration method.
    
    Args:
        lines: List of text lines to predict on
        model: Trained model
        feature_extractor: Feature extraction function
        device: Device to run on
        calibration_method: Calibration method ('temperature', 'platt', 'isotonic', 'none')
        temperature: Temperature for temperature scaling
        platt_A: Platt scaling A coefficient
        platt_B: Platt scaling B coefficient
        batch_size: Number of lines to process at once (to avoid memory issues)
    """
    if not lines:
        return [], []
    
    # If we have a very large dataset, process in batches
    if len(lines) > batch_size:
        print(f"⚠️  Dataset has {len(lines)} lines, processing in batches of {batch_size}")
        
        all_predictions = []
        all_confidences = []
        
        # Process in batches
        for i in range(0, len(lines), batch_size):
            batch_end = min(i + batch_size, len(lines))
            batch_lines = lines[i:batch_end]
            
            print(f"   Processing batch {i//batch_size + 1}/{(len(lines) + batch_size - 1)//batch_size}: "
                  f"lines {i+1} to {batch_end}")
            
            batch_predictions, batch_confidences = predict_with_model(
                batch_lines, model, feature_extractor, device,
                calibration_method, temperature, platt_A, platt_B,
                batch_size=batch_size  # This won't trigger recursion since len(batch_lines) <= batch_size
            )
            
            all_predictions.extend(batch_predictions)
            all_confidences.extend(batch_confidences)
        
        return all_predictions, all_confidences
    
    try:
        # Extract features
        features = feature_extractor(lines)  # (seq_len, feature_dim)
        features = features.unsqueeze(0).to(device)  # (1, seq_len, feature_dim)
        
        # Create mask (all positions valid for inference)
        mask = torch.ones(1, len(lines), dtype=torch.bool, device=device)
        
        # Get raw model predictions with appropriate calibration
        with torch.no_grad():
            if calibration_method == 'temperature':
                # Use temperature scaling
                predictions, confidences = model.predict_with_temperature(
                    features, mask, temperature=temperature
                )
            elif calibration_method == 'platt':
                # Get raw predictions and apply Platt scaling
                logits = model(features, mask)  # (1, seq_len, num_classes)
                probs = torch.softmax(logits, dim=-1)
                max_probs, predictions = torch.max(probs, dim=-1)
                
                # Apply Platt scaling: sigmoid(A * confidence + B)
                calibrated_confidences = torch.sigmoid(platt_A * max_probs + platt_B)
                
                predictions = predictions
                confidences = calibrated_confidences
            elif calibration_method == 'isotonic':
                # For isotonic, we can't apply the calibration without the fitted model
                # Fall back to temperature=1.0 and warn user
                print("⚠️ Isotonic calibration requires fitted model (not available in inference)")
                print("   Falling back to uncalibrated predictions")
                predictions, confidences = model.predict_with_temperature(
                    features, mask, temperature=1.0
                )
            else:
                # No calibration, use temperature=1.0
                predictions, confidences = model.predict_with_temperature(
                    features, mask, temperature=1.0
                )
        
        # Convert to lists
        predictions = predictions.squeeze(0).cpu().numpy().tolist()  # (seq_len,)
        confidences = confidences.squeeze(0).cpu().numpy().tolist()  # (seq_len,)
        
        return predictions, confidences
    
    except RuntimeError as e:
        if "must match" in str(e) and "non-singleton dimension" in str(e):
            # This is likely a positional encoding dimension error
            print(f"⚠️  Positional encoding error detected: {e}")
            print(f"    This is likely due to sequence length > max_seq_length")
            print(f"    Try reducing the dataset size or using smaller batch_size")
            raise RuntimeError(f"Sequence length error: {e}")
        else:
            # Re-raise other runtime errors
            raise e

def boundary_flags(labels: Sequence[int]) -> np.ndarray:
    """
    Training-style boundary map:
    - First position is a boundary
    - Boundary when label changes vs previous
    """
    n = len(labels)
    b = np.zeros(n, dtype=bool)
    if n == 0:
        return b
    b[0] = True
    if n > 1:
        lab = np.asarray(labels, dtype=int)
        b[1:] = (lab[1:] != lab[:-1])
    return b

def prf(tp: int, fp: int, fn: int) -> Tuple[float,float,float]:
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F = 2*P*R/(P+R) if (P+R) > 0 else 0.0
    return float(P), float(R), float(F)

def wd_counts(true_b: np.ndarray, pred_b: np.ndarray, k: int) -> Tuple[int,int]:
    n = true_b.shape[0]; k = max(1, min(k, n))
    total = max(1, n - k + 1); inc = 0
    cs_t = np.concatenate([[0], np.cumsum(true_b.astype(int))])
    cs_p = np.concatenate([[0], np.cumsum(pred_b.astype(int))])
    for i in range(total):
        j = i + k
        if (cs_t[j]-cs_t[i]) != (cs_p[j]-cs_p[i]):
            inc += 1
    return inc, total

def pk_surrogate_counts(true_b: np.ndarray, pred_b: np.ndarray, k: int) -> Tuple[int,int]:
    n = true_b.shape[0]; k = max(1, min(k, n))
    total = max(1, n - k + 1); inc = 0
    cs_t = np.concatenate([[0], np.cumsum(true_b.astype(int))])
    cs_p = np.concatenate([[0], np.cumsum(pred_b.astype(int))])
    for i in range(total):
        j = i + k
        has1_t = (cs_t[j]-cs_t[i]) == 1
        has1_p = (cs_p[j]-cs_p[i]) == 1
        if has1_t != has1_p:
            inc += 1
    return inc, total

def boundary_tp_fp_fn_with_tolerance(true_b: np.ndarray, pred_b: np.ndarray, tol: int) -> Tuple[int, int, int]:
    """
    One-to-one matching between true and predicted boundaries within ±tol.
    Two-pointer sweep (O(P+T)), maximizes TP for 1D absolute-distance threshold.
    Returns (tp, fp, fn).
    """
    true_idx = np.flatnonzero(true_b)
    pred_idx = np.flatnonzero(pred_b)

    if true_idx.size == 0 and pred_idx.size == 0:
        return 0, 0, 0

    tol = max(0, int(tol))
    i = j = 0
    tp = fp = fn = 0

    # Both arrays are sorted ascending; walk them once.
    while i < true_idx.size and j < pred_idx.size:
        t = true_idx[i]
        p = pred_idx[j]
        if p < t - tol:
            # pred is too early to match this (or any earlier) true
            fp += 1
            j += 1
        elif p > t + tol:
            # true is too early to be matched by this (or any later) pred
            fn += 1
            i += 1
        else:
            # |p - t| <= tol -> match them
            tp += 1
            i += 1
            j += 1

    # Whatever remains are unmatched
    fp += (pred_idx.size - j)
    fn += (true_idx.size - i)
    return tp, fp, fn



def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def calculate_basic_metrics(y_true: List[int], y_pred: List[int], label_map: Dict[int, str]) -> Dict[str, float]:
    """
    Training-style per-class F1 over a FIXED label set (verse, chorus).
    Manual TP/FP/FN. Ignores nothing here (we filter -100 earlier).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics: Dict[str, float] = {}
    if y_true.size == 0:
        for idx, name in label_map.items():
            metrics[f"f1_{name}"] = 0.0
            metrics[f"{name}_f1"] = 0.0
        metrics["accuracy"] = 0.0
        metrics["f1_macro"] = 0.0
        metrics["f1_micro"] = 0.0
        return metrics

    metrics["accuracy"] = float((y_true == y_pred).mean())

    f1s = []
    for idx, name in sorted(label_map.items()):
        tp = int(((y_pred == idx) & (y_true == idx)).sum())
        fp = int(((y_pred == idx) & (y_true != idx)).sum())
        fn = int(((y_pred != idx) & (y_true == idx)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f"f1_{name}"] = float(f1)     # e.g., f1_verse
        metrics[f"{name}_f1"] = float(f1)     # alias
        f1s.append(f1)

    macro = float(np.mean(f1s))
    metrics["f1_macro"] = macro
    metrics["line_f1"] = macro           # <— line_f1 = macro F1 (line-level)
    metrics["f1_micro"] = metrics["accuracy"]  # micro-F1 == accuracy in single-label tagging
    return metrics


def calculate_boundary_metrics(y_true: List[int], y_pred: List[int], tolerance: int = 0) -> Dict[str, float]:
    """
    Training-style boundary detection:
    - First position is a boundary
    - Strict position match (tolerance ignored)
    """
    import numpy as np
    n = len(y_true)
    if n == 0:
        return {"boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0}

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    true_b = np.zeros(n, dtype=bool)
    pred_b = np.zeros(n, dtype=bool)
    true_b[0] = True
    pred_b[0] = True
    if n > 1:
        true_b[1:] = (y_true[1:] != y_true[:-1])
        pred_b[1:] = (y_pred[1:] != y_pred[:-1])

    tp = int(np.sum(true_b & pred_b))
    fp = int(np.sum(~true_b & pred_b))
    fn = int(np.sum(true_b & ~pred_b))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"boundary_precision": float(precision), "boundary_recall": float(recall), "boundary_f1": float(f1)}



def validate_model(
    model: BLSTMTagger,
    feature_extractor,
    validation_dataset: SongsDataset,
    device: torch.device,
    calibration_method: str = "auto",
    calibration_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Training-parity validation:
    - Per-song evaluation (no flattening)
    - Training-style line F1s (manual, fixed label set)
    - Boundary F1 with tolerances t0 (strict), t1 (±1), t2 (±2)
    - WindowDiff & Pk surrogate with fixed k=5; pos0 is a boundary
    """
    if calibration_params is None:
        calibration_params = {}

    import numpy as np  # safe local import

    label_map = {0: "verse", 1: "chorus"}
    reverse_label_map = {v: k for k, v in label_map.items()}  # kept for symmetry

    # calibration params
    temperature = calibration_params.get('temperature', 1.0)
    platt_A = calibration_params.get('A', 1.0)
    platt_B = calibration_params.get('B', 0.0)

    # ---- gather per-song predictions ----
    sequences = extract_song_sequences(validation_dataset)

    all_true: List[int] = []
    all_pred: List[int] = []
    all_conf: List[float] = []

    # boundary tallies (micro over all songs)
    tp_b = fp_b = fn_b = 0    # strict t0
    tp_b1 = fp_b1 = fn_b1 = 0 # tolerant ±1
    tp_b2 = fp_b2 = fn_b2 = 0 # tolerant ±2

    # segmentation (macro over songs)
    wd_list: List[float] = []
    pk_list: List[float] = []

    predictions_dump = []
    buckets = {"t0": [], "t1": [], "t2": []}  # per-song (tp,fp,fn)
    pk_inc = pk_tot = 0
    wd_inc = wd_tot = 0
    K = 5  # training window size
    total_lines = 0

    for s_idx, (song_lines, song_labels) in enumerate(sequences, start=1):
        # predict this song as its own sequence
        y_pred_song, conf_song = predict_with_model(
            song_lines, model, feature_extractor, device,
            calibration_method=calibration_method,
            temperature=temperature, platt_A=platt_A, platt_B=platt_B,
            batch_size=128
        )

        # aggregate token metrics
        all_true.extend(song_labels)
        all_pred.extend(y_pred_song)
        all_conf.extend(conf_song)
        total_lines += len(song_labels)

        # boundary flags and tolerant matching
        tb = boundary_flags(song_labels)
        pb = boundary_flags(y_pred_song)

        for tol, key in ((0, "t0"), (1, "t1"), (2, "t2")):
            tp, fp, fn = boundary_tp_fp_fn_with_tolerance(tb, pb, tol=tol)
            # keep your old micro tallies (optional)
            if key == "t0":
                tp_b += tp; fp_b += fp; fn_b += fn
            elif key == "t1":
                tp_b1 += tp; fp_b1 += fp; fn_b1 += fn
            else:
                tp_b2 += tp; fp_b2 += fp; fn_b2 += fn
            # NEW: store per-song counts for MACRO aggregation
            buckets[key].append((tp, fp, fn))

        # NEW: training-style segmentation (window-weighted MICRO), k=5
        inc, tot = pk_surrogate_counts(tb, pb, k=K)
        pk_inc += inc; pk_tot += tot
        inc, tot = wd_counts(tb, pb, k=K)
        wd_inc += inc; wd_tot += tot


        # per-line dump (optional)
        for ln, t, p, c in zip(song_lines, song_labels, y_pred_song, conf_song):
            predictions_dump.append({
                "line_number": len(predictions_dump) + 1,
                "line": ln,
                "true_label": "chorus" if t == 1 else "verse",
                "predicted_label": "chorus" if p == 1 else "verse",
                "confidence": float(c),
            })

    # ---- assemble metrics ----
    metrics = calculate_basic_metrics(all_true, all_pred, label_map)

    # ensure line_f1 is exposed (line-level macro F1)
    if "line_f1" not in metrics:
        metrics["line_f1"] = metrics.get("f1_macro", 0.0)

    # --- Boundary metrics: MACRO (average per-song PRF) to match training ---
    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        return p, r, f

    for key, (pname, rname, fname) in {
        "t0": ("boundary_precision_t0","boundary_recall_t0","boundary_f1_t0"),
        "t1": ("boundary_precision_t1","boundary_recall_t1","boundary_f1_t1"),
        "t2": ("boundary_precision_t2","boundary_recall_t2","boundary_f1_t2"),
    }.items():
        Ps, Rs, Fs = [], [], []
        for tp, fp, fn in buckets[key]:
            p, r, f = _prf(tp, fp, fn)
            Ps.append(p); Rs.append(r); Fs.append(f)
        metrics[pname] = float(np.mean(Ps)) if Ps else 0.0
        metrics[rname] = float(np.mean(Rs)) if Rs else 0.0
        metrics[fname] = float(np.mean(Fs)) if Fs else 0.0

    # --- Segmentation metrics: MICRO (window-weighted), training surrogate, k=5 ---
    metrics["pk"] = (pk_inc / pk_tot) if pk_tot > 0 else 1.0
    metrics["windowdif"] = (wd_inc / wd_tot) if wd_tot > 0 else 1.0


    # counts
    metrics["verse_count"]       = int(sum(1 for y in all_true if y == 0))
    metrics["chorus_count"]      = int(sum(1 for y in all_true if y == 1))
    metrics["verse_pred_count"]  = int(sum(1 for y in all_pred if y == 0))
    metrics["chorus_pred_count"] = int(sum(1 for y in all_pred if y == 1))
    metrics["validation_count"]  = int(len(all_true))

    # optional boundary counts for debugging
    metrics["boundary_counts"] = {
        "t0": {"tp": tp_b,  "fp": fp_b,  "fn": fn_b},
        "t1": {"tp": tp_b1, "fp": fp_b1, "fn": fn_b1},
        "t2": {"tp": tp_b2, "fp": fp_b2, "fn": fn_b2},
    }

    metrics["predictions"] = predictions_dump
    return metrics



def load_calibration_from_session(session_dir: str) -> Dict[str, Any]:
    """
    Load calibration parameters from training session.
    
    Args:
        session_dir: Path to training session directory
        
    Returns:
        Calibration info dictionary, or None if not found
    """
    session_path = Path(session_dir)
    calibration_file = session_path / "calibration.json"
    
    if not calibration_file.exists():
        print(f"⚠️  No calibration file found: {calibration_file}")
        return None
    
    try:
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        
        print(f"📊 Loaded calibration from: {calibration_file}")
        print(f"   Method: {calibration_data.get('method', 'unknown')}")
        if 'params' in calibration_data:
            params = calibration_data['params']
            if 'temperature' in params:
                print(f"   Temperature: {params['temperature']:.3f}")
            if 'A' in params and 'B' in params:
                print(f"   Platt coefficients: A={params['A']:.3f}, B={params['B']:.3f}")
        
        return calibration_data
        
    except Exception as e:
        print(f"⚠️  Error loading calibration: {e}")
        return None


def select_calibration_method(
    calibration_info=None,
    cli_method=None,
    cli_temp=None,
    quiet=False
):
    """
    Select calibration method and parameters based on priorities:
    1. CLI method/args (highest priority)
    2. Auto-selection from calibration_info (recommended)
    3. No calibration (fallback)
    
    Returns:
        method: Selected calibration method ('temperature', 'platt', 'isotonic', 'none')
        params: Dictionary with method-specific parameters
    """
    # Priority 1: CLI overrides everything
    if cli_method is not None:
        method = cli_method
        params = {}
        
        if method == 'temperature':
            temp = cli_temp if cli_temp is not None else 1.0
            params = {'temperature': temp}
            if not quiet:
                print(f"🔧 Using CLI calibration method: {method} (temperature: {temp:.3f})")
        elif method == 'none':
            if not quiet:
                print(f"🔧 Using CLI calibration method: none (no calibration)")
        else:
            # Fall back to auto if invalid CLI method
            method = 'auto'
    else:
        method = 'auto'
        params = {}
    
    # Handle auto mode and calibration info
    if method == 'auto':
        if calibration_info and 'method' in calibration_info:
            # Use the pre-selected best method from calibration.json
            auto_method = calibration_info['method']
            auto_params = calibration_info['params']
            
            if not quiet:
                ece_after = calibration_info.get('ece_after', 'N/A')
                print(f"📊 Auto-selected calibration method: {auto_method} (ECE: {ece_after if isinstance(ece_after, str) else f'{ece_after:.4f}'})")
                
                if auto_method == 'temperature':
                    print(f"🎯 Using calibrated temperature: {auto_params['temperature']:.3f}")
                elif auto_method == 'platt':
                    print(f"🎯 Using calibrated Platt scaling: A={auto_params['A']:.3f}, B={auto_params['B']:.3f}")
                elif auto_method == 'isotonic':
                    print(f"🎯 Using isotonic calibration: knots={auto_params.get('knots', 0)}")
            
            return auto_method, auto_params
        
        # No calibration info available for auto-selection
        if not quiet:
            print(f"⚠️  'auto' method requires calibration.json but none found")
            print(f"    Falling back to 'none' (no calibration applied)")
        method = 'none'
        params = {}
    
    return method, params


def main():
    parser = argparse.ArgumentParser(
        description="Validate model performance with comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main arguments
    parser.add_argument('--session', required=True, help='Path to training session directory')
    parser.add_argument('--validation-file', required=True, help='Path to validation data file (JSONL format)')
    parser.add_argument('--output-dir', help='Directory to save validation results (default: session_dir/validation)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    
    # Calibration arguments
    parser.add_argument('--calibration-method', choices=['auto', 'temperature', 'platt', 'none'], 
                       help='Calibration method (auto uses best from calibration.json)')
    parser.add_argument('--temperature', type=float, help='Temperature scaling value (overrides calibration.json)')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🔍 Model Validation")
        print("=" * 30)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    if not args.quiet:
        print(f"🔧 Using device: {device}")
    
    # Validate session directory
    session_dir = Path(args.session)
    if not session_dir.exists():
        print(f"❌ Session directory not found: {session_dir}")
        return 1
    
    # Find model path (best_model.pt or last_model.pt)
    model_path = session_dir / "best_model.pt"
    if not model_path.exists():
        model_path = session_dir / "last_model.pt"
        if not model_path.exists():
            print(f"❌ No model found in session directory: {session_dir}")
            print("   Expected either best_model.pt or last_model.pt")
            return 1
    
    # Find training config
    config_path = session_dir / "training_config_snapshot.yaml"
    if not config_path.exists():
        print(f"⚠️  No training config snapshot found: {config_path}")
        print("   Feature extraction may not match training settings")
        config_path = None
    
    # Load validation data
    try:
        validation_dataset = load_dataset(args.validation_file)
        if not args.quiet:
            print(f"📊 Loaded validation data from: {args.validation_file}")
            print(f"   Songs: {validation_dataset.stats['num_songs']}")
            print(f"   Total lines: {validation_dataset.stats['total_lines']}")
            print(f"   Chorus: {validation_dataset.stats['chorus_lines']} ({validation_dataset.stats['chorus_ratio']:.2%})")
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        return 1
    
    # Create feature extractor and load model
    try:
        # Find training config for better feature extraction
        training_config_path = session_dir / "training_config_snapshot.yaml"
        
        # Load model first to get its expected input dimensions
        model = load_model(str(model_path), device, str(config_path) if config_path else None)
        
        # For session-based validation, create feature extractor directly from training config
        if training_config_path.exists():
            print(f"🔧 Using feature configuration from training session config")
            # Import the function from predict_baseline.py
            from predict_baseline import create_feature_extractor_from_training_config
            feature_extractor = create_feature_extractor_from_training_config(str(training_config_path))
            print(f"   Feature dimension: {feature_extractor.get_feature_dimension()}")
        else:
            # Fallback to prediction config (less reliable)
            print(f"⚠️ No training config found, using prediction config (may cause dimension mismatch)")
            pred_config, _ = create_prediction_config_from_training_session(str(session_dir))
            feature_extractor = get_feature_extractor_from_config(pred_config)
            
        # Load calibration data
        calibration_info = load_calibration_from_session(str(session_dir))
        
        # Select calibration method and parameters
        calibration_method, calibration_params = select_calibration_method(
            calibration_info=calibration_info,
            cli_method=args.calibration_method,
            cli_temp=args.temperature,
            quiet=args.quiet
        )
    except Exception as e:
        print(f"❌ Error setting up model and features: {e}")
        return 1
    
    # Run validation
    if not args.quiet:
        print(f"🔍 Running validation...")
    
    metrics = validate_model(
        model, 
        feature_extractor, 
        validation_dataset, 
        device, 
        calibration_method=calibration_method,
        calibration_params=calibration_params
    )
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = session_dir / "validation"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create timestamp for output files
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to JSON file
    output_file = output_dir / f"validation_metrics_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create a summary text file with key metrics
    summary_file = output_dir / f"validation_summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Model Validation Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Session: {session_dir}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Validation data: {args.validation_file}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Basic Metrics:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Macro: {metrics['f1_macro']:.4f}\n")
        f.write(f"  F1 Micro: {metrics['f1_micro']:.4f}\n\n")
        
        f.write(f"Class Metrics:\n")
        f.write(f"  Chorus - F1: {metrics['f1_chorus']:.4f}\n")
        f.write(f"  Verse  - F1: {metrics['f1_verse']:.4f}\n\n")
        
        f.write(f"Segmentation Metrics:\n")
        f.write(f"  Pk: {metrics['pk']:.4f}\n")
        f.write(f"  WindowDiff: {metrics['windowdif']:.4f}\n\n")
        
        f.write(f"Boundary Metrics:\n")
        f.write(f"  Strict (t=0) - F1: {metrics['boundary_f1_t0']:.4f}, Precision: {metrics['boundary_precision_t0']:.4f}, Recall: {metrics['boundary_recall_t0']:.4f}\n")
        f.write(f"  Relaxed (t=1) - F1: {metrics['boundary_f1_t1']:.4f}, Precision: {metrics['boundary_precision_t1']:.4f}, Recall: {metrics['boundary_recall_t1']:.4f}\n")
        f.write(f"  Relaxed (t=2) - F1: {metrics['boundary_f1_t2']:.4f}, Precision: {metrics['boundary_precision_t2']:.4f}, Recall: {metrics['boundary_recall_t2']:.4f}\n\n")
        
        # Create a simple confusion matrix manually
        # We don't have direct access to y_true/y_pred here, so just show class distribution
        verse_count = metrics.get("verse_count", 0)  # We'll add these to metrics
        chorus_count = metrics.get("chorus_count", 0)
        verse_pred_count = metrics.get("verse_pred_count", 0)
        chorus_pred_count = metrics.get("chorus_pred_count", 0)
        
        # Create distribution summary instead of confusion matrix
        f.write(f"Class Distribution:\n")
        f.write(f"  Ground Truth: {verse_count} Verse, {chorus_count} Chorus\n")
        f.write(f"  Predicted:    {verse_pred_count} Verse, {chorus_pred_count} Chorus\n\n")
        
        f.write(f"Full metrics saved to: {output_file}\n")
    
    if not args.quiet:
        # Print key metrics
        print("\n📊 Validation Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"  F1 Micro: {metrics['f1_micro']:.4f}")
        print(f"  Chorus F1: {metrics['f1_chorus']:.4f}")
        print(f"  Verse F1: {metrics['f1_verse']:.4f}")
        print(f"  Pk: {metrics['pk']:.4f}")
        print(f"  WindowDiff: {metrics['windowdif']:.4f}")
        print(f"  Boundary F1 (t=0): {metrics['boundary_f1_t0']:.4f}")
        print(f"  Boundary F1 (t=1): {metrics['boundary_f1_t1']:.4f}")
        print(f"  Boundary F1 (t=2): {metrics['boundary_f1_t2']:.4f}")
        print(f"\n💾 Results saved to:")
        print(f"   {output_file}")
        print(f"   {summary_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
