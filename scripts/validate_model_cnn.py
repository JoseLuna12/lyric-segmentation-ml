#!/usr/bin/env python3
"""
Validation script for CNN-based taggers.

This script provides comprehensive validation metrics for CNN models:
- Per-song evaluation (maintaining song boundaries) 
- Line-level F1 metrics (overall and per-class)
- Boundary detection metrics with tolerances (t0/t1/t2)
- Pk surrogate + WindowDiff with k=5 (window-weighted micro)
- Support for temperature/Platt calibration

Usage:
    python validate_model_cnn.py --session <cnn_session_dir> --validation-file <val_jsonl> [options]
    
Works with CNN session folders that contain:
- best_cnn_model.pt or last_cnn_model.pt 
- training_config_snapshot.yaml
- calibration.json (optional)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Sequence

import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Project imports
from segmodel.data.dataset import SongsDataset
from segmodel.features import FeatureExtractor
from segmodel.utils.prediction_config import (
    create_prediction_config_from_training_session,
    get_feature_extractor_from_config,
)

from segmodel.models import CNNTagger


# -----------------------------
# Data helpers
# -----------------------------
def load_dataset(file_path: str) -> SongsDataset:
    return SongsDataset(file_path)


def extract_song_sequences(dataset: SongsDataset) -> List[Tuple[List[str], List[int]]]:
    """
    Return a list of (song_lines, song_label_indices) sequences.
    Filters out ignore labels (e.g., -100) and preserves per-song boundaries.
    """
    sequences: List[Tuple[List[str], List[int]]] = []
    valid = {0, 1}
    for i in range(len(dataset)):
        _, song_lines, song_labels = dataset[i]
        lines, labels = [], []
        for line, lab in zip(song_lines, song_labels):
            if lab in valid:
                lines.append(line.strip() if isinstance(line, str) else str(line))
                labels.append(int(lab))
        if lines:
            sequences.append((lines, labels))
    return sequences


# -----------------------------
# CNN model loading
# -----------------------------
def _read_yaml_config(training_config_path: str) -> Dict[str, Any]:
    if not training_config_path or not os.path.exists(training_config_path):
        return {}
    try:
        import yaml
        with open(training_config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _cfg_get(cfg: Dict[str, Any], key: str, default=None):
    if key in cfg:
        return cfg[key]
    if isinstance(cfg.get("model"), dict) and key in cfg["model"]:
        return cfg["model"][key]
    return default


def load_cnn_model(model_path: str, device: torch.device, training_config_path: str = None) -> CNNTagger:
    """
    Load a CNNTagger with hyperparams from training_config_snapshot.yaml when possible,
    otherwise infer minimal shapes from the state_dict.
    """
    print(f"📦 Loading CNN model from {model_path}...")
    state = torch.load(model_path, map_location=device)
    
    # Handle different ways the model weights might be saved
    if isinstance(state, dict):
        # Case 1: The state is a state_dict itself (most common)
        if "lstm.weight_ih_l0" not in state and "classifier.weight" in state:
            state_dict = state
            print("   Model saved as direct state_dict")
        # Case 2: The state contains a "state_dict" key (with PyTorch Lightning or checkpoints)
        elif "state_dict" in state:
            state_dict = state["state_dict"]
            print("   Model saved with 'state_dict' key (checkpoint format)")
        # Case 3: It's an entire model instance dictionary
        else:
            state_dict = state
            print("   Model saved as dictionary (full save format)")
    else:
        # Unexpected format - try to continue anyway
        state_dict = state
        print("⚠️  Unexpected model save format - attempting to load anyway")

    cfg = _read_yaml_config(training_config_path)

    # Infer dimensions from state_dict if needed
    try:
        # Get number of classes from classifier weights
        if "classifier.weight" in state_dict:
            num_classes = state_dict["classifier.weight"].shape[0]
            print(f"   Detected num_classes: {num_classes} (from classifier.weight)")
        elif "output.weight" in state_dict:
            num_classes = state_dict["output.weight"].shape[0]
            print(f"   Detected num_classes: {num_classes} (from output.weight)")
        else:
            # Default to 2 classes if can't detect from state_dict
            num_classes = _cfg_get(cfg, "num_classes", 2)
            print(f"⚠️  Could not detect num_classes from state_dict, using config or default: {num_classes}")
        
        # Try to infer hidden dimension if not specified in config
        hidden_dim_detected = False
        if "input_projection.weight" in state_dict:
            detected_hidden_dim = state_dict["input_projection.weight"].shape[0]
            hidden_dim_detected = True
            print(f"   Detected hidden_dim: {detected_hidden_dim} (from input_projection.weight)")
        elif "cnn_blocks.0.convs.0.weight" in state_dict:
            # May need to divide by number of kernels depending on model structure
            detected_hidden_dim = state_dict["cnn_blocks.0.convs.0.weight"].shape[0]
            hidden_dim_detected = True
            print(f"   Detected hidden_dim: {detected_hidden_dim} (from cnn_blocks.0.convs.0.weight)")
    except Exception as e:
        print(f"⚠️  Error detecting dimensions from state_dict: {e}")
        print("   Falling back to config or default values")

    # Get feature dimension 
    # First priority: Config file
    feat_dim = _cfg_get(cfg, "feat_dim", None)
    
    # Second priority: Try to detect from state dictionary
    if feat_dim is None:
        try:
            if "input_projection.weight" in state_dict:
                # Input projection weight has shape (hidden_dim, feat_dim)
                detected_feat_dim = state_dict["input_projection.weight"].shape[1]
                print(f"   Detected feature dimension: {detected_feat_dim} (from input_projection.weight)")
                feat_dim = detected_feat_dim
        except Exception as e:
            print(f"⚠️  Could not detect feature dimension from state dict: {e}")
    
    # If still None, this will be inferred later from the feature extractor

    # CNN hyperparams with safe defaults (override from cfg if available)
    num_layers = _cfg_get(cfg, "num_layers", 3)
    # Use detected hidden_dim if available, otherwise from config or default
    if 'hidden_dim_detected' in locals() and hidden_dim_detected:
        # Config takes precedence if explicitly specified
        config_hidden_dim = _cfg_get(cfg, "hidden_dim", None)
        if config_hidden_dim is not None:
            if config_hidden_dim != detected_hidden_dim:
                print(f"⚠️  Hidden dimension mismatch: config={config_hidden_dim}, detected={detected_hidden_dim}")
                print(f"   Using config value: {config_hidden_dim}")
            hidden_dim = config_hidden_dim
        else:
            hidden_dim = detected_hidden_dim
    else:
        hidden_dim = _cfg_get(cfg, "hidden_dim", 128)
    kernel_sizes = _cfg_get(cfg, "kernel_sizes", [3, 5, 7])
    # CNN-specific params
    cnn_kernel_sizes = _cfg_get(cfg, "cnn_kernel_sizes", kernel_sizes)
    dilation_rates = _cfg_get(cfg, "dilation_rates", [1, 2, 4]) 
    cnn_dilation_rates = _cfg_get(cfg, "cnn_dilation_rates", dilation_rates)
    use_residual = _cfg_get(cfg, "use_residual", True)
    cnn_use_residual = _cfg_get(cfg, "cnn_use_residual", use_residual)
    
    # Dropout parameters
    dropout = _cfg_get(cfg, "dropout", 0.3)
    layer_dropout = _cfg_get(cfg, "layer_dropout", 0.0)
    
    # Attention parameters
    attention_enabled = _cfg_get(cfg, "attention_enabled", False)
    attention_type = _cfg_get(cfg, "attention_type", 'self')
    attention_heads = _cfg_get(cfg, "attention_heads", 8)
    attention_dropout = _cfg_get(cfg, "attention_dropout", 0.1)
    attention_dim = _cfg_get(cfg, "attention_dim", None)
    positional_encoding = _cfg_get(cfg, "positional_encoding", True)
    max_seq_length = _cfg_get(cfg, "max_seq_length", 1000)
    window_size = _cfg_get(cfg, "window_size", 7)
    boundary_temperature = _cfg_get(cfg, "boundary_temperature", 2.0)

    print(f"🔧 CNN Model parameters:")
    print(f"   Layers: {num_layers}, Hidden dim: {hidden_dim}")
    print(f"   Kernel sizes: {cnn_kernel_sizes}, Dilation rates: {cnn_dilation_rates}")
    print(f"   Residual connections: {cnn_use_residual}")
    print(f"   Attention enabled: {attention_enabled}")
    if attention_enabled:
        print(f"   Attention type: {attention_type}, Heads: {attention_heads}")

    # Create model
    model = CNNTagger(
        feat_dim=feat_dim,  # most CNNs in this project take (B, T, feat_dim)
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        layer_dropout=layer_dropout,
        attention_enabled=attention_enabled,
        attention_type=attention_type,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        attention_dim=attention_dim,
        positional_encoding=positional_encoding,
        max_seq_length=max_seq_length,
        window_size=window_size,
        boundary_temperature=boundary_temperature,
        kernel_sizes=cnn_kernel_sizes,
        dilation_rates=cnn_dilation_rates,
        use_residual=cnn_use_residual
    )

    # Load weights (non-strict in case of auxiliary buffers)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️  load_state_dict non-strict:\n    missing: {missing}\n    unexpected: {unexpected}")

    model.to(device)
    model.eval()
    print("✅ CNN model loaded")
    return model


# -----------------------------
# Prediction (mirrors BLSTM validator)
# -----------------------------
def predict_with_model(
    lines: List[str],
    model: CNNTagger,
    feature_extractor: FeatureExtractor,
    device: torch.device,
    calibration_method: str = "none",
    temperature: float = 1.0,
    platt_A: float = 1.0,
    platt_B: float = 0.0,
    batch_size: int = 128,
) -> Tuple[List[int], List[float]]:
    if not lines:
        return [], []

    if len(lines) > batch_size:
        all_preds, all_confs = [], []
        for i in range(0, len(lines), batch_size):
            b = lines[i : i + batch_size]
            preds, confs = predict_with_model(
                b, model, feature_extractor, device,
                calibration_method, temperature, platt_A, platt_B, batch_size=batch_size
            )
            all_preds.extend(preds)
            all_confs.extend(confs)
        return all_preds, all_confs

    feats = feature_extractor(lines).unsqueeze(0).to(device)  # (1, T, F)
    mask = torch.ones(1, feats.shape[1], dtype=torch.bool, device=device)

    with torch.no_grad():
        if calibration_method == "temperature":
            if hasattr(model, "predict_with_temperature"):
                preds, confs = model.predict_with_temperature(feats, mask, temperature=float(temperature))
            else:
                logits = model(feats, mask)
                logits = logits / float(temperature)
                probs = torch.softmax(logits, dim=-1)
                confs, preds = torch.max(probs, dim=-1)
        elif calibration_method == "platt":
            logits = model(feats, mask)
            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)
            confs = torch.sigmoid(platt_A * max_probs + platt_B)
        else:
            if hasattr(model, "predict_with_temperature"):
                preds, confs = model.predict_with_temperature(feats, mask, temperature=1.0)
            else:
                logits = model(feats, mask)
                probs = torch.softmax(logits, dim=-1)
                confs, preds = torch.max(probs, dim=-1)

    return preds.squeeze(0).cpu().tolist(), confs.squeeze(0).cpu().tolist()


# -----------------------------
# Metric helpers (identical to BLSTM validator)
# -----------------------------
def boundary_flags(labels: Sequence[int]) -> np.ndarray:
    n = len(labels)
    b = np.zeros(n, dtype=bool)
    if n == 0:
        return b
    b[0] = True
    if n > 1:
        lab = np.asarray(labels, dtype=int)
        b[1:] = (lab[1:] != lab[:-1])
    return b


def boundary_tp_fp_fn_with_tolerance(true_b: np.ndarray, pred_b: np.ndarray, tol: int) -> Tuple[int, int, int]:
    true_idx = np.flatnonzero(true_b)
    pred_idx = np.flatnonzero(pred_b)
    if true_idx.size == 0 and pred_idx.size == 0:
        return 0, 0, 0
    tol = max(0, int(tol))
    i = j = 0
    tp = fp = fn = 0
    while i < true_idx.size and j < pred_idx.size:
        t = true_idx[i]; p = pred_idx[j]
        if p < t - tol:
            fp += 1; j += 1
        elif p > t + tol:
            fn += 1; i += 1
        else:
            tp += 1; i += 1; j += 1
    fp += pred_idx.size - j
    fn += true_idx.size - i
    return tp, fp, fn   


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def wd_counts(true_b: np.ndarray, pred_b: np.ndarray, k: int) -> Tuple[int, int]:
    n = true_b.shape[0]; k = max(1, min(k, n))
    total = max(1, n - k + 1); inc = 0
    cs_t = np.concatenate([[0], np.cumsum(true_b.astype(int))])
    cs_p = np.concatenate([[0], np.cumsum(pred_b.astype(int))])
    for i in range(total):
        j = i + k
        if (cs_t[j] - cs_t[i]) != (cs_p[j] - cs_p[i]):
            inc += 1
    return inc, total


def pk_surrogate_counts(true_b: np.ndarray, pred_b: np.ndarray, k: int) -> Tuple[int, int]:
    n = true_b.shape[0]; k = max(1, min(k, n))
    total = max(1, n - k + 1); inc = 0
    cs_t = np.concatenate([[0], np.cumsum(true_b.astype(int))])
    cs_p = np.concatenate([[0], np.cumsum(pred_b.astype(int))])
    for i in range(total):
        j = i + k
        has1_t = (cs_t[j] - cs_t[i]) == 1
        has1_p = (cs_p[j] - cs_p[i]) == 1
        if has1_t != has1_p:
            inc += 1
    return inc, total


def calculate_basic_metrics(y_true: List[int], y_pred: List[int], label_map: Dict[int, str]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    metrics: Dict[str, float] = {}
    if y_true.size == 0:
        for idx, name in label_map.items():
            metrics[f"f1_{name}"] = 0.0
            metrics[f"{name}_f1"] = 0.0
        metrics["accuracy"] = 0.0
        metrics["f1_macro"] = 0.0
        metrics["line_f1"] = 0.0
        metrics["f1_micro"] = 0.0
        return metrics

    metrics["accuracy"] = float((y_true == y_pred).mean())

    f1s = []
    for idx in sorted(label_map.keys()):
        name = label_map[idx]
        tp = int(((y_pred == idx) & (y_true == idx)).sum())
        fp = int(((y_pred == idx) & (y_true != idx)).sum())
        fn = int(((y_pred != idx) & (y_true == idx)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[f"f1_{name}"] = float(f1)
        metrics[f"{name}_f1"] = float(f1)
        f1s.append(f1)

    macro = float(np.mean(f1s))
    metrics["f1_macro"] = macro
    metrics["line_f1"] = macro
    metrics["f1_micro"] = metrics["accuracy"]
    return metrics


# -----------------------------
# Validation (same flow as BLSTM)
# -----------------------------
def validate_model(
    model: CNNTagger,
    feature_extractor: FeatureExtractor,
    validation_dataset: SongsDataset,
    device: torch.device,
    calibration_method: str = "auto",
    calibration_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    if calibration_params is None:
        calibration_params = {}

    label_map = {0: "verse", 1: "chorus"}
    temperature = float(calibration_params.get("temperature", 1.0))
    platt_A = float(calibration_params.get("A", 1.0))
    platt_B = float(calibration_params.get("B", 0.0))

    sequences = extract_song_sequences(validation_dataset)

    all_true: List[int] = []
    all_pred: List[int] = []
    all_conf: List[float] = []

    # Boundary macro buckets + segmentation micro accumulators
    buckets = {"t0": [], "t1": [], "t2": []}  # per-song (tp, fp, fn)
    pk_inc = pk_tot = 0
    wd_inc = wd_tot = 0
    K = 5

    predictions_dump = []

    for song_lines, song_labels in sequences:
        y_pred_song, conf_song = predict_with_model(
            song_lines, model, feature_extractor, device,
            calibration_method=calibration_method,
            temperature=temperature, platt_A=platt_A, platt_B=platt_B,
            batch_size=128
        )

        all_true.extend(song_labels)
        all_pred.extend(y_pred_song)
        all_conf.extend(conf_song)

        tb = boundary_flags(song_labels)
        pb = boundary_flags(y_pred_song)

        for tol, key in ((0, "t0"), (1, "t1"), (2, "t2")):
            tp, fp, fn = boundary_tp_fp_fn_with_tolerance(tb, pb, tol=tol)
            buckets[key].append((tp, fp, fn))

        # training-style segmentation (surrogate), window-weighted micro
        inc, tot = pk_surrogate_counts(tb, pb, k=K)
        pk_inc += inc; pk_tot += tot
        inc, tot = wd_counts(tb, pb, k=K)
        wd_inc += inc; wd_tot += tot

        for ln, t, p, c in zip(song_lines, song_labels, y_pred_song, conf_song):
            predictions_dump.append({
                "line_number": len(predictions_dump) + 1,
                "line": ln,
                "true_label": "chorus" if t == 1 else "verse",
                "predicted_label": "chorus" if p == 1 else "verse",
                "confidence": float(c),
            })

    metrics = calculate_basic_metrics(all_true, all_pred, label_map)
    if "line_f1" not in metrics:
        metrics["line_f1"] = metrics.get("f1_macro", 0.0)

    # Boundary MACRO (per-song)
    for key, (pname, rname, fname) in {
        "t0": ("boundary_precision_t0", "boundary_recall_t0", "boundary_f1_t0"),
        "t1": ("boundary_precision_t1", "boundary_recall_t1", "boundary_f1_t1"),
        "t2": ("boundary_precision_t2", "boundary_recall_t2", "boundary_f1_t2"),
    }.items():
        Ps, Rs, Fs = [], [], []
        for tp, fp, fn in buckets[key]:
            p, r, f = prf_from_counts(tp, fp, fn)
            Ps.append(p); Rs.append(r); Fs.append(f)
        metrics[pname] = float(np.mean(Ps)) if Ps else 0.0
        metrics[rname] = float(np.mean(Rs)) if Rs else 0.0
        metrics[fname] = float(np.mean(Fs)) if Fs else 0.0

    # Segmentation MICRO (window-weighted)
    metrics["pk"] = (pk_inc / pk_tot) if pk_tot > 0 else 1.0
    metrics["windowdif"] = (wd_inc / wd_tot) if wd_tot > 0 else 1.0

    # Counts
    metrics["verse_count"]       = int(sum(1 for y in all_true if y == 0))
    metrics["chorus_count"]      = int(sum(1 for y in all_true if y == 1))
    metrics["verse_pred_count"]  = int(sum(1 for y in all_pred if y == 0))
    metrics["chorus_pred_count"] = int(sum(1 for y in all_pred if y == 1))
    metrics["validation_count"]  = int(len(all_true))

    metrics["predictions"] = predictions_dump
    return metrics


# -----------------------------
# Calibration utils (same idea)
# -----------------------------
def load_calibration_from_session(session_dir: str) -> Dict[str, Any]:
    p = Path(session_dir) / "calibration.json"
    if not p.exists():
        print(f"⚠️  No calibration file found: {p}")
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading calibration: {e}")
        return None


def select_calibration_method(calibration_info=None, cli_method=None, cli_temp=None, quiet=False):
    if cli_method is not None:
        if cli_method == "temperature":
            t = cli_temp if cli_temp is not None else 1.0
            if not quiet:
                print(f"🔧 Using CLI calibration: temperature={t:.3f}")
            return "temperature", {"temperature": t}
        elif cli_method == "platt":
            if not quiet:
                print("🔧 Using CLI calibration: platt (expects A/B in calibration.json)")
            return "platt", {}
        elif cli_method == "none":
            if not quiet:
                print("🔧 Using CLI calibration: none")
            return "none", {}
        # fall through to auto

    if calibration_info and "method" in calibration_info:
        m = calibration_info["method"]
        params = calibration_info.get("params", {})
        if not quiet:
            print(f"📊 Auto-selected calibration: {m}")
        return m, params

    if not quiet:
        print("⚠️  No calibration info; using 'none'")
    return "none", {}


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validate CNN model with comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a CNN model with auto-calibration
  python validate_model_cnn.py --session training_sessions/cnn_session_*/ --validation-file data/val.jsonl
  
  # Use specific calibration method
  python validate_model_cnn.py --session training_sessions/cnn_session_*/ --validation-file data/val.jsonl --calibration-method temperature
  
  # Use specific temperature value
  python validate_model_cnn.py --session training_sessions/cnn_session_*/ --validation-file data/val.jsonl --calibration-method temperature --temperature 1.5
  
  # Custom output directory
  python validate_model_cnn.py --session training_sessions/cnn_session_*/ --validation-file data/val.jsonl --output-dir results/cnn_validation
        """
    )
    parser.add_argument('--session', required=True, help='Path to CNN training session directory (contains best_cnn_model.pt)')
    parser.add_argument('--validation-file', required=True, help='Path to validation data file (JSONL format)')
    parser.add_argument('--output-dir', help='Directory to save validation results (default: session_dir/validation)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    parser.add_argument('--calibration-method', choices=['auto', 'temperature', 'platt', 'none'], 
                        help='Calibration method (auto uses best from calibration.json)')
    parser.add_argument('--temperature', type=float, help='Temperature scaling value (for --calibration-method temperature)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    if not args.quiet:
        print(f"🔧 Using device: {device}")

    session_dir = Path(args.session)
    if not session_dir.exists():
        print(f"❌ Session directory not found: {session_dir}")
        return 1

    # Choose model file
    model_path = session_dir / "best_cnn_model.pt"
    if not model_path.exists():
        model_path = session_dir / "last_cnn_model.pt"
        if not model_path.exists():
            print(f"❌ No model found in {session_dir}")
            return 1

    # Training snapshot config (for features + cnn hyperparams)
    cfg_path = session_dir / "training_config_snapshot.yaml"
    if not cfg_path.exists():
        print(f"⚠️  No training config snapshot found: {cfg_path}")
        cfg_path = None

    # Load validation dataset
    try:
        val_ds = load_dataset(args.validation_file)
        if not args.quiet:
            print(f"📊 Loaded validation data from: {args.validation_file}")
            print(f"   Songs: {val_ds.stats['num_songs']}")
            print(f"   Total lines: {val_ds.stats['total_lines']}")
            print(f"   Chorus: {val_ds.stats['chorus_lines']} ({val_ds.stats['chorus_ratio']:.2%})")
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        return 1

    # Build feature extractor
    try:
        if cfg_path and cfg_path.exists():
            print("🔧 Building feature extractor from training snapshot...")
            from predict_baseline import create_feature_extractor_from_training_config
            feat_extractor = create_feature_extractor_from_training_config(str(cfg_path))
        else:
            print("⚠️ No training config; using prediction-config fallback")
            pred_config, _ = create_prediction_config_from_training_session(str(session_dir))
            feat_extractor = get_feature_extractor_from_config(pred_config)
        print(f"   Feature dim: {feat_extractor.get_feature_dimension()}")
    except Exception as e:
        print(f"❌ Error creating feature extractor: {e}")
        return 1

    # Load CNN model with feature_extractor's dimension if needed
    try:
        model = load_cnn_model(str(model_path), device, str(cfg_path) if cfg_path else None)
        
        # Update model's feature dimension from feature_extractor if needed
        if hasattr(model, 'feat_dim') and model.feat_dim is None:
            if hasattr(feat_extractor, 'get_feature_dimension'):
                extractor_dim = feat_extractor.get_feature_dimension()
                print(f"🔄 Updating CNN model's feature dimension from extractor: {extractor_dim}")
                model.feat_dim = extractor_dim
                
                # Recreate input projection if needed
                if hasattr(model, 'input_projection') and model.input_projection is not None:
                    if model.input_projection.in_features != extractor_dim:
                        print(f"⚠️  Recreating input projection layer to match feature dimension")
                        model.input_projection = torch.nn.Linear(extractor_dim, model.hidden_dim)
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")
        return 1

    # Calibration
    calib_info = load_calibration_from_session(str(session_dir))
    calib_method, calib_params = select_calibration_method(
        calibration_info=calib_info,
        cli_method=args.calibration_method,
        cli_temp=args.temperature,
        quiet=args.quiet
    )

    if not args.quiet:
        print("🔍 Running validation...")

    metrics = validate_model(
        model=model,
        feature_extractor=feat_extractor,
        validation_dataset=val_ds,
        device=device,
        calibration_method=calib_method,
        calibration_params=calib_params
    )

    # Output
    out_dir = Path(args.output_dir) if args.output_dir else (session_dir / "validation")
    out_dir.mkdir(exist_ok=True, parents=True)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"validation_metrics_cnn_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    summary_path = out_dir / f"validation_summary_cnn_{ts}.txt"
    with open(summary_path, "w") as f:
        f.write("Model Validation Summary (CNN)\n")
        f.write("===============================\n\n")
        f.write(f"Session: {session_dir}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Validation data: {args.validation_file}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Basic Metrics:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Macro: {metrics['f1_macro']:.4f}\n")
        f.write(f"  F1 Micro: {metrics['f1_micro']:.4f}\n")
        f.write(f"  Line F1 : {metrics['line_f1']:.4f}\n\n")

        f.write("Class Metrics:\n")
        f.write(f"  Chorus - F1: {metrics['f1_chorus']:.4f}\n")
        f.write(f"  Verse  - F1: {metrics['f1_verse']:.4f}\n\n")

        f.write("Segmentation Metrics:\n")
        f.write(f"  Pk: {metrics['pk']:.4f}\n")
        f.write(f"  WindowDiff: {metrics['windowdif']:.4f}\n\n")

        f.write("Boundary Metrics:\n")
        f.write(f"  Strict (t=0) - F1: {metrics['boundary_f1_t0']:.4f}, Precision: {metrics['boundary_precision_t0']:.4f}, Recall: {metrics['boundary_recall_t0']:.4f}\n")
        f.write(f"  Relaxed (t=1) - F1: {metrics['boundary_f1_t1']:.4f}, Precision: {metrics['boundary_precision_t1']:.4f}, Recall: {metrics['boundary_recall_t1']:.4f}\n")
        f.write(f"  Relaxed (t=2) - F1: {metrics['boundary_f1_t2']:.4f}, Precision: {metrics['boundary_precision_t2']:.4f}, Recall: {metrics['boundary_recall_t2']:.4f}\n\n")

        verse_count = metrics.get("verse_count", 0)
        chorus_count = metrics.get("chorus_count", 0)
        verse_pred_count = metrics.get("verse_pred_count", 0)
        chorus_pred_count = metrics.get("chorus_pred_count", 0)
        f.write("Class Distribution:\n")
        f.write(f"  Ground Truth: {verse_count} Verse, {chorus_count} Chorus\n")
        f.write(f"  Predicted:    {verse_pred_count} Verse, {chorus_pred_count} Chorus\n\n")

        f.write(f"Full metrics saved to: {json_path}\n")

    if not args.quiet:
        print("\n📊 Validation Results (CNN):")
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
        print(f"\n💾 Results saved to:\n   {json_path}\n   {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
