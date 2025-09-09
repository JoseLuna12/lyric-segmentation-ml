import os
import sys
from pathlib import Path
import html
import streamlit as st


# ---- Session choices ----
# Assumes session directories contain:
# - model checkpoint file
# - training_config_snapshot.yaml
# - calibration.json
SESSION_OPTIONS = {
    "Robust Model": "/Users/joseluna/master/Tesis/lyric-segmentation-ml/training_sessions/session_20250908_142738_all_features_boundary_aware_loss_tuned_best_optuna",
    "Lite Mode": "/Users/joseluna/master/Tesis/lyric-segmentation-ml/training_sessions/session_20250908_154356_all_features_boundary_aware_loss_tuned_best_optuna_lite",
}
DEFAULT_SESSION_LABEL = "Robust Model"


## Ensure repository root is importable (for predict_baseline, segmodel, etc.)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@st.cache_resource(show_spinner=False)
def _boot_inference_state(session_dir: str):
    """Load and cache model + calibration for the given session directory."""
    import torch
    from predict_baseline import (
        load_model as baseline_load_model,
        load_calibration_from_session,
        select_calibration_method,
    )
    from segmodel.utils.prediction_config import (
        create_prediction_config_from_training_session,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    )

    # Minimal, robust-enough discovery using existing helper
    pred_config, model_path = create_prediction_config_from_training_session(session_dir)

    # Training config is expected to exist alongside the session
    training_config_path = os.path.join(session_dir, "training_config_snapshot.yaml")
    if not os.path.exists(training_config_path):
        training_config_path = None

    # Load model
    model = baseline_load_model(
        model_path=model_path,
        device=device,
        training_config_path=training_config_path,
    )

    # Load calibration and select final method/params (auto from session)
    calibration_info = load_calibration_from_session(session_dir)
    cal_method, cal_params = select_calibration_method(
        calibration_info=calibration_info,
        config_method="auto",
        quiet=True,
    )

    # Prepare feature extractor from training config (assumed present)
    feature_extractor = None
    feature_dim = None
    if training_config_path:
        from predict_baseline import create_feature_extractor_from_training_config
        feature_extractor = create_feature_extractor_from_training_config(training_config_path)
        if hasattr(feature_extractor, "get_feature_dimension"):
            try:
                feature_dim = feature_extractor.get_feature_dimension()
            except Exception:
                feature_dim = None

    return {
        "model": model,
        "device": str(device),
        "session_dir": str(session_dir),
        "model_path": str(model_path),
        "calibration_method": cal_method,
        "calibration_params": cal_params,
        "training_config_path": training_config_path,
        "feature_extractor": feature_extractor,
        "feature_dim": feature_dim,
    }


@st.cache_resource(show_spinner=False)
def _get_feature_extractor(training_config_path: str):
    """Create and cache the FeatureExtractor from training config."""
    from predict_baseline import create_feature_extractor_from_training_config
    extractor = create_feature_extractor_from_training_config(training_config_path)
    return extractor


def _parse_lyrics_input(text: str):
    """Split multiline text into non-empty, stripped lines (like load_lines_from_file)."""
    if not text:
        return []
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def main():
    # Page setup
    st.set_page_config(page_title="Lyric Segmentation", layout="wide")

    # Model/session selection (top of app)
    labels = list(SESSION_OPTIONS.keys())
    default_index = labels.index(DEFAULT_SESSION_LABEL) if DEFAULT_SESSION_LABEL in labels else 0
    selected_label = st.selectbox("Model Session", labels, index=default_index, key="session_select")

    # Resolve path (handle relative by anchoring to repo root)
    selected_dir = SESSION_OPTIONS[selected_label]
    if not os.path.isabs(selected_dir):
        selected_dir = str((REPO_ROOT / selected_dir).resolve())

    # Remember previous selection to detect changes
    prev_dir = st.session_state.get("selected_session_dir")
    if prev_dir != selected_dir:
        # Clear previous predictions/inputs on session change
        st.session_state["selected_session_dir"] = selected_dir
        st.session_state.pop("predictions", None)
        st.session_state.pop("parsed_lines", None)

    # Attempt to load model + calibration at startup
    model_loaded = False
    load_error = None
    inference_state = None

    if selected_dir.strip():
        try:
            with st.spinner("Loading model, calibration and feature extractor..."):
                inference_state = _boot_inference_state(selected_dir.strip())
                st.session_state["inference_state"] = inference_state
                model_loaded = True  # Consider ready if both model+calibration load succeeds
        except Exception as e:
            load_error = str(e)
            model_loaded = False
    else:
        model_loaded = False

    # Top: status notifications
    status_col = st.container()
    with status_col:
        if model_loaded and inference_state:
            device_str = inference_state.get("device", "?")
            cal_m = inference_state.get("calibration_method", "none")
            st.success(f"Model ready (device: {device_str}, calibration: {cal_m})")
            feat_dim = inference_state.get("feature_dim")
            if feat_dim is not None:
                st.success(f"Feature extractor ready (dimension: {feat_dim})")
            else:
                st.success("Feature extractor ready")
        elif load_error:
            st.error(f"Model failed to load: {load_error}")
        else:
            st.info("Model not loaded. Select a model session to enable predictions.")

    # Below notifications: predict button
    predict_clicked = st.button("Predict chorus", type="primary", disabled=not model_loaded)

    # Row 2: Two columns with equal width
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        lyrics_text = st.text_area(
            label="Lyrics",
            key="lyrics_text",
            placeholder="Paste lyrics here...",
            height=1100,  # >50 lines equivalent height
        )

    with right_col:
        # Visualization settings above predictions
        viz_mode = st.radio(
            "Visualization",
            ("Category and confidence", "Genius", "Chorus only"),
            horizontal=True,
            key="viz_mode",
        )

        # Output area
        output_placeholder = st.empty()

        # Rendering helpers
        def render_category_confidence(predictions):
            parts = []
            parts.append(
                "<div style=\"font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;\">"
            )
            for p in predictions:
                label = str(p.get("predicted_label", "")).upper()
                conf = float(p.get("confidence", 0.0))
                line_txt = html.escape(str(p.get("line", "")))
                if label == "CHORUS":
                    color = "#16a34a"  # green-600
                    bg = "#DCFCE7"    # green-100
                    border = "#86EFAC"  # green-300
                else:
                    color = "#2563eb"  # blue-600
                    bg = "#DBEAFE"    # blue-100
                    border = "#93C5FD"  # blue-300

                # Fixed-width wrappers for alignment
                label_pill = (
                    f"<span style=\"background:{bg}; color:{color}; border:1px solid {border}; padding:2px 10px; border-radius:999px; font-weight:600; font-size:12px; line-height:1.2; white-space:nowrap;\">{label}</span>"
                )
                label_cell = (
                    f"<span style=\"display:inline-flex; width:96px; min-width:96px;\">{label_pill}</span>"
                )
                conf_cell = (
                    f"<span style=\"display:inline-block; width:72px; min-width:72px; color:#6b7280; font-size:12px; line-height:1.6; text-align:right; font-variant-numeric: tabular-nums;\">({conf:.3f})</span>"
                )
                text_cell = f"<span style=\"flex:1; line-height:1.5;\">{line_txt}</span>"

                row = ("<div style=\"display:flex; align-items:flex-start; gap:10px; margin:6px 0;\">"
                       f"{label_cell}{conf_cell}{text_cell}"
                       "</div>")
                parts.append(row)
            parts.append("</div>")
            return "\n".join(parts)

        def render_genius(predictions):
            parts = []
            parts.append(
                "<div style=\"font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;\">"
            )
            # Group contiguous lines by label while preserving order
            current_label = None
            group_lines = []

            def flush_group(label, lines_acc):
                if label is None or not lines_acc:
                    return
                label_up = label.upper()
                if label_up == "CHORUS":
                    color = "#16a34a"  # green-600
                else:
                    color = "#2563eb"  # blue-600
                header = f"<div style=\"margin:10px 0 4px; font-weight:700; color:{color};\">[{label_up.title()}]</div>"
                parts.append(header)
                for lt in lines_acc:
                    parts.append(f"<div style=\"margin:2px 0; line-height:1.5;\">{html.escape(lt)}</div>")

            for p in predictions:
                label = str(p.get("predicted_label", "")).lower()
                lt = str(p.get("line", ""))
                if current_label is None:
                    current_label = label
                    group_lines = [lt]
                elif label == current_label:
                    group_lines.append(lt)
                else:
                    flush_group(current_label, group_lines)
                    current_label = label
                    group_lines = [lt]
            # Flush last group
            flush_group(current_label, group_lines)

            parts.append("</div>")
            return "\n".join(parts)

        def render_chorus_only(predictions):
            parts = []
            parts.append(
                "<div style=\"font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;\">"
            )
            # Group contiguous CHORUS sections and show a separator/header for each
            chorus_color = "#16a34a"  # green-600
            chorus_bg = "#DCFCE7"     # green-100
            chorus_border = "#86EFAC"  # green-300

            def make_row(p):
                conf = float(p.get("confidence", 0.0))
                line_txt = html.escape(str(p.get("line", "")))
                label_pill = (
                    f"<span style=\"background:{chorus_bg}; color:{chorus_color}; border:1px solid {chorus_border}; padding:2px 10px; border-radius:999px; font-weight:600; font-size:12px; line-height:1.2; white-space:nowrap;\">CHORUS</span>"
                )
                label_cell = f"<span style=\"display:inline-flex; width:96px; min-width:96px;\">{label_pill}</span>"
                conf_cell = (
                    f"<span style=\"display:inline-block; width:72px; min-width:72px; color:#6b7280; font-size:12px; line-height:1.6; text-align:right; font-variant-numeric: tabular-nums;\">({conf:.3f})</span>"
                )
                text_cell = f"<span style=\"flex:1; line-height:1.5;\">{line_txt}</span>"
                return ("<div style=\"display:flex; align-items:flex-start; gap:10px; margin:6px 0;\">"
                        f"{label_cell}{conf_cell}{text_cell}"
                        "</div>")

            # Iterate and collect contiguous chorus groups
            group = []
            any_chorus = False
            def flush_group():
                if not group:
                    return
                nonlocal any_chorus
                any_chorus = True
                # Separator + header for each section
                parts.append("<div style=\"margin:12px 0; border-top:1px dashed #cbd5e1;\"></div>")
                parts.append(f"<div style=\"margin:6px 0; font-weight:700; color:{chorus_color};\">[Chorus]</div>")
                for item in group:
                    parts.append(make_row(item))
                group.clear()

            for p in predictions:
                lab = str(p.get("predicted_label", "")).upper()
                if lab == "CHORUS":
                    group.append(p)
                else:
                    flush_group()
            flush_group()  # flush any trailing group

            if not any_chorus:
                parts.append("<div style=\"color:#6b7280;\">No chorus lines detected.</div>")

            parts.append("</div>")
            return "\n".join(parts)

        # Handle click: preprocess text and run predictions
        if predict_clicked:
            lines = _parse_lyrics_input(lyrics_text)

            if not lines:
                output_placeholder.warning("No input lines found. Paste lyrics on the left.")
            else:
                st.session_state["parsed_lines"] = lines

                # Ensure we have everything needed to run inference
                extractor = inference_state.get("feature_extractor") if inference_state else None
                model = inference_state.get("model") if inference_state else None
                device_str = inference_state.get("device") if inference_state else "cpu"
                cal_method = inference_state.get("calibration_method", "none") if inference_state else "none"
                cal_params = inference_state.get("calibration_params", {}) if inference_state else {}

                if not model or not extractor:
                    output_placeholder.error("Inference not ready (model or extractor missing).")
                else:
                    try:
                        import torch
                        from predict_baseline import predict_lyrics_structure

                        with st.spinner("Running predictions..."):
                            preds = predict_lyrics_structure(
                                lines=lines,
                                model=model,
                                feature_extractor=extractor,
                                device=torch.device(device_str),
                                calibration_method=cal_method,
                                temperature=cal_params.get("temperature", 1.0),
                                platt_A=cal_params.get("A", 1.0),
                                platt_B=cal_params.get("B", 0.0),
                                isotonic_knots=cal_params.get("knots", 0),
                            )

                        st.session_state["predictions"] = preds
                    except Exception as e:
                        output_placeholder.error(f"Prediction failed: {e}")

        # If we have predictions, render based on selected visualization mode
        preds_state = st.session_state.get("predictions")
        if preds_state:
            if viz_mode == "Genius":
                output_placeholder.markdown(render_genius(preds_state), unsafe_allow_html=True)
            elif viz_mode == "Chorus only":
                output_placeholder.markdown(render_chorus_only(preds_state), unsafe_allow_html=True)
            else:
                output_placeholder.markdown(render_category_confidence(preds_state), unsafe_allow_html=True)

    # Bottom: Model information inside a dropdown
    if model_loaded and inference_state:
        with st.expander("Model and Features Overview", expanded=False):
            model = inference_state.get("model")
            extractor = inference_state.get("feature_extractor")
            device_str = inference_state.get("device", "?")
            cal_m = inference_state.get("calibration_method", "none")
            cal_params = inference_state.get("calibration_params", {})
            feat_dim = inference_state.get("feature_dim")

            # Build features summary
            features_rows = []
            if extractor is not None and hasattr(extractor, "extractors"):
                for fname, fext in extractor.extractors.items():
                    dim = getattr(fext, "output_dim", getattr(fext, "dimension", None))
                    features_rows.append({"Feature": fname, "Dim": f"{dim}D" if dim is not None else "-"})

            # Build model summary
            arch = getattr(model, "__class__", type("_", (), {})).__name__ if model is not None else "?"
            hidden_dim = getattr(model, "hidden_dim", "?")
            num_layers = getattr(model, "num_layers", "?")
            layer_dropout = getattr(model, "layer_dropout_p", None)
            attention_enabled = getattr(model, "attention_enabled", False)
            attention_type = getattr(model, "attention_type", None)
            attention_heads = getattr(model, "attention_heads", None)

            # Layout
            ov_left, ov_right = st.columns(2)
            with ov_left:
                st.caption("Model")
                st.markdown(
                    "\n".join([
                        f"- Device: {device_str}",
                        f"- Calibration: {cal_m}",
                        f"- Architecture: {arch}",
                        f"- Hidden dim: {hidden_dim}",
                        f"- Layers: {num_layers}",
                        f"- Attention: {'on' if attention_enabled else 'off'}" + (f" ({attention_type}, {attention_heads} heads)" if attention_enabled else ""),
                    ])
                )
            with ov_right:
                st.caption("Features")
                if feats := features_rows:
                    st.table(feats)
                else:
                    st.markdown("- (no features)")
                if feat_dim is not None:
                    st.caption(f"Total feature dimension: {feat_dim}")


if __name__ == "__main__":
    main()
