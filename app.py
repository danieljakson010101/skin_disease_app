"""
streamlit_app.py — DermScan AI: Skin Disease Pattern Recognition
==================================================================
Streamlit prototype satisfying Step 4 of the assignment:
  • Dataset upload & preprocessing controls   (sidebar)
  • Model training & evaluation dashboard     (sidebar + main)
  • Result display panel                      (main, "Analyze Image" tab)
  • Real-time prediction on simulated stream  (main, "Live Webcam" tab)

Reuses model/classifier.py unchanged — all preprocessing, feature
extraction, training, and chart-generation logic is identical to the
original Flask version.
"""

import os
import sys
import base64
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.classifier import (
    run_pipeline, plot_class_distribution, train_and_save,
    load_saved_model, _find_dataset_split, CLASSES, DISPLAY_NAMES,
)

# ══════════════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DermScan AI — Skin Disease Recognition",
    page_icon="🔬",
    layout="wide",
)

# ── Minimal CSS to echo the original cream/rust/sage palette ───────
st.markdown("""
<style>
:root {
  --cream:#faf8f4; --rust:#c4572a; --sage:#3d6b5a; --ink:#1a1612;
}
.stApp { background-color: var(--cream); }
h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif !important; color: var(--ink); }
.metric-card {
  background:#fff; border:1px solid #e2ddd6; border-radius:10px;
  padding:14px; text-align:center;
}
.metric-val { font-size:22px; font-weight:700; color:var(--rust); }
.metric-lbl { font-size:11px; color:#6b6059; text-transform:uppercase; }
.mode-real { color:var(--sage); font-weight:600; }
.mode-sim  { color:var(--rust); font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  Session state
# ══════════════════════════════════════════════════════════════════
if "train_log" not in st.session_state:
    st.session_state.train_log = []
if "result" not in st.session_state:
    st.session_state.result = None

# ══════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════
st.title("🔬 DermScan AI")
st.caption("Real-Time Skin Disease Pattern Recognition — Steps 1–5 prototype")

dataset_ok = _find_dataset_split() is not None
with st.expander("System status", expanded=False):
    status_cols = st.columns(3)
    with status_cols[0]:
        st.success("Dataset found") if dataset_ok else st.error("No dataset (demo mode)")
    for i, m in enumerate(["cnn", "knn"]):
        _, meta = load_saved_model(m, "clahe", "minmax")
        with status_cols[i + 1]:
            if meta:
                st.success(f"{m.upper()} · F1 {meta['metrics']['f1']}%")
            else:
                st.error(f"{m.upper()} · not trained")

# ══════════════════════════════════════════════════════════════════
#  Sidebar — Dataset & Training (Step 4b-i, 4b-ii) + Upload controls
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    with st.expander("⚡ Train on Your Dataset", expanded=False):
        st.code("skin_disease_app/\n  dataset/train/\n    Acne/*.jpg\n    Eczema/*.jpg\n    ... (22 folders)", language=None)

        train_model = st.selectbox(
            "Model to train", ["cnn", "knn"],
            format_func=lambda x: {
                "cnn": "CNN — convolutional neural network (Keras)",
                "knn": "KNN — color histogram + LBP features",
            }[x],
            key="train_model_select",
        )
        train_enh = st.selectbox(
            "Enhancement", ["clahe", "hist", "gaussian", "none"],
            format_func=lambda x: {
                "clahe": "CLAHE — adaptive histogram equalization",
                "hist": "Global histogram equalization",
                "gaussian": "Gaussian blur — noise reduction",
                "none": "None",
            }[x],
            key="train_enh_select",
        )
        train_norm = st.selectbox(
            "Normalization", ["minmax", "zscore", "none"],
            format_func=lambda x: {
                "minmax": "Min-Max [0, 1]",
                "zscore": "Z-score (μ=0, σ=1)",
                "none": "None",
            }[x],
            key="train_norm_select",
        )

        train_epochs = 15
        if train_model == "cnn":
            train_epochs = st.slider(
                "Training epochs", min_value=3, max_value=30, value=15,
                help="More epochs = longer training, generally better accuracy "
                     "(up to a point). Keep low (5-10) for quick local testing.",
                key="train_epochs_slider",
            )

        if st.button("Train Model on Dataset", use_container_width=True, type="primary"):
            if not dataset_ok:
                st.error("No dataset found. Place images in dataset/train/")
            else:
                log_box = st.empty()
                log_lines = []

                def logger(status, msg):
                    log_lines.append((status, msg))
                    rendered = "\n".join(
                        f"{'✓' if s == 'ok' else '✗'} {m}" for s, m in log_lines
                    )
                    log_box.code(rendered, language=None)

                with st.spinner(f"Training {train_model.upper()}…"):
                    if train_model == "cnn":
                        metrics = train_and_save(train_model, train_enh, train_norm,
                                                  log_fn=logger, epochs=train_epochs)
                    else:
                        metrics = train_and_save(train_model, train_enh, train_norm, log_fn=logger)

                if metrics:
                    st.success(
                        f"Model trained — Acc: {metrics['acc']}% F1: {metrics['f1']}%"
                    )
                    st.rerun()
                else:
                    st.error("Training failed — dataset not found or too few images.")

    st.divider()

    # ── Upload & pipeline controls (Step 1–2) ──────────────────────
    st.header("📤 Upload Image")
    st.caption("Step 1–2")

    uploaded_file = st.file_uploader(
        "Click or drag a skin image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="main_upload",
    )

    if uploaded_file:
        st.image(uploaded_file, caption=f"{uploaded_file.name} · {uploaded_file.size/1024:.1f} KB",
                  use_container_width=True)

    model_type = st.selectbox(
        "Model", ["cnn", "knn"],
        format_func=lambda x: {
            "cnn": "CNN — convolutional neural network (Keras)",
            "knn": "KNN — color histogram + LBP features",
        }[x],
        key="predict_model_select",
    )
    enh_method = st.selectbox(
        "Enhancement", ["clahe", "hist", "gaussian", "none"],
        format_func=lambda x: {
            "clahe": "CLAHE — adaptive histogram equalization",
            "hist": "Global histogram equalization",
            "gaussian": "Gaussian blur — noise reduction",
            "none": "None",
        }[x],
        key="predict_enh_select",
    )
    norm_method = st.selectbox(
        "Normalization", ["minmax", "zscore", "none"],
        format_func=lambda x: {
            "minmax": "Min-Max [0, 1]",
            "zscore": "Z-score (μ=0, σ=1)",
            "none": "None",
        }[x],
        key="predict_norm_select",
    )

    st.caption("Augmentation")
    aug_flip = st.checkbox("H-Flip", value=False)
    aug_rotate = st.checkbox("Rotation ±15°", value=True)
    aug_zoom = st.checkbox("Zoom", value=False)

    run_clicked = st.button(
        "Run Analysis Pipeline", use_container_width=True,
        type="primary", disabled=uploaded_file is None,
    )

# ══════════════════════════════════════════════════════════════════
#  Run pipeline when requested
# ══════════════════════════════════════════════════════════════════
if run_clicked and uploaded_file is not None:
    with st.spinner("Running pipeline…"):
        image_bytes = uploaded_file.getvalue()
        try:
            st.session_state.result = run_pipeline(
                image_bytes=image_bytes,
                model_type=model_type,
                norm_method=norm_method,
                enh_method=enh_method,
                aug_flip=aug_flip,
                aug_rotate=aug_rotate,
                aug_zoom=aug_zoom,
            )
        except ValueError as e:
            st.error(str(e))
            st.session_state.result = None
        except Exception as e:
            st.error(f"Internal error: {e}")
            st.session_state.result = None

# ══════════════════════════════════════════════════════════════════
#  Main area — tabs
# ══════════════════════════════════════════════════════════════════
tab_results, tab_webcam, tab_ethics = st.tabs(
    ["📊 Results", "📷 Live Webcam (Step 4c)", "⚖️ Ethics (Step 5)"]
)

# ── Results tab ──────────────────────────────────────────────────
with tab_results:
    d = st.session_state.result
    if not d:
        st.info("Upload an image in the sidebar and click **Run Analysis Pipeline** to see results here.")
        st.subheader("Dataset Class Distribution")
        dist_b64 = plot_class_distribution()
        st.image(base64.b64decode(dist_b64), use_container_width=True)
    else:
        # ── Prediction card (always visible, compact, top of screen) ──
        mode_label = "Real model prediction" if d["mode"] == "real" else "Simulated (demo) prediction"
        mode_class = "mode-real" if d["mode"] == "real" else "mode-sim"
        col_pred, col_conf, col_img = st.columns([2, 1, 1])
        with col_pred:
            st.markdown(f"<span class='{mode_class}'>{mode_label}</span>", unsafe_allow_html=True)
            st.subheader(d["prediction"])
        with col_conf:
            st.metric("Confidence", f"{d['confidence']}%")
        with col_img:
            st.image(base64.b64decode(d["preprocessed_img"]), caption="Preprocessed", width=120)

        # ── Sub-tabs: only ONE section renders at a time, page stays short ──
        sub_top5, sub_metrics, sub_charts, sub_log = st.tabs(
            ["🏆 Top 5", "📈 Metrics", "📊 Charts", "📝 Log"]
        )

        with sub_top5:
            for item in d["top5"]:
                c1, c2, c3 = st.columns([2, 5, 1])
                c1.write(item["name"])
                c2.progress(item["prob"])
                c3.write(f"{item['prob']*100:.1f}%")

        with sub_metrics:
            m = d["metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(f"<div class='metric-card'><div class='metric-val'>{m['acc']}%</div>"
                         f"<div class='metric-lbl'>Accuracy</div></div>", unsafe_allow_html=True)
            mc2.markdown(f"<div class='metric-card'><div class='metric-val'>{m['prec']}%</div>"
                         f"<div class='metric-lbl'>Precision</div></div>", unsafe_allow_html=True)
            mc3.markdown(f"<div class='metric-card'><div class='metric-val'>{m['rec']}%</div>"
                         f"<div class='metric-lbl'>Recall</div></div>", unsafe_allow_html=True)
            mc4.markdown(f"<div class='metric-card'><div class='metric-val'>{m['f1']}%</div>"
                         f"<div class='metric-lbl'>F1-Score</div></div>", unsafe_allow_html=True)

        with sub_charts:
            charts = d["charts"]
            chart_pick = st.selectbox(
                "Choose a chart to view",
                ["Pixel Distribution", "Training Curves", "Model Comparison",
                 "Preprocessing Variations", "Dataset Class Distribution"],
                key="chart_picker",
            )
            chart_map = {
                "Pixel Distribution": "pixel",
                "Training Curves": "training",
                "Model Comparison": "model_compare",
                "Preprocessing Variations": "preproc_compare",
                "Dataset Class Distribution": "distribution",
            }
            st.image(base64.b64decode(charts[chart_map[chart_pick]]), use_container_width=True)

        with sub_log:
            for status, msg in d["log"]:
                st.write(("✅ " if status == "ok" else "❌ ") + msg)

# ── Live Webcam tab (Step 4c — real-time prediction requirement) ───
with tab_webcam:
    st.subheader("Live Webcam Stream")
    st.caption(
        "Simulates a real-time input stream: capture a frame from your camera and "
        "run it through the same preprocessing → feature → model pipeline as Step 1–3."
    )

    col_cam, col_ctrl = st.columns([1.2, 1])
    with col_ctrl:
        wc_model = st.selectbox(
            "Model", ["cnn", "knn"], key="wc_model",
            format_func=lambda x: x.upper(),
        )
        wc_enh = st.selectbox(
            "Enhancement", ["clahe", "hist", "gaussian", "none"], key="wc_enh",
        )
        wc_norm = st.selectbox(
            "Normalization", ["minmax", "zscore", "none"], key="wc_norm",
        )
        st.caption(
            "Streamlit's camera widget captures one still frame per interaction "
            "(browser permission required). Click **Take Photo**, then re-open the "
            "camera to simulate the next frame in the stream."
        )

    with col_cam:
        cam_frame = st.camera_input("Capture frame", key="wc_camera")

    if cam_frame is not None:
        frame_bytes = cam_frame.getvalue()
        with st.spinner("Predicting on captured frame…"):
            try:
                wc_result = run_pipeline(
                    image_bytes=frame_bytes,
                    model_type=wc_model,
                    norm_method=wc_norm,
                    enh_method=wc_enh,
                    aug_flip=False, aug_rotate=False, aug_zoom=False,
                )
            except Exception as e:
                wc_result = None
                st.error(f"Prediction failed: {e}")

        if wc_result:
            mode_label = "Real model prediction" if wc_result["mode"] == "real" else "Simulated (demo) prediction"
            mode_class = "mode-real" if wc_result["mode"] == "real" else "mode-sim"
            st.markdown(f"<span class='{mode_class}'>{mode_label}</span>", unsafe_allow_html=True)
            wc1, wc2 = st.columns([2, 1])
            wc1.subheader(wc_result["prediction"])
            wc2.metric("Confidence", f"{wc_result['confidence']}%")

            st.write("**Top 5 — this frame**")
            for item in wc_result["top5"]:
                c1, c2, c3 = st.columns([2, 5, 1])
                c1.write(item["name"])
                c2.progress(item["prob"])
                c3.write(f"{item['prob']*100:.1f}%")

            if "wc_stream_log" not in st.session_state:
                st.session_state.wc_stream_log = []
            st.session_state.wc_stream_log.insert(
                0, f"{wc_result['prediction']} ({wc_result['confidence']}%) — {wc_result['mode']}"
            )
            st.session_state.wc_stream_log = st.session_state.wc_stream_log[:20]

            st.write("**Stream log**")
            st.code("\n".join(st.session_state.wc_stream_log), language=None)
    else:
        st.info("Allow camera access and click **Take Photo** to run a live prediction.")

# ── Ethics tab (Step 5) ─────────────────────────────────────────────
with tab_ethics:
    st.subheader("⚖️ Ethical & Practical Reflection")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🩺 Bias & Fairness")
        st.write(
            "Dermatology image datasets are known to skew toward lighter skin tones, "
            "which can depress accuracy for darker skin and for under-represented "
            "conditions such as *Vasculitis* or *Vascular Tumors* that have far fewer "
            "training samples than common classes like *Acne* or *Tinea*. A model "
            "trained on this distribution risks confidently mis-classifying exactly "
            "the patients it has seen least."
        )

        st.markdown("#### 🔒 Privacy")
        st.write(
            "Skin images are sensitive biometric and health data. Webcam capture "
            "raises the stakes further, since frames may unintentionally include "
            "faces or other identifying context. This prototype keeps inference "
            "local to the session and does not persist uploaded or captured images "
            "beyond what is needed to render a result."
        )

    with c2:
        st.markdown("#### 🔍 Transparency")
        st.write(
            "The system surfaces its full pipeline — enhancement method, "
            "normalization, feature type, model, and a Top-5 probability "
            "breakdown — rather than a single opaque label. It also distinguishes "
            "**real** model predictions from **simulated** demo output whenever a "
            "trained model file is unavailable."
        )

        st.markdown("#### 📡 Live-Stream Deployment Risks")
        st.write(
            "Real-time webcam classification of medical conditions invites misuse: "
            "unsupervised self-diagnosis, false reassurance from a wrong 'benign' "
            "call, false alarm from a wrong 'cancer' call, or informal surveillance "
            "if a camera stream is repurposed without consent. Continuous inference "
            "can also amplify a single bad frame (motion blur, poor lighting) into a "
            "confidently wrong streamed result."
        )

    st.markdown("#### 🛡️ Mitigation Strategies")
    st.markdown(
        "- Always present confidence and Top-5 alternatives, never just the top label.\n"
        "- Label every prediction with its mode (`real` vs `simulated`) and exact "
        "preprocessing/model configuration used.\n"
        "- Treat the tool as decision-support / educational only, with an explicit "
        "on-screen disclaimer that it is not a diagnostic device.\n"
        "- Average predictions across several consecutive frames before surfacing a "
        "'stable' webcam result, reducing the impact of any single noisy frame.\n"
        "- Require explicit camera consent and discard frames immediately after use.\n"
        "- Audit per-class performance (not just overall accuracy) before any real "
        "deployment, to catch skewed, low-sample-class weaknesses."
    )

    st.markdown(
        "> *\"Raise your words, not voice. It is rain that grows flowers, not "
        "thunder.\"* — Rumi"
    )