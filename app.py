import os
import platform

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils import (CLASS_NAMES, CONDITIONS, IMG_SIZE, apply_condition,
                   avg_latency_ms, detect_hands, enhance_exposure, gradcam,
                   load_rgb, overlay_heatmap, predict_probs, to_array)

st.set_page_config(page_title="HandGesture AI", page_icon="✋", layout="wide")

st.title("HandGesture AI")
st.write("Hand gesture recognition (peace / stop / thumbs up): "
         "upload or camera → prediction → visual explanation (Grad-CAM).")


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


model_paths = {os.path.splitext(f)[0]: os.path.join("models", f)
               for f in sorted(os.listdir("models")) if f.endswith(".keras")} \
    if os.path.isdir("models") else {}
if not model_paths:
    st.error("No models found in ./models. Run the training cells of the notebook first.")
    st.stop()

default_name = None
if os.path.exists("models/default_model.txt"):
    default_name = open("models/default_model.txt").read().strip()
names = list(model_paths)
default_idx = names.index(default_name) if default_name in names else 0

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", names, index=default_idx)
    use_mediapipe = st.checkbox("Show MediaPipe hand landmarks", value=True)

    st.header("Environment")
    gpus = tf.config.list_physical_devices("GPU")
    st.write(f"**Device:** {'GPU' if gpus else 'CPU'}")
    st.write(f"**Platform:** {platform.system()} {platform.release()}")
    st.write(f"**Python:** {platform.python_version()} | **TF:** {tf.__version__}")
    try:
        ua = st.context.headers.get("User-Agent", "unknown")
    except Exception:
        ua = "unavailable"
    st.write(f"**Browser:** {ua}")

    st.header("Ethics & consent")
    st.caption(
        "Photos are processed in memory and are not stored. Only upload images "
        "of your own hand or images you have permission to use. Images produced by "
        "the stress-test generator or by text-to-video models are SYNTHETIC "
        "and are labelled as such. The model was trained on a small dataset "
        "from a few people and may perform worse for other skin tones, "
        "hand sizes, lighting and backgrounds."
    )

model = load_model(model_paths[model_name])

st.subheader("1. Input")
source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
if source == "Upload":
    uploaded = st.file_uploader("Upload hand gesture image",
                                type=["jpg", "jpeg", "png", "heic", "heif", "webp"])
else:
    uploaded = st.camera_input("Take a photo of your hand gesture")

if uploaded is None:
    st.info("Provide an image to start.")
    st.stop()

pil_img = load_rgb(uploaded)
arr = to_array(pil_img)

tab_pred, tab_stress, tab_results = st.tabs(
    ["Prediction + Grad-CAM", "Stress-test generator", "Experiment results"])

with tab_pred:
    probs = predict_probs(model, arr)
    top = int(np.argmax(probs))
    mean_ms, std_ms = avg_latency_ms(model, arr)

    c_img, c_res = st.columns([1, 2])
    with c_img:
        st.image(pil_img, caption="Input image", width=300)
    with c_res:
        st.markdown("**Predicted gesture**")
        st.subheader(CLASS_NAMES[top].replace("_", " ").title())
        st.markdown(f"**Confidence:** {probs[top] * 100:.2f}%")
        st.markdown(f"**Avg. inference latency (20 runs):** {mean_ms:.1f} ± {std_ms:.1f} ms "
                    f"on {'GPU' if gpus else 'CPU'}")
        for name, p in zip(CLASS_NAMES, probs):
            st.write(f"{name}: {p * 100:.1f}%")
            st.progress(float(p))

    st.subheader("Explainable AI: Grad-CAM")
    heat, _ = gradcam(model, arr)
    overlay = overlay_heatmap(arr, heat)
    g1, g2 = st.columns(2)
    with g1:
        st.image(arr, caption="Model input", width=360)
    with g2:
        st.image(overlay, caption="Grad-CAM overlay (red = most important region)", width=360)

    if use_mediapipe:
        st.subheader("MediaPipe hand detection")
        try:
            big = pil_img.copy()
            big.thumbnail((640, 640))
            annotated, n_hands, n_lm = detect_hands(np.asarray(big))
            st.markdown(f"**Detected hands:** {n_hands} | **Landmarks:** {n_lm}")
            if n_hands > 0:
                st.image(annotated, caption="Hand landmarks", width=360)
            else:
                st.warning("No hands detected in the image.")
        except Exception as e:
            st.warning(f"MediaPipe is not available: {e}")

with tab_stress:
    st.write("Generate **synthetic** degraded versions of your image and see how "
             "the selected model reacts. Generated images are not real photos.")
    true_label = st.selectbox("True label (optional)", ["(unknown)"] + CLASS_NAMES)
    cond = st.selectbox("Condition", [c for c in CONDITIONS if c != "normal"])
    degraded = apply_condition(arr, cond, np.random.default_rng(0))
    p_deg = predict_probs(model, degraded)
    enhanced = enhance_exposure(degraded)
    p_enh = predict_probs(model, enhanced)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.image(arr, caption="Original", width=260)
        st.write(f"Prediction: **{CLASS_NAMES[int(np.argmax(probs))]}** "
                 f"({probs.max() * 100:.1f}%)")
    with s2:
        st.image(degraded, caption=f"SYNTHETIC: {cond}", width=260)
        st.write(f"Prediction: **{CLASS_NAMES[int(np.argmax(p_deg))]}** "
                 f"({p_deg.max() * 100:.1f}%)")
    with s3:
        st.image(enhanced, caption="SYNTHETIC + exposure enhancement (CLAHE)", width=260)
        st.write(f"Prediction: **{CLASS_NAMES[int(np.argmax(p_enh))]}** "
                 f"({p_enh.max() * 100:.1f}%)")
    st.caption("Exposure enhancement is a lightweight classical stand-in "
               "(auto-gamma + CLAHE), not the ArtAug model.")

    st.subheader("Full stress test (all conditions)")
    rows = []
    for c in CONDITIONS:
        img_c = apply_condition(arr, c, np.random.default_rng(0))
        p = predict_probs(model, img_c)
        k = int(np.argmax(p))
        row = {"condition": c, "prediction": CLASS_NAMES[k],
               "confidence_%": round(float(p[k]) * 100, 2)}
        if true_label != "(unknown)":
            row["correct"] = CLASS_NAMES[k] == true_label
        rows.append(row)
    st.dataframe(pd.DataFrame(rows))


with tab_results:
    st.write("Results produced by the notebook (clean test set, synthetic "
             "stress conditions and, if provided, a real hard test set).")
    for title, f in [("Accuracy by condition", "results/results_accuracy.csv"),
                     ("Macro-F1 by condition", "results/results_f1.csv"),
                     ("Latency", "results/latency.csv"),
                     ("Dataset", "results/dataset_summary.csv")]:
        if os.path.exists(f):
            st.subheader(title)
            st.dataframe(pd.read_csv(f))
    for f in ["results/training_curves.png", "results/accuracy_by_condition.png"]:
        if os.path.exists(f):
            st.image(f, width=900)
