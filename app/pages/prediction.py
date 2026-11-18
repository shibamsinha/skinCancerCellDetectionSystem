"""Prediction page: upload -> classify -> Grad-CAM explanation.

Uploaded images are processed in memory only and are never written to disk.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from app.components import charts, state, ui
from skincancer.inference import ModelUnavailableError
from skincancer.preprocessing import ImageLoadError, load_image

ss = st.session_state
ss.setdefault("upload_key", 0)
ss.setdefault("result", None)

statuses = state.model_statuses()
available = [s for s in statuses if s.trained]
by_name = {s.name: s for s in statuses}

ui.page_header(
    "Classify a skin lesion image",
    "Upload a dermatoscopic image, choose a model, and get a benign/malignant prediction with a Grad-CAM explanation.",
)
ui.disclaimer(compact=True)
st.write("")

if not available:
    ui.notice(
        "<b>No trained model weights were found.</b> The application never shows predictions from untrained or "
        "random weights. Train the primary model first:<br><code>python scripts/train.py --model modified_densenet201</code>",
        "warn",
    )
    st.stop()

left, right = st.columns([1, 1.1], gap="large")

with left:
    with st.container(border=True):
        st.markdown("##### 1 · Upload an image")
        uploaded = st.file_uploader(
            "Dermatoscopic image (JPEG or PNG, up to 20 MB)",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{ss.upload_key}",
            help="Images are processed in memory and are not stored.",
        )
        # Decode immediately so the Classify button can be disabled for invalid files.
        image, file_hash, load_error, data = None, None, None, b""
        if uploaded is not None:
            data = uploaded.getvalue()
            file_hash = hashlib.sha256(data).hexdigest()
            try:
                image = load_image(data)
            except ImageLoadError as exc:
                load_error = str(exc)

        st.markdown("##### 2 · Choose a model")
        names = [s.name for s in available]
        primary = next((s.name for s in available if s.is_primary), names[0])
        model_name = st.selectbox(
            "Model architecture",
            names,
            index=names.index(primary),
            format_func=lambda n: by_name[n].display_name + ("  ·  primary" if by_name[n].is_primary else ""),
        )
        unavailable = [s.display_name for s in statuses if not s.trained]
        info = f"{by_name[model_name].input_size}×{by_name[model_name].input_size} px input"
        mm = state.model_metrics(model_name)
        if mm:
            info += f" · test accuracy {ui.fmt_pct(mm['metrics']['accuracy'])}, ROC-AUC {mm['metrics']['roc_auc']:.3f}"
        st.caption(info)
        if by_name[model_name].training_state == "training":
            st.warning("This model is still training. The weights are the best checkpoint so far and may change.",
                       icon=":material/hourglass_top:")
        if unavailable:
            st.caption("Not yet trained (hidden): " + ", ".join(unavailable))

        st.markdown("##### 3 · Run")
        # Stacked (not side by side) so the labels never wrap in a narrow column.
        run = st.button("Classify image", type="primary", disabled=image is None, use_container_width=True,
                        icon=":material/play_arrow:")
        if st.button("Clear image and results", type="tertiary", disabled=uploaded is None,
                     icon=":material/delete:", help="Remove the uploaded image and its results"):
            ss.upload_key += 1
            ss.result = None
            st.rerun()

with right:
    with st.container(border=True):
        st.markdown("##### Image preview")
        if uploaded is None:
            st.markdown(
                "<div style='height:300px;display:flex;align-items:center;justify-content:center;"
                "color:#7a8696;background:#fbfcfe;border:1px dashed #d5dde8;border-radius:12px;text-align:center'>"
                "No image uploaded yet.<br>Drag and drop a dermatoscopic JPEG or PNG on the left.</div>",
                unsafe_allow_html=True,
            )
        elif load_error:
            st.error(f"This file can't be used: {load_error}")
        else:
            st.image(image, use_container_width=True)
            st.caption(f"{uploaded.name} · {image.width}×{image.height} px · {len(data) / 1024:.0f} KB")
            if min(image.size) < 150:
                st.warning("Very small image: details may be lost after resizing, so treat the result with caution.")

if run and image is not None:
    status = by_name[model_name]
    try:
        with st.spinner(f"Running {status.display_name} and computing Grad-CAM…"):
            predictor = state.get_predictor(model_name)
            prediction = predictor.predict(image)
            explanation = predictor.explain(image)
        ss.result = {"hash": file_hash, "model": model_name, "prediction": prediction,
                     "explanations": {prediction.label: explanation}}
    except ModelUnavailableError:
        st.error("The weights for this model are not available. Train it first.")
    except Exception as exc:  # noqa: BLE001 - surface any inference failure to the user
        st.error(f"Inference failed: {type(exc).__name__}: {exc}")

result = ss.result
if result and image is not None and result["hash"] == file_hash:
    if result["model"] != model_name:
        ui.notice(f"Results below are from <b>{by_name[result['model']].display_name}</b>. "
                  "Click <b>Classify image</b> to run the newly selected model.")
    pred = result["prediction"]
    rstatus = by_name[result["model"]]
    rmetrics = state.model_metrics(result["model"])

    st.write("")
    st.markdown("### Prediction")
    c1, c2 = st.columns([1.05, 1], gap="large")
    with c1:
        kind = "malignant" if pred.label == "Malignant" else "benign"
        icon = "!" if kind == "malignant" else "✓"
        meaning = (
            "The model associates this image with features of <b>malignant</b> lesions in its training data "
            "(melanoma, basal cell carcinoma, actinic keratosis). This is not a diagnosis. Please consult a dermatologist."
            if kind == "malignant"
            else "The model associates this image with features of <b>benign</b> lesions in its training data. "
            "This does not rule out disease. Any changing or concerning lesion should be examined by a dermatologist."
        )
        st.markdown(
            f"""<div class="sc-result {kind}"><div class="sc-label">Predicted category</div>
            <div class="sc-verdict"><span class="sc-icon">{icon}</span>{pred.label}</div>
            <p>{meaning}</p></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown('<div class="sc-figure-title">Class probabilities (softmax output)</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.probability_bars(pred.probabilities, pred.label), use_container_width=True,
                        config=charts.CONFIG)
    with c2:
        k1, k2 = st.columns(2)
        k1.metric("Model confidence", ui.fmt_pct(pred.confidence),
                  help="Softmax probability of the predicted class. It is model certainty, not the probability that a patient has cancer.")
        k2.metric("Malignant probability", ui.fmt_pct(pred.probabilities["Malignant"]))
        k3, k4 = st.columns(2)
        k3.metric("Inference time", f"{pred.inference_ms:.0f} ms", help=f"Forward pass on {pred.device.upper()}")
        k4.metric("Input resolution", f"{pred.input_size}×{pred.input_size}")
        st.markdown(f"**Model:** {rstatus.display_name}")
        if rmetrics:
            m = rmetrics["metrics"]
            st.caption(
                f"On {m['n']:,} held-out test images this model reached sensitivity {ui.fmt_pct(m['sensitivity'])} "
                f"and specificity {ui.fmt_pct(m['specificity'])}, so roughly "
                f"{ui.fmt_pct(1 - m['sensitivity'], 0)} of malignant test lesions were missed and "
                f"{ui.fmt_pct(1 - m['specificity'], 0)} of benign ones were flagged."
            )
        if pred.confidence < 0.65:
            ui.notice("<b>Low confidence.</b> The two class probabilities are close, so the model is uncertain about "
                      "this image. Treat the result as inconclusive.", "warn")

    # ------------------------------------------------------------ Grad-CAM
    st.write("")
    st.markdown("### Grad-CAM explainability")
    target = st.segmented_control(
        "Show the image regions that support the class:",
        ["Benign", "Malignant"],
        default=pred.label,
        key=f"cam_class_{result['hash'][:12]}_{result['model']}",
    ) or pred.label
    if target not in result["explanations"]:
        with st.spinner("Computing Grad-CAM…"):
            predictor = state.get_predictor(result["model"])
            result["explanations"][target] = predictor.explain(image, class_index=predictor.class_names.index(target))
    ex = result["explanations"][target]

    g1, g2, g3 = st.columns(3, gap="medium")
    for col, img, title, cap in (
        (g1, ex.original, "Original image", f"Uploaded image, resized to {ex.original.shape[0]}×{ex.original.shape[1]}"),
        (g2, ex.heatmap, "Grad-CAM heat map", f"Evidence for '{ex.explained_class}' (red = strongest)"),
        (g3, ex.overlay, "Grad-CAM overlay", "Heat map blended over the image"),
    ):
        with col:
            st.markdown(f'<div class="sc-figure-title">{title}</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown(f'<div class="sc-caption">{cap}</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin:10px 0 2px;color:#4b5868;font-size:0.82rem'>"
        "<span>Low influence</span><span style='flex:0 0 220px;height:10px;border-radius:5px;"
        "background:linear-gradient(90deg,#00007f,#0000ff,#00ffff,#7fff7f,#ffff00,#ff0000,#7f0000)'></span>"
        "<span>High influence</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Target layer: `{ex.target_layer}` · feature map {ex.feature_map_size[0]}×{ex.feature_map_size[1]}, "
        f"upsampled to the input size · computed in {ex.seconds:.2f} s from this image and this prediction."
    )

    with st.expander("How Grad-CAM works and how to read it"):
        st.markdown(
            f"""
**Gradient-weighted Class Activation Mapping** (Selvaraju et al., 2017) shows which parts of an image
pushed the model towards a class.

1. The image passes through the network. The feature maps of the last convolutional layer before global
   average pooling are recorded. For Modified DenseNet201 this is **CONV2** (64 maps of 7×7).
2. The score of the chosen class (*{ex.explained_class}*) is differentiated with respect to those feature maps.
3. Each map's gradient is averaged into an importance weight. The weighted sum of the maps, passed through
   ReLU, is the heat map. It is then upsampled to the image size.

**Reading it:** red areas contributed most to the score of the selected class. The 7×7 resolution makes the
map coarse, so it highlights regions, not precise lesion borders. A heat map centred on rulers, hair, ink
marks or the dark corners of the dermatoscope shows the model may be using artefacts rather than the lesion.
Grad-CAM explains the model's behaviour. It does not validate the prediction.
"""
        )
    with st.expander("What the model actually sees (after preprocessing)"):
        p1, p2, _ = st.columns([1, 1, 1])
        p1.image(ex.original, caption="Resized (squashed to a square, as in the original Keras pipeline)",
                 use_container_width=True)
        p2.image(ex.model_input, caption="After contrast stretching (2nd–98th percentile)", use_container_width=True)

elif uploaded is not None and image is not None:
    st.write("")
    ui.notice("Image ready. Click <b>Classify image</b> to run the selected model.")

st.write("")
with st.expander("Limitations of this demonstration"):
    st.markdown(ui.MEDICAL_DISCLAIMER, unsafe_allow_html=True)
ui.footer()
