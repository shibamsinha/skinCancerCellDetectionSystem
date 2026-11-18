"""Overview / landing page."""

from __future__ import annotations

import streamlit as st

from app.components import state, ui

cfg = state.get_config()
statuses = state.model_statuses()
primary = next(s for s in statuses if s.is_primary)
report = state.data_report()
trained = [s for s in statuses if s.trained]

ui.hero(
    eyebrow="Academic research reconstruction · Dermatoscopic image analysis",
    title="Skin Cancer Classification",
    subtitle="Modified Deep Learning Protocol",
    body=(
        "This application classifies dermatoscopic images of skin lesions as <b>benign</b> or <b>malignant</b> with a "
        "modified, ImageNet-pretrained <b>DenseNet201</b>: three extra convolutional layers after the fourth dense "
        "block and a custom classification head. Every prediction comes with a <b>Grad-CAM</b> heat map showing which "
        "regions influenced it. Models were rebuilt and retrained from the original 2024 project report, and every "
        "number shown comes from an actual evaluation run."
    ),
)

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.page_link("pages/prediction.py", label="Classify an image", icon=":material/upload_file:", use_container_width=True)
with c2:
    st.page_link("pages/model_comparison.py", label="Compare models", icon=":material/bar_chart:", use_container_width=True)
with c3:
    st.page_link("pages/project_information.py", label="Read the methodology", icon=":material/menu_book:", use_container_width=True)

st.write("")
total = report["validation"]["valid"] if report else 10015
binary = report["class_distribution"]["binary"] if report else {}
mal = binary.get("Malignant")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Dermatoscopic images", f"{total:,}")
m2.metric("Unique lesions", f"{sum(v['lesions'] for v in report['splits'].values()):,}" if report else "7,470")
m3.metric("Malignant share", ui.fmt_pct(mal / total) if mal else "19.5%", help="mel + bcc + akiec")
finished = [s for s in trained if s.training_state != "training"]
in_progress = len(trained) - len(finished)
m4.metric("Trained models", f"{len(finished)} / {len(statuses)}",
          help=f"{in_progress} more currently training" if in_progress else None)

st.write("")
left, right = st.columns([1.35, 1])
with left:
    metrics = state.model_metrics(primary.name)
    if metrics:
        m = metrics["metrics"]
        body = (
            "<p>DenseNet201 (4 dense blocks, 3 transition layers) &rarr; CONV1_1 3&times;1&times;128 &rarr; "
            "CONV1_2 1&times;3&times;128 &rarr; CONV2 3&times;3&times;64 &rarr; GAP &rarr; BatchNorm &rarr; Dense(256) "
            "&rarr; Dense(2) + softmax.</p>"
            f"<ul><li>Test accuracy <b>{ui.fmt_pct(m['accuracy'])}</b> &middot; balanced accuracy "
            f"<b>{ui.fmt_pct(m['balanced_accuracy'])}</b></li>"
            f"<li>Sensitivity <b>{ui.fmt_pct(m['sensitivity'])}</b> &middot; specificity <b>{ui.fmt_pct(m['specificity'])}</b>"
            f" &middot; ROC-AUC <b>{m['roc_auc']:.3f}</b></li>"
            f"<li>Measured on {m['n']:,} held-out test images from lesions never seen in training.</li></ul>"
        )
        chip = ui.chip(*primary.chip)
    else:
        body = (
            "<p>DenseNet201 &rarr; CONV1_1 3&times;1&times;128 &rarr; CONV1_2 1&times;3&times;128 &rarr; CONV2 3&times;3&times;64 "
            "&rarr; GAP &rarr; BatchNorm &rarr; Dense &rarr; Dense + softmax.</p>"
            + ("<p style='margin-top:8px'>Training is in progress; test-set metrics appear here once it finishes.</p>"
               if primary.training_state == "training" else
               "<p style='margin-top:8px'>No evaluated checkpoint yet. Train it with "
               "<code>python scripts/train.py --model modified_densenet201</code>.</p>")
        )
        chip = ui.chip(*primary.chip)
    ui.card("Primary model: Modified DenseNet201", body, kicker="Proposed architecture", chip=chip)

with right:
    ds = (
        "<p>HAM10000: 10,015 dermatoscopic images of common pigmented lesions (ISIC archive; "
        "Tschandl et al., 2018), seven diagnostic categories grouped into two classes.</p><ul>"
        "<li><b>Malignant</b>: melanoma, basal cell carcinoma, actinic keratoses / intraepithelial carcinoma</li>"
        "<li><b>Benign</b>: melanocytic nevi, benign keratosis-like lesions, dermatofibroma, vascular lesions</li>"
        "<li>Lesion-grouped split, 64 / 16 / 20 (train / validation / test)</li></ul>"
    )
    ui.card("Dataset", ds, kicker="HAM10000 · CC BY-NC 4.0")

st.write("")
st.markdown("#### Available model architectures")
cols = st.columns(3)
for i, s in enumerate(statuses):
    with cols[i % 3]:
        status_chip = ui.chip(*s.chip)
        role = "Primary model" if s.is_primary else ("Ablation" if s.name == "densenet201" else "Comparison model")
        detail = f"{'ImageNet-pretrained' if s.pretrained else 'Trained from scratch'} &middot; {s.input_size}&times;{s.input_size} px"
        ui.card(s.display_name, f"<p>{detail}</p>", kicker=role, chip=status_chip, tile=True)
        st.write("")

st.markdown("#### How a prediction is made")
ui.flow([
    ("Upload JPEG / PNG", "ink"), ("Resize", ""), ("Contrast stretching", ""), ("Normalise", ""),
    ("Modified DenseNet201", "teal"), ("Softmax: Benign / Malignant", "teal"), ("Grad-CAM on CONV2", "ink"),
])
st.write("")
ui.disclaimer()
ui.footer()
