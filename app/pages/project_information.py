"""Project information: methodology, architecture, stack, limitations and deviations."""

from __future__ import annotations

import streamlit as st

from app.components import state, ui
from skincancer.data import DX_FULL_NAMES

cfg = state.get_config()
report = state.data_report()

_MODULE_LABELS = {"additional": "added convolution layers", "head": "classification head", "classifier": "classifier"}


def _module_names(modules: list[str]) -> str:
    return " and ".join(_MODULE_LABELS.get(m, m) for m in modules)

ui.page_header(
    "Project information",
    "Research objective, dataset, methodology and limitations of the reconstructed project.",
)

tabs = st.tabs(["Objective", "Dataset", "Preprocessing", "Architecture", "Training & evaluation",
                "Grad-CAM", "Technical stack", "Limitations & deviations"])

# ------------------------------------------------------------------ objective
with tabs[0]:
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        st.markdown(
            """
#### Research objective
The original project, *Classification of Skin Cancer Using Modified Deep Learning Protocol* (BCA Major Project,
The Heritage Academy, Kolkata, 2021–24), set out to build a deep-learning model that classifies
dermatoscopic images of skin lesions as **benign** or **malignant**. It also set out to show that a **modified,
pretrained DenseNet201** beats standard CNN architectures, and to explain predictions with **Grad-CAM**.

#### About this reconstruction
The original source code and trained weights were lost. This application rebuilds the project from the surviving
report. Each implementation detail is classified as *documented*, *inferred from the report's figures*, *taken from
the published paper the methodology was adapted from*, *reconstructed with standard practice*, or *unrecoverable*.
Every model shown here was retrained, and every metric was measured on a held-out test set.
"""
        )
    with c2:
        ui.card(
            "Methodology source",
            "<p>The report's methodology, workflow figure and architecture figure follow "
            "<b>Rehman et al., \"Classification of Skin Cancer Lesions Using Explainable Deep Learning\"</b>, "
            "<i>Sensors</i> 2022, 22(18):6915. That paper's Table 2 supplies the layer details the report "
            "refers to but omits (CONV1_1 kernel 3&times;1, CONV2 with 64 filters).</p>",
            kicker="Key finding of the documentation analysis",
        )
        st.write("")
        ui.card(
            "Full specification",
            "<p>See <code>RECONSTRUCTION_SPECIFICATION.md</code> in the repository for every documented, inferred "
            "and reconstructed detail, the contradictions found, and the assumptions made.</p>",
            kicker="Documentation",
        )

# -------------------------------------------------------------------- dataset
with tabs[1]:
    st.markdown(
        """
#### HAM10000
*Human Against Machine with 10000 training images* (Tschandl, Rosendahl & Kittler, *Scientific Data* 2018):
**10,015 dermatoscopic images** of **7,470 lesions** from multiple sources and acquisition devices, part of the ISIC
archive. More than half of the diagnoses are confirmed by histopathology; the rest by follow-up, expert consensus or
confocal microscopy. Licence: CC BY-NC 4.0.
"""
    )
    if report:
        dx_counts = report["class_distribution"]["dx"]
        rows = []
        for dx in ["mel", "bcc", "akiec", "nv", "bkl", "df", "vasc"]:
            n = dx_counts.get(dx, 0)
            group = "Malignant" if dx in cfg.data.malignant_dx else "Benign"
            rows.append([DX_FULL_NAMES[dx], dx, group, f"{n:,}", ui.fmt_pct(n / report["validation"]["valid"])])
        ui.html_table(["Diagnosis", "Code", "Binary class", "Images", "Share"], rows, numeric_cols={3, 4})
        st.write("")
        sp = report["splits"]
        ui.html_table(
            ["Split", "Images", "Lesions", "Benign", "Malignant", "Malignant share"],
            [[k.capitalize(), f"{v['images']:,}", f"{v['lesions']:,}", f"{v['benign']:,}", f"{v['malignant']:,}",
              ui.fmt_pct(v["malignant_fraction"])] for k, v in sp.items()],
            numeric_cols={1, 2, 3, 4, 5},
        )
    st.write("")
    ui.notice(
        "<b>How the benign/malignant mapping was established.</b> The report never states it. Its Fig. 3 (male ≈ 1,230 vs "
        "female ≈ 725 malignant cases) and Fig. 2 (age histogram) were matched against the real metadata. Only "
        "<b>melanoma + basal cell carcinoma + actinic keratoses</b> reproduces them exactly (1,227 male / 727 female). "
        "<br><b>Leakage prevention.</b> Images of the same lesion are kept in the same split (grouped by "
        "<code>lesion_id</code>, stratified by diagnosis). Split ratios of 64/16/20 were inferred from the batch counts "
        "in the original notebook's screenshots."
    )

# -------------------------------------------------------------- preprocessing
with tabs[2]:
    ui.flow([("Load RGB", "ink"), ("Resize to 224×224", ""), ("Contrast stretching", ""),
             ("Augment (training only)", "teal"), ("Backbone normalisation", ""), ("Tensor", "ink")])
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.card(
            "Contrast stretching",
            "<p>Applied to <b>every</b> image (training, validation, test and uploads) to improve the visibility of "
            "faded lesions. Intensities between the 2nd and 98th percentile are mapped linearly onto 0–255, with "
            "percentiles computed over all channels together to preserve hue.</p>"
            "<p style='margin-top:6px'><i>Documented: technique. Reconstructed: exact algorithm and percentiles.</i></p>",
            kicker="Preprocessing",
        )
    with c2:
        ui.card(
            "Data augmentation: training images only",
            "<ul><li><b>Horizontal flip</b>, probability 0.5</li>"
            "<li><b>Rotation</b> by one of 0°, ±15°, ±45° (angles from the source paper)</li>"
            "<li><b>Gaussian noise</b>, variance 0.1 (as documented), applied to 25% of draws</li></ul>"
            "<p style='margin-top:6px'>Applied online <i>after</i> the split, so augmented copies can never leak into "
            "the test set. The minority (malignant) class is oversampled, each draw with a fresh augmentation.</p>",
            kicker="Augmentation",
        )

# --------------------------------------------------------------- architecture
with tabs[3]:
    st.markdown("#### Modified DenseNet201")
    st.graphviz_chart(
        """
digraph G {
  rankdir=TB; newrank=true; bgcolor="transparent"; nodesep=0.28; ranksep=0.42;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.14,0.07",
        color="#cfe1f6", fillcolor="#e8f1fb", fontcolor="#14202e"];
  edge [color="#7a8696", arrowsize=0.6];
  in [label="Input\\n224×224×3", fillcolor="#f1f3f6", color="#e3e8ef"];
  stem [label="Conv 7×7 /2\\nmax-pool 3×3 /2"];
  db1 [label="Dense block 1 (6)\\n+ transition 1"];
  db2 [label="Dense block 2 (12)\\n+ transition 2"];
  db3 [label="Dense block 3 (48)\\n+ transition 3"];
  db4 [label="Dense block 4 (32)\\n1920 × 7 × 7"];
  c11 [label="CONV1_1\\n3×1, 128", fillcolor="#e5f5f3", color="#9fd6cf"];
  c12 [label="CONV1_2\\n1×3, 128", fillcolor="#e5f5f3", color="#9fd6cf"];
  c2  [label="CONV2\\n3×3, 64\\nGrad-CAM target", fillcolor="#e5f5f3", color="#9fd6cf"];
  gap [label="GAP", fillcolor="#ffffff", color="#e3e8ef"];
  bn  [label="BatchNorm", fillcolor="#ffffff", color="#e3e8ef"];
  d1  [label="Dense 1\\n256, ReLU", fillcolor="#ffffff", color="#e3e8ef"];
  d2  [label="Dense 2\\n2 units", fillcolor="#ffffff", color="#e3e8ef"];
  sm  [label="Softmax\\nBenign | Malignant", fillcolor="#f1f3f6", color="#e3e8ef"];
  subgraph cluster_b { label="Pretrained DenseNet201 (ImageNet): 4 dense blocks, 3 transition layers";
    fontname="Helvetica"; fontsize=10; fontcolor="#4b5868"; color="#e3e8ef"; style="rounded";
    { rank=same; in; stem; db1; db2; db3; db4; } }
  subgraph cluster_a { label="Additional layers"; fontname="Helvetica"; fontsize=10; fontcolor="#0e8f84";
    color="#9fd6cf"; style="rounded"; c11; c12; c2; }
  subgraph cluster_h { label="Classification head"; fontname="Helvetica"; fontsize=10; fontcolor="#4b5868";
    color="#e3e8ef"; style="rounded"; gap; bn; d1; d2; sm; }
  feat [label="Dense block 4\\nfeatures", style="rounded,dashed", fillcolor="#ffffff", color="#7a8696"];
  { rank=same; feat; c11; c12; c2; gap; bn; d1; d2; sm; }
  in -> feat [style=invis];
  in -> stem -> db1 -> db2 -> db3 -> db4;
  feat -> c11;
  c11 -> c12 -> c2 -> gap -> bn -> d1 -> d2 -> sm;
}
""",
        use_container_width=True,
    )
    ui.html_table(
        ["Layer", "Configuration", "Evidence"],
        [
            ["DenseNet201 backbone", "4 dense blocks (6/12/48/32 layers), 3 transition layers (1×1 conv + 2×2 avg-pool); ImageNet weights", "Report text"],
            ["CONV1_1", "3×1 convolution, 128 filters, ReLU, 'same' padding", "Report (filters) + source paper (kernel)"],
            ["CONV1_2", "1×3 convolution, 128 filters, ReLU", "Report text"],
            ["CONV2", "3×3 convolution, 64 filters, ReLU; Grad-CAM target", "Report Fig. 5 + source paper"],
            ["Head", "GAP → BatchNorm → Dense(256, ReLU) → Dense(2) → softmax", "Report Fig. 4; Dense 1 width reconstructed"],
        ],
    )
    st.caption("Factorising a 3×3 convolution into 3×1 followed by 1×3 keeps the receptive field while reducing "
               "parameters and computation. This is the motivation the report gives.")
    st.write("")
    st.markdown("#### Comparison architectures")
    cols = st.columns(3)
    comps = [
        ("MobileNetV2", "Depthwise-separable inverted-residual network (Fig. 4 of the report). ImageNet-pretrained, shared head."),
        ("ResNet50", "Residual network; depth not given in the report, so the standard ResNet50 is used. ImageNet-pretrained."),
        ("Xception", "Depthwise-separable 'extreme Inception' network; original Keras ImageNet weights."),
        ("GoogleNet", "The notebook's own Inception-v1 variant: 64×64 input, two inception modules, Dense(512) + dropout 0.5, trained from scratch. Reconstructed exactly from screenshots."),
        ("DenseNet201 (ablation)", "Same backbone and head as the primary model but without the three extra convolutions. Isolates the effect of the modification."),
    ]
    for i, (t, d) in enumerate(comps):
        with cols[i % 3]:
            ui.card(t, f"<p>{d}</p>")
            st.write("")

# ------------------------------------------------------------------- training
with tabs[4]:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.card(
            "Two-phase transfer learning",
            "<ul><li><b>Phase 1 (transfer learning):</b> backbone frozen; train the additional layers and head with "
            "Adam (lr 1e-3) for up to 10 epochs.</li>"
            "<li><b>Phase 2 (fine-tuning):</b> also unfreeze the last backbone stage (DenseNet dense block 4); "
            "Adam (lr 1e-4) for up to 10 epochs.</li>"
            "<li>Cross-entropy loss on a 2-way softmax, batch size 32 (from the original notebook).</li>"
            "<li>Training batches are class-balanced by oversampling the malignant images.</li>"
            "<li>Validation ROC-AUC drives checkpoint selection, learning-rate reduction on plateau (×0.5, patience 2) "
            "and early stopping (patience 4).</li>"
            "<li>Checkpoints every epoch; interrupted runs resume with <code>--resume</code>.</li></ul>",
            kicker="Training",
        )
    with c2:
        ui.card(
            "Evaluation protocol",
            "<ul><li>Held-out, lesion-disjoint <b>test split</b>, never used for training or model selection.</li>"
            "<li>Accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, confusion "
            "matrix and per-diagnosis rates. Positive class = malignant.</li>"
            "<li>95% confidence intervals from 1,000 lesion-level bootstrap resamples (images of one lesion "
            "stay together).</li>"
            "<li>Model differences tested with a paired bootstrap on the same test images.</li>"
            "<li>Accuracy is reported next to the 'always benign' baseline (≈ 80.6%) because the classes are "
            "imbalanced.</li></ul>",
            kicker="Evaluation",
        )

# -------------------------------------------------------------------- gradcam
with tabs[5]:
    st.markdown(
        """
#### Gradient-weighted Class Activation Mapping
For a class *c* and the feature maps *A<sup>k</sup>* of the last convolutional layer, Grad-CAM weights each map by
its average gradient, α<sub>k</sub> = mean<sub>i,j</sub> ∂y<sup>c</sup>/∂A<sup>k</sup><sub>ij</sub>, and computes
ReLU(Σ<sub>k</sub> α<sub>k</sub>A<sup>k</sup>). The result is upsampled to the image size and overlaid on it.

- **Target layer:** CONV2, the last convolution before global average pooling (64 × 7 × 7 at 224 px). For the
  comparison models it is the backbone's final convolutional stage.
- **Generated per request:** every heat map is computed from the uploaded image and the model's prediction. No
  static example images are shown.
- **Either class can be explained:** the prediction page can show evidence for *benign* or *malignant*.
- **Caveat:** Grad-CAM shows where the model looked, not whether it was right. Heat maps on rulers, hair or
  vignetting reveal shortcut learning.
""",
        unsafe_allow_html=True,
    )
    gv = state.read_json(cfg.results_dir(cfg.primary_model) / "gradcam_validation.json")
    st.markdown("#### Quantitative validation of the Grad-CAM maps")
    if gv:
        rows = [[k, ui.fmt_pct(gv["cam_drop_mean"][k]), ui.fmt_pct(gv["random_drop_mean"][k]),
                 ui.fmt_pct(gv["cam_beats_random_fraction"][k])] for k in gv["cam_drop_mean"]]
        ui.html_table(["Area blurred", "Confidence drop: Grad-CAM regions", "Confidence drop: random regions",
                       "Images where Grad-CAM > random"], rows, numeric_cols={1, 2, 3})
        st.caption(
            f"Deletion test on {gv['n_images']} held-out test images (half benign, half malignant), target layer "
            f"`{gv['target_layer']}`. The most important pixels according to Grad-CAM are blurred and the predicted "
            "class probability is re-measured. The same is done for equally large random regions (3 draws). "
            "Model-randomisation check (Adebayo et al., 2018): after re-initialising the trained layers "
            f"({_module_names(gv['randomised_modules'])}), "
            f"the maps have a mean Spearman correlation of {gv['randomization_spearman_mean']:.2f} with the trained maps "
            "(low values mean the explanations depend on the learned weights)."
        )
    else:
        st.caption("Run `python scripts/validate_gradcam.py --model modified_densenet201` after training to add the "
                   "deletion and model-randomisation tests here.")

# ---------------------------------------------------------------------- stack
with tabs[6]:
    ui.html_table(
        ["Component", "Original (2024)", "Reconstruction"],
        [
            ["Language", "Python 3 (Anaconda), Jupyter", "Python 3.13"],
            ["Deep learning", "TensorFlow / Keras", "PyTorch 2.10 + torchvision 0.25 + timm (Xception)"],
            ["Hardware", "Core i7/i9, ≥ 8 GB RAM", "Apple M1, 8 GB (Metal / MPS GPU)"],
            ["Data", "HAM10000 via Kaggle", "HAM10000 via Harvard Dataverse (MD5-verified)"],
            ["Evaluation", "Keras accuracy, matplotlib", "scikit-learn metrics, bootstrap CIs, matplotlib + Plotly"],
            ["Interface", "Notebook", "Streamlit web application"],
        ],
    )
    st.caption(
        "Why PyTorch: in this environment TensorFlow 2.20 segfaults on import alongside Streamlit's pyarrow, and it cannot "
        "use the M1 GPU on Python 3.13. torchvision's DenseNet201 is topologically identical to Keras' and uses the same "
        "input normalisation."
    )

# -------------------------------------------------------------- limitations
with tabs[7]:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
#### Research limitations
- **Not clinically validated.** No prospective study, no regulatory approval, no clinical use.
- **Dataset bias.** HAM10000 comes mostly from fair-skinned European and Australian populations and
  dermatoscopic devices. Performance on other skin types, devices or ordinary phone photos is unknown and may be poor.
- **Label granularity.** Seven diagnoses are collapsed into two classes. Actinic keratoses are pre-malignant or
  in-situ lesions, grouped here as malignant by convention.
- **Imbalance.** Only about 19.5% of images are malignant. Accuracy alone is misleading; see balanced accuracy,
  sensitivity and specificity.
- **Confidence is not risk.** Softmax confidence is uncalibrated model certainty, not the probability of cancer.
- **Coarse explanations.** The 7×7 Grad-CAM resolution localises regions, not boundaries.
"""
        )
    with c2:
        st.markdown(
            """
#### Deviations from the original
- **Framework:** PyTorch instead of TensorFlow/Keras (same architectures and weights family).
- **Leakage-free protocol:** lesion-grouped split and training-only online augmentation. The original approach
  (augmenting before splitting) can leak augmented copies of test images into training.
- **Reconstructed details:** contrast-stretch percentiles, augmentation probabilities, Dense 1 width (256), ReLU in
  the added convolutions, learning rates, phase schedule and class balancing were not documented.
- **Comparison models** use the shared classification head, because their original heads are unknown.
- **Results:** the report's Table 1 values are shown only as historical figures, because they are internally
  inconsistent and contradict the report's own training plots.
"""
        )
    st.write("")
    ui.disclaimer()

st.write("")
with st.expander("References"):
    st.markdown(
        """
1. Rehman, M. Z. U., Ahmed, F., Alsuhibany, S. A., Jamal, S. S., Ali, M. Z., Ahmad, J. *Classification of Skin Cancer
   Lesions Using Explainable Deep Learning.* Sensors 22(18), 6915 (2022). doi:10.3390/s22186915
2. Tschandl, P., Rosendahl, C., Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic
   images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018). doi:10.7910/DVN/DBW86T
3. Huang, G., Liu, Z., van der Maaten, L., Weinberger, K. Q. *Densely Connected Convolutional Networks.* CVPR (2017).
4. Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV (2017).
5. Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR (2018).
6. He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR (2016).
7. Chollet, F. *Xception: Deep Learning with Depthwise Separable Convolutions.* CVPR (2017).
8. Szegedy, C. et al. *Going Deeper with Convolutions.* CVPR (2015).
"""
    )
ui.footer()
