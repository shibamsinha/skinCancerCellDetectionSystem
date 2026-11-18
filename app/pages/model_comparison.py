"""Model comparison: measured results vs. historical (2024 report) values, kept strictly apart."""

from __future__ import annotations

import streamlit as st

from app.components import charts, state, ui
from skincancer.data import DX_FULL_NAMES

statuses = state.model_statuses()
evaluated = [(s, state.model_metrics(s.name)) for s in statuses if s.evaluated]
evaluated = [(s, m) for s, m in evaluated if m]

ui.page_header(
    "Model comparison",
    "Results measured in this reconstruction, shown separately from the historical values printed in the 2024 report.",
)

tab_measured, tab_details, tab_hist = st.tabs(
    ["Reproduced results", "Per-model details", "Historical results (2024 report)"]
)

METRICS = [
    ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"),
    ("sensitivity", "Sensitivity (recall)"), ("specificity", "Specificity"), ("f1", "F1-score"), ("roc_auc", "ROC-AUC"),
]

# ----------------------------------------------------------------- measured
with tab_measured:
    if not evaluated:
        ui.notice("<b>No evaluated models yet.</b> Run <code>python scripts/train.py --model modified_densenet201</code> "
                  "(training includes test-set evaluation) or <code>python scripts/evaluate.py --all</code>.", "warn")
    else:
        n_test = evaluated[0][1]["metrics"]["n"]
        n_mal = evaluated[0][1]["metrics"]["n_malignant"]
        baseline = evaluated[0][1]["metrics"]["majority_class_accuracy"]
        ui.notice(
            f"All values were computed by <code>scripts/evaluate.py</code> on the same held-out test set of "
            f"<b>{n_test:,} images</b> ({n_mal} malignant). No lesion in it appears in training or validation. "
            "Positive class = malignant; decision rule = softmax argmax. "
            f"A classifier that always answers <i>benign</i> would score <b>{ui.fmt_pct(baseline)}</b> accuracy, "
            "so balanced accuracy, sensitivity, specificity and ROC-AUC are more informative."
        )
        st.write("")
        show_ci = st.toggle("Show 95% bootstrap confidence intervals", value=False)
        headers = ["Model"] + [label for _, label in METRICS] + ["Best epoch"]
        rows = []
        for s, res in evaluated:
            m, ci = res["metrics"], res.get("ci95", {})
            cells = [s.display_name + ("  ★" if s.is_primary else "")]
            for key, _ in METRICS:
                val = f"{m[key]:.3f}" if key == "roc_auc" else ui.fmt_pct(m[key])
                if show_ci and key in ci:
                    lo, hi = ci[key]
                    val += f" [{lo:.3f}–{hi:.3f}]" if key == "roc_auc" else f" [{100 * lo:.1f}–{100 * hi:.1f}]"
                cells.append(val)
            ck = res["checkpoint"]
            cells.append(f"{ck['epoch']} ({ck['phase'].replace('_', ' ')})")
            rows.append(cells)
        ui.html_table(headers, rows, numeric_cols=set(range(1, len(METRICS) + 1)))
        ci_method = evaluated[0][1].get("ci_method", "1,000 image-level bootstrap resamples of the test set")
        st.caption(f"★ primary model. Confidence intervals: {ci_method}.")

        paired = (state.results_summary() or {}).get("paired_vs_primary") or {}
        comparisons = paired.get("comparisons") or {}
        if comparisons:
            names = {s.name: s.display_name for s in statuses}
            primary_label = names.get(paired.get("reference", ""), "primary model")
            st.write("")
            st.markdown(f"##### Are the differences real? {primary_label} vs. each model on the same test images")

            def _delta(d: dict, auc: bool = False) -> str:
                if auc:
                    return f"{d['difference']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
                return (f"{100 * d['difference']:+.1f} pts [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}]")

            def _verdict(d: dict) -> str:
                if d["ci_low"] > 0:
                    return f"{primary_label} better"
                if d["ci_high"] < 0:
                    return "Comparison model better"
                return "No clear difference"

            def _p(d: dict) -> str:
                floor = 2 / max(d.get("n_boot", 2000), 1)
                return f"< {floor:.3f}" if d["p_value"] < floor else f"{d['p_value']:.3f}"

            prow = []
            for name, c in comparisons.items():
                prow.append([names.get(name, name), _delta(c["roc_auc"], auc=True), _p(c["roc_auc"]),
                             _verdict(c["roc_auc"]), _delta(c["balanced_accuracy"]), _delta(c["sensitivity"]),
                             _delta(c["specificity"])])
            ui.html_table(["Comparison model", "Δ ROC-AUC [95% CI]", "p", "ROC-AUC verdict", "Δ Balanced accuracy",
                           "Δ Sensitivity", "Δ Specificity"], prow, numeric_cols={1, 2, 4, 5, 6})
            st.caption(
                f"Positive Δ favours {primary_label}. Method: {paired.get('method', 'paired bootstrap')}. "
                "ROC-AUC compares how well the models rank lesions across all thresholds. Sensitivity and specificity "
                "differences at the fixed 0.5 threshold mostly reflect a different operating point (one model flags "
                "more lesions as malignant), not better or worse discrimination."
            )

        st.write("")
        a, b = st.columns([1, 1.15], gap="large")
        with a:
            choice = st.segmented_control("Metric", [label for _, label in METRICS], default="Balanced accuracy")
            key = next(k for k, label in METRICS if label == (choice or "Balanced accuracy"))
            fig = charts.metric_bars(
                [s.name for s, _ in evaluated], [s.display_name for s, _ in evaluated],
                [res["metrics"][key] for _, res in evaluated], choice or "Balanced accuracy",
                baseline=baseline if key == "accuracy" else (0.5 if key in ("balanced_accuracy", "roc_auc") else None),
                baseline_label="always-benign baseline" if key == "accuracy" else "chance",
            )
            st.plotly_chart(fig, use_container_width=True, config=charts.CONFIG)
        with b:
            curves = {s.name: {**res["roc"], "auc": res["metrics"]["roc_auc"], "label": s.display_name} for s, res in evaluated}
            st.plotly_chart(charts.roc_comparison(curves), use_container_width=True, config=charts.CONFIG)

        not_eval = [s.display_name for s in statuses if not s.evaluated]
        if not_eval:
            st.caption("Not yet trained or evaluated (no metrics shown): " + ", ".join(not_eval))

# ------------------------------------------------------------------ details
with tab_details:
    with_hist = [s for s in statuses if state.model_history(s.name)]
    if not with_hist:
        ui.notice("No training history available yet.", "warn")
    else:
        pick = st.selectbox("Model", [s.name for s in with_hist],
                            format_func=lambda n: next(s.display_name for s in with_hist if s.name == n))
        history = state.model_history(pick)
        res = state.model_metrics(pick)
        summ = state.training_summary(pick)

        st.markdown("##### Training and validation curves")
        c1, c2 = st.columns(2, gap="large")
        c1.plotly_chart(charts.training_curves(history, "accuracy"), use_container_width=True, config=charts.CONFIG)
        c2.plotly_chart(charts.training_curves(history, "loss"), use_container_width=True, config=charts.CONFIG)
        monitor = (summ or {}).get("monitor", "val_roc_auc")
        best_epoch = (summ or {}).get("best_epoch") or (res or {}).get("checkpoint", {}).get("epoch")
        best = next((h for h in history if h["epoch"] == best_epoch), None) or max(history, key=lambda h: h["val_roc_auc"])
        st.caption(
            f"{len(history)} epochs recorded. Checkpoint selected by {monitor.replace('_', ' ')}: epoch {best['epoch']} "
            f"({best['phase'].replace('_', ' ')}, validation ROC-AUC {best['val_roc_auc']:.3f}). Those weights are used "
            "for testing and in the app. "
            "Training accuracy is measured on class-balanced, augmented batches, so it is not directly comparable "
            "with validation accuracy on the natural class distribution."
        )

        if res:
            st.markdown("##### Test-set confusion matrix and per-diagnosis behaviour")
            d1, d2 = st.columns([1, 1.1], gap="large")
            with d1:
                cm = res["confusion_matrix"]
                st.plotly_chart(charts.confusion_heatmap(cm["matrix"], cm["labels"]), use_container_width=True,
                                config=charts.CONFIG)
            with d2:
                rows = []
                for dx, v in res["per_diagnosis"].items():
                    group = "Malignant" if dx in ("mel", "bcc", "akiec") else "Benign"
                    meaning = "sensitivity" if group == "Malignant" else "false-positive rate"
                    rows.append([DX_FULL_NAMES.get(dx, dx), group, f"{v['n']:,}", ui.fmt_pct(v["predicted_malignant_rate"]), meaning])
                rows.sort(key=lambda r: (r[1] != "Malignant", r[0]))
                ui.html_table(["Diagnosis", "Class", "Test images", "Predicted malignant", "Meaning"], rows, numeric_cols={2, 3})
        else:
            ui.notice("This model has a training history but has not been evaluated on the test set yet.", "warn")
        if summ:
            st.caption(
                f"Device: {summ['device']} · batch size {summ['batch_size']} · imbalance handling: "
                f"{summ['imbalance_strategy']} · seed {summ['seed']}"
            )

# --------------------------------------------------------------- historical
with tab_hist:
    ui.notice(
        "<b>Historical values, not measurements from this application.</b> The table reproduces Table 1 exactly "
        "as printed in the 2024 project report. The values are internally inconsistent and do not match the report's "
        "own training plots, so they are not used as reproduction targets (see the notes below and "
        "<code>RECONSTRUCTION_SPECIFICATION.md</code> §9).",
        "warn",
    )
    st.write("")
    st.markdown("##### Report Table 1, as printed")
    ui.html_table(
        ["Model", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1"],
        [
            ["Modified DenseNet201", "95.00%", "92.16%", "96.05%", "95.96%", "94.02%"],
            ["MobileNet", "80.00%", "83.96%", "87.06%", "87.02%", "85.46%"],
            ["Xception", "77.00%", "73.96%", "77.06%", "77.02%", "75.46%"],
            ["ResNet", "75.00%", "73.96%", "77.06%", "77.02%", "75.46%"],
        ],
        numeric_cols={1, 2, 3, 4, 5},
    )
    st.caption("Column headings in the report were 'Perfection, Sensitiveness, Specifications, Exactness, F1 Val'; "
               "their mapping to standard metric names is confirmed by the source paper's column order.")
    st.write("")
    st.markdown("##### Values visible in the report's own training plots and notebook screenshots")
    ui.html_table(
        ["Model", "Final validation accuracy (plot)", "Source"],
        [
            ["Modified DenseNet201", "≈ 0.785 (peak ≈ 0.79)", "Figs. 6–7"],
            ["MobileNet", "≈ 0.744 (peak ≈ 0.758)", "Figs. 8–9"],
            ["ResNet", "≈ 0.748 (peak ≈ 0.757)", "Figs. 10–11"],
            ["Xception", "≈ 0.747", "Figs. 12–13"],
            ["GoogleNet", "test accuracy 0.7089", "notebook screenshot"],
        ],
    )
    st.write("")
    st.markdown("##### Why the historical numbers are not used as targets")
    st.markdown(
        """
- **The plots contradict the table.** The report's only Modified DenseNet201 curves peak at about 79% validation
  accuracy, not 95%.
- **Some rows are mathematically impossible.** Accuracy must lie between sensitivity and specificity. MobileNet's
  80.00% lies outside [83.96%, 87.06%].
- **The values match a published paper.** The "Modified DenseNet201" row equals the *unmodified* DenseNet201 row of
  Rehman et al., *Sensors* 2022, 22(18):6915 (Table 4), apart from accuracy. The other rows equal that paper's
  Modified DenseNet201 row minus 10 or 20 percentage points. The paper used a different, balanced
  3,297-image dataset.
- **The trivial baseline.** On binary HAM10000, always predicting *benign* already scores about 80.5% accuracy.
"""
    )
ui.footer()
