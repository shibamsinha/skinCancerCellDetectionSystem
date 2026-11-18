# Skin Cancer Classification: Modified Deep Learning Protocol

A reconstruction of the 2024 BCA major project *"Classification of Skin Cancer Using Modified Deep Learning
Protocol"* (The Heritage Academy, Kolkata). The original code and weights were lost. Everything here was rebuilt
from the surviving project report, retrained on HAM10000, and measured on a held-out test set.

It contains:

1. **A faithful research implementation**: Modified DenseNet201 (pretrained DenseNet201 + CONV1_1 3×1×128 →
   CONV1_2 1×3×128 → CONV2 3×3×64 + GAP/BatchNorm/Dense/Dense/softmax head), the comparison models (MobileNetV2,
   ResNet50, Xception, the notebook's GoogleNet, and an unmodified-DenseNet201 ablation), the documented
   preprocessing (contrast stretching; flip/rotation/Gaussian-noise augmentation), a leakage-free training and
   evaluation pipeline, and Grad-CAM.
2. **A Streamlit web application**: upload a dermatoscopic image, choose a model, and get a benign/malignant
   prediction with class probabilities, inference time, and a live Grad-CAM heat map and overlay. It also includes
   a model comparison page (measured results kept separate from the historical report values) and a project
   information page.

> **Not a medical device.** Predictions come from an experimental research model. They are not diagnoses and do
> not replace a dermatologist. Model confidence is not the probability that a patient has cancer.

See **[RECONSTRUCTION_SPECIFICATION.md](RECONSTRUCTION_SPECIFICATION.md)** for the full documentation analysis.
Every detail there is tagged as documented, inferred from figures, taken from the source paper, reconstructed with
standard practice, or unrecoverable. It also lists every contradiction found in the report.

---

## Key findings from the documentation analysis

- **Methodology source.** The report's methodology follows Rehman et al., *"Classification of Skin Cancer Lesions
  Using Explainable Deep Learning"*, *Sensors* 2022, 22(18):6915. That paper's Table 2 supplies the layer details
  the report refers to but omits: CONV1_1 kernel **3×1**, CONV2 **64 filters, 3×3**.
- **Binary mapping established exactly.** The report's Fig. 3 (male ≈1,230 / female ≈725 malignant) and its Fig. 2
  age histogram are reproduced only by **malignant = melanoma + basal cell carcinoma + actinic keratoses**
  (1,227 / 727 in the real metadata).
- **Split ratio recovered.** The GoogleNet screenshot's 201 train / 63 test batches of 32 imply a 64/16/20 split of
  all 10,015 images.
- **The report's Table 1 cannot be a reproduction target.** It contradicts the report's own training plots
  (validation accuracy ≈0.79 for the "95%" model). Some rows are mathematically impossible. Its values match
  numbers published in the source paper, which used a different dataset. The app shows them only as clearly
  labelled historical values.

## Quick start

```bash
# 1. Environment (Python 3.11-3.13)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Dataset: HAM10000 from Harvard Dataverse (~2.8 GB, MD5-verified, CC BY-NC 4.0)
python scripts/download_ham10000.py --output data/HAM10000

# 3. Validate images and create the lesion-grouped split
python scripts/prepare_data.py

# 4. Train + evaluate (the primary model first; or everything with train_all.sh)
python scripts/train.py --model modified_densenet201
bash scripts/train_all.sh

# 5. Launch the web application (local only)
streamlit run app/main.py --server.address localhost
```

The app runs at <http://localhost:8501>. Uploaded images are processed in memory and never written to disk.

**Only want the app?** Steps 2–4 are optional. The trained weights for all six models are published as a GitHub
release (`weights-v1`). When `checkpoints/<model>/best.pt` is missing, the app downloads that file on first use
(10–97 MB) and verifies its SHA-256 before loading it. Locally trained weights always take precedence.

### Running on a laptop (thermal and memory limits)

`configs/default.yaml` → `resources:` is tuned for a **fanless 8 GB MacBook Air**:

- lower CPU priority (`nice +10`) and 2 CPU threads
- 2 data-loading workers
- GPU memory capped at ~3.1 GB, so the process errors out instead of pushing macOS into heavy swapping
- a 0.35 s pause after every batch, doubled while macOS reports a *fair* thermal state
- a full pause whenever macOS reports *serious* until the machine cools

Training takes roughly twice as long but runs much cooler. On a machine with active cooling, set
`cooldown_seconds_per_batch: 0` and `thermal_guard: false`. Training can be stopped at any time and continued
with `--resume`.

### Deploying to Streamlit Community Cloud

The repository is ready for [Streamlit Community Cloud](https://share.streamlit.io) (free; apps get up to about
2.7 GB of memory and 2 CPU cores):

1. Sign in at <https://share.streamlit.io> with GitHub and click **Create app** → *Yup, I have an app*.
2. Repository `shibamsinha/skinCancerCellDetectionSystem`, branch `main`, main file path `app/main.py`.
3. Optionally choose a subdomain under **App URL** (`<name>.streamlit.app`). Custom top-level domains are not
   supported on Community Cloud.
4. **Advanced settings** → Python **3.12** or **3.13**. On these versions `requirements.txt` installs the CPU-only
   PyTorch wheels (≈190 MB instead of several GB of CUDA libraries). No secrets are needed.
5. Deploy. The first start installs dependencies (a few minutes). Each model's weights are downloaded the first time
   someone selects it. The app keeps at most two models in memory.

Uploaded images are processed in memory on Streamlit's servers and are not stored by the app. HAM10000 is
licensed CC BY-NC 4.0: keep the deployment non-commercial.

### Dataset setup without the download script

Download HAM10000 from Kaggle (`kmader/skin-cancer-mnist-ham10000`) or Harvard Dataverse
(doi:10.7910/DVN/DBW86T). Put **all 10,015 JPEGs in one folder** at `data/HAM10000/images/` and the metadata at
`data/HAM10000/HAM10000_metadata.csv`. Different paths can be set in `configs/default.yaml` (`data:` section).

### Model checkpoints

- Training writes `checkpoints/<model>/best.pt` (selected weights + the model and preprocessing config) and
  `checkpoints/<model>/last.pt` (full resume state) after every epoch.
- Resume an interrupted run with `python scripts/train.py --model <name> --resume`, or
  `bash scripts/train_all.sh --resume`.
- The app lists a model only if it has a local `best.pt` or a published checkpoint with a known SHA-256
  (`configs/default.yaml` → `weights`). It never shows predictions from untrained weights.
- Checkpoints contain only tensors and plain Python values and are loaded with `torch.load(weights_only=True)`.
- Checkpoints are not committed to git (up to 97 MB each). The `best.pt` files used for the reported results are
  attached to the `weights-v1` GitHub release as `<model>.pt`.

Available model keys (`python scripts/train.py --list`): `modified_densenet201` (primary), `densenet201`
(ablation), `mobilenet_v2`, `resnet50`, `xception`, `googlenet`.

## Project structure

```
app/                      Streamlit UI only
  main.py                 entry point (theme + navigation)
  pages/                  home, prediction, model_comparison, project_information
  components/             theme (CSS), ui (cards/notices), charts (Plotly), state (cached models & results)
skincancer/               core library (no UI code)
  config.py               typed configuration (dataclasses) loaded from configs/default.yaml
  data/                   metadata + binary mapping, image validation, lesion-grouped splits, Dataset/DataLoaders
  preprocessing/          contrast stretching, augmentation, safe image loading, transforms
  models/                 modified_densenet201, mobilenet, resnet, xception, googlenet, shared base + registry
  training/               phase-wise trainer, callbacks, checkpointing
  evaluation/             metrics (+ bootstrap CIs), confusion matrix, charts, test-set evaluation runner
  explainability/         Grad-CAM
  inference/              Predictor used by the app
scripts/                  download_ham10000.py, prepare_data.py, train.py, evaluate.py, train_all.sh
configs/default.yaml      all hyper-parameters, with evidence tags
tests/                    unit + integration tests (pytest)
results/                  metrics.json, history.json, predictions.csv, PNG charts per model; summary.json
```

The core library is a single importable package (`skincancer/`) rather than top-level `models/`, `training/`, etc.
Generic top-level names would collide with other installed packages, and a single package makes the scripts, the
tests and the Streamlit app import the same code.

## Methodology (reconstructed)

| Stage | Implementation | Evidence |
|---|---|---|
| Data | HAM10000, 10,015 images / 7,470 lesions; malignant = mel + bcc + akiec (19.5%) | report text + Figs. 2–3 verified |
| Split | Lesion-grouped, diagnosis-stratified 64/16/20 (`StratifiedGroupKFold`, seed 42) | ratio from screenshots; grouping added to prevent leakage |
| Preprocessing | Resize to 224² → contrast stretch (2nd–98th percentile) → backbone normalisation | technique documented; parameters reconstructed |
| Augmentation (train only) | H-flip p=0.5; rotation ∈ {0, ±15°, ±45°}; Gaussian noise σ²=0.1 (p=0.25) | ops + variance documented; angles from source paper |
| Imbalance | Minority oversampling (each draw freshly augmented) | reconstructed (paper balanced via augmentation) |
| Model | DenseNet201 (ImageNet) → CONV1_1 3×1×128 → CONV1_2 1×3×128 → CONV2 3×3×64 → GAP → BN → Dense(256) → Dense(2) → softmax | report Figs. 4–5 + paper Table 2 |
| Training | Phase 1: frozen backbone, Adam 1e-3, ≤10 epochs. Phase 2: + dense block 4, Adam 1e-4, ≤10 epochs. Batch 32, cross-entropy | Adam/CE/batch/epochs from screenshots; phases reconstructed |
| Selection | Best validation ROC-AUC; LR halved on plateau; early stopping (patience 4) | reconstructed (see spec §7) |
| Evaluation | Held-out test set: accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, confusion matrix, per-diagnosis rates, 95% bootstrap CIs | — |
| Explainability | Grad-CAM on CONV2 (64×7×7), any class, generated per upload | report Fig. 4 |

## Results

Measured by `scripts/evaluate.py` on the held-out test set: 2,014 images (390 malignant) from 1,495 lesions,
none of which appear in training or validation. Decision rule: softmax argmax. Stored in
`results/<model>/metrics.json` and `results/summary.json`. Confidence intervals and paired significance tests are
in the app's **Model comparison** page and in `RECONSTRUCTION_SPECIFICATION.md` §14.

| Model | ROC-AUC | Balanced acc. | Sensitivity | Specificity | Accuracy | Precision | F1 |
|---|---|---|---|---|---|---|---|
| **Modified DenseNet201** (primary) | 0.897 | 81.4% | 84.1% | 78.8% | 79.8% | 48.7% | 0.617 |
| DenseNet201, unmodified (ablation) | 0.904 | 81.8% | 83.3% | 80.2% | 80.8% | 50.2% | 0.627 |
| ResNet50 | 0.895 | 80.5% | 78.2% | 82.8% | 81.9% | 52.1% | 0.626 |
| MobileNetV2 | 0.889 | 80.2% | 74.9% | 85.5% | 83.4% | 55.3% | 0.636 |
| Xception | 0.867 | 78.4% | 80.3% | 76.5% | 77.3% | 45.1% | 0.578 |
| GoogleNet (report notebook, from scratch, 64 px) | 0.830 | 75.0% | 86.2% | 63.9% | 68.2% | 36.4% | 0.512 |

An "always benign" classifier would score 80.6% accuracy on this test set, so accuracy alone is not informative.

What the measurements show:

- **The modification does not measurably help.** Against the same DenseNet201 without the three extra
  convolutions, the ROC-AUC difference is −0.007 (paired 95% CI −0.017 to +0.003). The report's central claim,
  that the modified network is clearly superior, is not supported by this reconstruction.
- **The pretrained models are close.** Modified DenseNet201 is not significantly different from ResNet50 or
  MobileNetV2 in ROC-AUC. It is significantly better than Xception (+0.030) and the from-scratch GoogleNet
  (+0.067).
- **Differences in sensitivity and specificity are mostly operating points.** At the 0.5 threshold, Modified
  DenseNet201 flags more lesions as malignant: it has the highest sensitivity among the pretrained models and a
  lower specificity than DenseNet201, ResNet50 or MobileNetV2. MobileNetV2 has the highest accuracy because it
  flags the fewest lesions.
- **The report's Table 1 is not reproduced** (95% / 80% / 77% / 75% accuracy). Measured accuracies of 77–83% for
  the pretrained models agree with the report's own training plots (validation accuracy ≈ 0.75–0.79). GoogleNet's
  68.2% is close to the 70.89% visible in the report's notebook screenshot.
- **Grad-CAM is faithful to the model** (200 test images, Modified DenseNet201). Blurring the 10% of the image
  that Grad-CAM marks as most important lowers the predicted-class probability by 42.8 points on average, against
  9.3 points for random regions of the same size. Re-initialising the trained layers changes the maps almost
  completely (mean Spearman ρ = 0.11).

## Testing

```bash
python -m pytest -q tests
```

The tests cover contrast stretching, augmentation (including the documented noise variance), image-upload
validation (corrupt, empty, GIF, tiny, RGBA, grayscale), the binary mapping on the real metadata, split leakage
and determinism, the exact layer configuration of the modified architecture, phase freezing, BatchNorm handling,
every comparison model, Grad-CAM (shape, range, hooks cleaned up, both classes), metrics, checkpoint round-trips,
RNG restore, and end-to-end train → resume → evaluate → predict → explain runs on a synthetic dataset.

## Framework choice

The original used TensorFlow/Keras. This reconstruction uses **PyTorch** for two reasons. In the development
environment (Anaconda, Python 3.13), TensorFlow 2.20 segfaults on import because it conflicts with the conda
`pyarrow` that Streamlit requires. TensorFlow also cannot use the Apple-silicon GPU on Python 3.13, while PyTorch's
MPS backend runs DenseNet201 several times faster. torchvision's DenseNet201 is topologically identical to
`keras.applications.DenseNet201` and uses the same input normalisation. The Xception weights are a port of the
original Keras weights.

## Known limitations and deviations

- Research demonstration only: not clinically validated, not approved, and not a substitute for a dermatologist.
- HAM10000 is dominated by fair-skinned populations and dermatoscopic images. Phone photos, other devices and other
  skin types are out of distribution.
- Confidence (softmax) is uncalibrated model certainty, not the probability of cancer.
- Grad-CAM at 7×7 resolution localises regions coarsely. It explains model behaviour; it does not validate it.
- The reconstructed modification (three extra convolutions after dense block 4) performs on par with the
  unmodified DenseNet201, not better. The original project's claimed advantage could not be confirmed.
- The decision threshold is fixed at 0.5 (softmax argmax, as in the original). With class-balanced training this
  favours sensitivity over precision: about half of the "malignant" predictions on the test set are benign lesions.
- All results come from one training run per model (seed 42). Run-to-run variation is not measured.
- Deviations from the original (all documented in the spec): PyTorch instead of Keras; a lesion-grouped split with
  training-only augmentation instead of augmenting before splitting; undocumented hyper-parameters reconstructed;
  comparison models share the classification head; ROC-AUC used for checkpoint selection.

## Attribution

- HAM10000: Tschandl P., Rosendahl C., Kittler H. *Sci. Data* 5, 180161 (2018), doi:10.7910/DVN/DBW86T,
  CC BY-NC 4.0 (non-commercial use only).
- Methodology source: Rehman M.Z.U. et al., *Sensors* 22(18), 6915 (2022), doi:10.3390/s22186915.
