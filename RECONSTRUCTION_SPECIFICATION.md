# Reconstruction Specification

**Project:** *Classification of Skin Cancer Using Modified Deep Learning Protocol*
**Original:** BCA Major Project, The Heritage Academy, Kolkata (MAKAUT), session 2021–24. The report's embedded images and screenshots are dated May 2024.
**Source document analysed:** `MajorReportWORD.docx` (34 pages, 21 non-decorative embedded figures, all text-box content extracted).
**Specification date:** 2026-11-18

---

## 0. How to read this document

Every technical statement is tagged with its **evidence class**:

| Tag | Meaning |
|---|---|
| **[DOC]** | Stated explicitly in the text of the original report. |
| **[FIG]** | Inferred from a figure, diagram or code screenshot embedded in the report. |
| **[SRC]** | Taken from the published paper that the report's methodology is adapted from (see §0.1). This is an *external* source, not the student project itself. |
| **[DATA]** | Verified against the real HAM10000 metadata (`HAM10000_metadata.csv`, 10,015 rows). |
| **[STD]** | Reconstructed using standard, widely accepted practice because no project-specific information exists. |
| **[N/R]** | Not recoverable from any available evidence. |

### 0.1 Key discovery: the methodology's source paper

The report's methodology sections are adapted, closely and often word for word, from:

> M. Zia Ur Rehman, F. Ahmed, S. A. Alsuhibany, S. S. Jamal, M. Zulfiqar Ali, J. Ahmad.
> **"Classification of Skin Cancer Lesions Using Explainable Deep Learning."** *Sensors* 2022, 22(18), 6915.
> doi:[10.3390/s22186915](https://doi.org/10.3390/s22186915). Open access (CC BY 4.0).

Evidence:
- The report's Fig. 4 (workflow) and Fig. 5 (architecture) reproduce the paper's Figs. 1 and 5.
- The report's DATA PREPROCESSING, PROPOSED METHODOLOGY and DenseNet201 paragraphs paraphrase the paper's §3.1, §3 and §3.2.2.
- The report says *"Refer to Table 2 for a comprehensive breakdown of the architectural specifications"*, but **the report has no Table 2**. The paper's Table 2 is exactly that table.
- The report's references overlap heavily with the paper's reference list (Dorj 2018, Kaur 2022, Khan 2022 on cucumber leaf disease, and others).

**Consequence:** the paper is the only surviving source for several architecture details that the report omits, including CONV1_1's kernel shape and CONV2's filter count. Where it is used, it is tagged **[SRC]**. The paper and the student project differ in important ways (dataset, results), and these are documented in §11.

---

## 1. Original project overview

| Item | Value | Evidence |
|---|---|---|
| Goal | Classify dermatoscopic skin-lesion images as **Benign** or **Malignant** | [DOC] (§5 "two types: malignant and benign"), [FIG] (Fig. 4 output nodes) |
| Primary model | Pre-trained **DenseNet201** modified with additional convolutional layers and a custom classification head | [DOC], [FIG] Fig. 4, Fig. 5 |
| Comparison models | MobileNet (Fig. 4 says **MobileNetV2**), ResNet, Xception, GoogleNet (Inception v1) | [DOC] §6, §7; [FIG] Fig. 4 and GoogleNet screenshots |
| Explainability | Grad-CAM for benign and malignant predictions | [FIG] Fig. 4 |
| Training workflow | "data augmentation to increase the dataset, followed by retraining the Deep Learning model through multiple phases" | [DOC] §5 Proposed Methodology |
| Headline claim | Modified DenseNet201 ≈ 95% accuracy; MobileNet ≈ 80%; Xception ≈ 77%; ResNet ≈ 75% | [DOC] §7, Table 1. §9 and §11 discuss reliability. |

The objectives listed in §3 of the report (robust model, data preparation, optimisation, evaluation with accuracy, sensitivity and specificity, comparison with existing methods, ethics, documentation) are generic and add no implementation detail.

---

## 2. Dataset information

### 2.1 What the report states
- **HAM10000**, "available on Kaggle", "approximately 10,000 dermatoscopic images", "from the ISIC collection". [DOC]
- Seven lesion categories: benign keratosis-like lesions, melanocytic nevi, dermatofibroma, melanoma, vascular lesions, basal cell carcinoma, actinic keratoses (misspelt "Accitinic Keratoses"). [DOC]
- "The dataset comprises two types: malignant and benign." [DOC] HAM10000 actually has 7 diagnostic labels, so the binary label must be derived (§2.3).
- The dataset has class imbalance and variable image quality. [DOC]

### 2.2 What the figures show
- **Fig. 1 ("dataset distribution")** is a seaborn pair-plot of per-image `Red_mean`, `Green_mean`, `Blue_mean` (each divided by the grey mean) and `Gray_mean`, coloured by `cell_type` across the 7 categories. It is exploratory analysis of the full 7-class HAM10000 data. [FIG]
- **Fig. 2 ("Age distribution for malignant cases")** is a histogram in 5-year bins, peaking at age 70 with about 308 cases. [FIG]
- **Fig. 3 ("Male vs female malignant cases")** shows male ≈ 1,230 and female ≈ 725. [FIG]

### 2.3 Benign/Malignant mapping, established exactly
The binary mapping is **never stated** in the report, so it was recovered by matching Fig. 2 and Fig. 3 against the real metadata. [DATA]

| Candidate "malignant" set | n | male | female | Matches Fig. 3? |
|---|---|---|---|---|
| mel | 1,113 | 689 | 424 | no |
| mel + bcc | 1,627 | 1,006 | 621 | no |
| **mel + bcc + akiec** | **1,954** | **1,227** | **727** | **yes** |
| mel + bcc + akiec + df | 2,069 | 1,290 | 779 | no |

The age histogram of `mel + bcc + akiec` reproduces Fig. 2 bin by bin (e.g. 70→307, 65→250, 75→214, 60→199, 55→194, 80→195, 85→153).

**Mapping used:**
- **Malignant (1):** `mel` (melanoma, 1,113), `bcc` (basal cell carcinoma, 514), `akiec` (actinic keratoses / intraepithelial carcinoma, 327). Total **1,954 (19.51%)**.
- **Benign (0):** `nv` (melanocytic nevi, 6,705), `bkl` (benign keratosis-like, 1,099), `vasc` (vascular, 142), `df` (dermatofibroma, 115). Total **8,061 (80.49%)**.

Clinical note: `akiec` covers actinic keratoses (pre-malignant) and Bowen's disease (in-situ squamous cell carcinoma). Grouping it as malignant is the convention in HAM10000 binary studies and is what the original project did.

### 2.4 Dataset facts relevant to leakage [DATA]
- 10,015 images of **7,470 unique lesions**. Many lesions have 2–6 images, so an image-level random split puts near-duplicate views of the same lesion in both train and test.
- `dx_type`: histopathology 5,340; follow-up 3,704; expert consensus 902; confocal 69.
- Metadata fields: `lesion_id, image_id, dx, dx_type, age, sex, localization, dataset`.

### 2.5 Split used by the original notebook
The GoogleNet screenshots show Keras progress bars of **201/201 steps** on the training set and **63/63 steps** on the test set, at `batch_size = 32`. [FIG]
- 63 test batches means 1,985–2,016 images. 20% of 10,015 = **2,003**. ✓
- 201 training batches means 6,401–6,432 images. 80% × 80% of 10,015 ≈ **6,410**. ✓

So the notebook most likely used **20% test, then 20% of the remainder for validation (64/16/20)**, probably with image-level `train_test_split`. Lesion grouping is [N/R]; image-level splitting is the default in such notebooks.

The source paper used a *different* dataset and split (§11, C1).

---

## 3. Original technology stack

| Component | Value | Evidence |
|---|---|---|
| Language | Python 3.x | [DOC] |
| Distribution | Anaconda | [DOC] |
| "NLTK Toolkit" | Listed as a requirement. It is an NLP library with no role in image classification; probably a template leftover. | [DOC]; contradiction C10 |
| OS | UNIX/Linux/Windows | [DOC] |
| Hardware | Core i7/i9, ≥ 8 GB RAM, ≥ 60 GB disk | [DOC] |
| Deep-learning framework | **TensorFlow / Keras** (`Conv2D`, `MaxPooling2D`, `concatenate`, `Input`, `Model`, `Flatten`, `Dense`, `Dropout`, `model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`, `model.fit(...)`, `model.evaluate(...)`; the progress-bar format is TF2/Keras 2.x) | [FIG] GoogleNet screenshots |
| Environment | Jupyter notebook (VS Code-style notebook UI; cell counters [53]–[65]); data held as in-memory NumPy arrays (`train_images`, `val_images`, `test_images`) | [FIG] |
| Plotting | matplotlib (Figs. 6–13), seaborn `pairplot` (Fig. 1), pandas bar plot (Fig. 3) | [FIG] |
| Source paper environment | Google Colab; framework unstated | [SRC] |

---

## 4. Image preprocessing pipeline

| Step | Specification | Evidence |
|---|---|---|
| Contrast stretching | Applied to the dataset as the first preprocessing stage, "for enhanced visual clarity". The algorithm and its parameters are **not given**. | [DOC], [FIG] Fig. 4; algorithm [N/R] |
| Resizing | **Not documented** for the pretrained models. The only explicit input size in the report is **64 × 64 × 3** (GoogleNet screenshot). The source paper states 224 × 224 × 3 for MobileNetV2. | [FIG] (64), [SRC] (224) |
| Normalisation | Not documented | [N/R] |
| Colour space | RGB (Fig. 1 uses R, G and B means) | [FIG] |

Fig. 1's caption says it shows contrast-stretched examples, but Fig. 1 is a statistics pair-plot, so the contrast-stretching example images are missing (C6).

---

## 5. Data augmentation methodology

| Item | Specification | Evidence |
|---|---|---|
| Techniques | Exactly three: **horizontal flip, rotation, noise addition** | [DOC] ("three distinct augmentation techniques"), [FIG] Fig. 4 |
| Noise | **Gaussian, variance 0.1** | [DOC] |
| Rotation angles | **±15° and ±45°** (clockwise and anticlockwise), "followed by horizontal flipping" | [SRC] §3.1 |
| Purpose | Enlarge the dataset (offline augmentation) and reduce generalisation error | [DOC] |
| Multiplicity | Paper: 3,297 → 7,327 images (benign ×2.07, malignant ×2.40), which near-balances the classes. Student project: [N/R]. | [SRC] |
| Noise scale | Whether variance 0.1 applies to [0,1] or [0,255] intensities is not stated | [N/R] |
| Augmentation before or after splitting | Paper: **after** augmentation (its Table 1 splits the augmented pool 70/20/10, which leaks augmented copies across splits). Student project: [N/R]. | [SRC] |

---

## 6. Model architectures

### 6.1 Modified DenseNet201 (primary model)

**Backbone, from the report:** pre-trained DenseNet201 (ImageNet) used as feature extractor; four dense blocks and three transition layers; dense connectivity; transition layers downsample with average pooling. [DOC]

**Fig. 5 (architecture diagram)** [FIG]:
```
Original model:  Conv → DenseBlock1 → Conv → DenseBlock2 → Conv → DenseBlock3 → Conv → DenseBlock4
Additional layers (appended after Dense Block 4):  CONV1_1 → CONV1_2 → CONV2
                                                                    └→ Classification Head
```
The report's conclusion confirms that **three** convolutional layers are appended. [DOC]

**Additional layers:**

| Layer | Filters | Kernel | Report [DOC] | Paper [SRC] |
|---|---|---|---|---|
| CONV1_1 | 128 | **3 × 1** | "128 filters" (kernel not stated) | "128 filters with a kernel size of 3 × 1" |
| CONV1_2 | 128 | **1 × 3** | "128 feature maps with a kernel size of 1 × 3" | same |
| CONV2 | **64** | **3 × 3** | named in Fig. 5 only | "64 filters with a kernel size of 3 × 3" |

Together, CONV1_1 and CONV1_2 factorise a 3 × 3 convolution into 3 × 1 and 1 × 3 to "mitigate computational complexity". [DOC]

**Classification head** (Fig. 4) [FIG]/[SRC]:
`GAP → Batch Normalization → Dense Layer 1 → Dense Layer 2 → Softmax`.
The paper: "a GAP layer, a batch normalization layer, and two dense layers. **The final dense layer is modified according to the desired requirements**". So Dense Layer 2 is the output layer with one unit per class, followed by softmax.

**Unknown details:**

| Detail | Status | Reconstruction choice |
|---|---|---|
| Activation of CONV1_1/1_2/2 | [N/R] | ReLU [STD] |
| Padding / stride | [N/R] | `same` padding, stride 1, keeping the 7 × 7 map at 224 px input [STD] |
| BatchNorm inside the additional convs | not mentioned | none (faithful to the text) |
| Dense Layer 1 units and activation | [N/R] | 256, ReLU [STD] |
| Dropout | not mentioned | none by default (configurable) |
| Number of output units | binary | 2, softmax [DOC]/[FIG] |
| Input size | [N/R] | 224 × 224 [SRC]/[STD], configurable |

Paper Table 2 lists the DenseNet201 stem as "2 × 2 max pool" and transition 1 as "3 × 3 max pool". Both are errors relative to real DenseNet201, which uses a 3 × 3 max-pool stem and 2 × 2 average-pool transitions (C12). Because the backbone is ImageNet-pretrained, the **standard** DenseNet201 topology is used unchanged.

### 6.2 Comparison architectures

| Model | Specification found | Reconstruction |
|---|---|---|
| MobileNet | Report text says "MobileNet" (cites Howard et al.). Fig. 4 and the paper say **MobileNetV2**. | **MobileNetV2** (ImageNet) [FIG]/[SRC] |
| ResNet | Depth not stated; text describes residual learning only | **ResNet50** (ImageNet), the most common Keras/TF choice [STD] |
| Xception | Text describes depthwise-separable convolutions only | **Xception** (ImageNet; Keras-ported weights) [STD] |
| GoogleNet | **Fully specified in screenshots** (§6.3) | Exact reconstruction, trained from scratch [FIG] |

Whether the comparison models used the same head, or the extra convolutions, is [N/R]. The report presents them as unmodified existing models, but its conclusion says three conv layers were added "at the culmination of each model" (C11). Reconstruction choice: comparison models use the **same classification head and training protocol, without the additional conv layers**. The extra layers are a per-model configuration switch, so a "Modified MobileNetV2" (Fig. 4) can also be trained.

### 6.3 GoogleNet / Inception v1 (from the screenshots) [FIG]

```python
inception_module(x, filters=[f0..f5]):
    b1 = Conv2D(f0, 1×1, relu, same)(x)
    b2 = Conv2D(f2, 3×3, relu, same)(Conv2D(f1, 1×1, relu, same)(x))
    b3 = Conv2D(f4, 5×5, relu, same)(Conv2D(f3, 1×1, relu, same)(x))
    b4 = Conv2D(f5, 1×1, relu, same)(MaxPooling2D(3×3, strides 1, same)(x))
    return concatenate([b1, b2, b3, b4], axis=-1)

Input(64, 64, 3)
Conv2D(64, 7×7, strides 2, relu, same)
MaxPooling2D(3×3, strides 2, same)          # partly hidden by an IDE tooltip; inferred
Conv2D(64, 1×1, relu, same)
Conv2D(192, 3×3, relu, same)
MaxPooling2D(3×3, strides 2, same)
inception_module([64, 96, 128, 16, 32, 32])      # → 256 channels
inception_module([128, 128, 192, 32, 96, 64])    # → 480 channels
MaxPooling2D(3×3, strides 2, same)               # → 4 × 4 × 480
Flatten → Dense(512, relu) → Dropout(0.5) → Dense(NUM_CLASSES, softmax)
compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 10, batch_size = 32
fit(train_images, train_labels, validation_data=(val_images, val_labels))
```
Reported output: training accuracy **0.8195** (loss 0.4598); test accuracy **0.7089** (loss 0.7801). [FIG]
`NUM_CLASSES` is not visible [N/R].

---

## 7. Training methodology and configuration

| Item | Report / figures | Paper | Reconstruction |
|---|---|---|---|
| Strategy | Transfer learning; "retraining ... through multiple phases" [DOC] | "transfer learning, fine-tuning ... hyperparameter tuning" | **Phase 1:** frozen ImageNet backbone, train new layers. **Phase 2:** fine-tune the last backbone stage. [STD] |
| Optimizer | Adam (GoogleNet) [FIG] | – | Adam for all models [FIG] |
| Loss | categorical cross-entropy (GoogleNet) [FIG] | – | Cross-entropy on 2-class softmax [FIG] |
| Batch size | 32 (GoogleNet) [FIG] | – | 32 [FIG] |
| Epochs | 10 in every curve (Figs. 6, 9, 11, 13) and GoogleNet code [FIG] | – | Phase 1 = 10 epochs [FIG]; Phase 2 = up to 10 epochs with early stopping [STD] |
| Learning rate | [N/R] | – | Phase 1 1e-3 (Keras Adam default); Phase 2 1e-4; ReduceLROnPlateau [STD] |
| Frozen layers | [N/R] | – | Phase 1: whole backbone. Phase 2: DenseNet `denseblock4` + `norm5`; MobileNetV2 last 5 stages; ResNet50 `layer4`; Xception exit flow. Backbone BatchNorm statistics stay frozen. [STD] |
| Class imbalance | Acknowledged, method [N/R] | Balanced through augmentation | Transparent random **oversampling of the minority class** with independent augmentation per draw; class weights available as an alternative [STD] |
| Checkpoint selection | [N/R] | – | Best **validation ROC-AUC**, which also drives LR-on-plateau (×0.5, patience 2) and early stopping (patience 4) [STD]. Validation loss (the Keras default) was tried first and rejected: training batches are class-balanced while validation keeps the natural 19–21% prevalence, so validation loss rose while ROC-AUC improved (epochs 1–3: loss 0.53→0.63→0.60, AUC 0.859→0.865→0.866) and would have selected epoch 1. ROC-AUC does not depend on prevalence. |
| Seeds | [N/R] | – | 42 throughout [STD] |
| Decision rule | softmax argmax [FIG] | softmax | argmax, equivalent to a 0.5 threshold |

---

## 8. Evaluation methodology

- The report's objectives name accuracy, sensitivity and specificity. [DOC]
- Table 1 has columns **Perfection, Sensitiveness, Specifications, Exactness, F1 Val**. These are reworded versions of the paper's **Accuracy, Sensitivity, Specificity, Precision, F1 Score**. The mapping is confirmed by column order, by matching values (§9), and by F1 = 2PR/(P+R) holding only under this mapping. [DOC]/[SRC]
- Which split (validation or test) produced the reported numbers is [N/R].
- **Reconstruction:** all metrics are computed on a held-out, lesion-disjoint **test** split that is never used for training, model selection or threshold tuning, with **malignant as the positive class**. Metrics: accuracy, precision, recall/sensitivity, specificity, F1, balanced accuracy, ROC-AUC, confusion matrix, per-diagnosis sensitivity, and training/validation curves.
- **Uncertainty [STD]:** 95% confidence intervals use a percentile bootstrap with 1,000 **lesion-level** resamples. The test set holds 2,014 images of fewer lesions, and images of one lesion are correlated, so resampling whole lesions avoids intervals that are too narrow. Model differences use a **paired** lesion-level bootstrap (2,000 resamples) on the same test images, which cancels the shared difficulty of the test set. No multiple-comparison correction is applied.

---

## 9. Original reported results

### 9.1 Table 1 as printed in the report [DOC]

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| Modified DenseNet201 | 95.00% | 92.16% | 96.05% | 95.96% | 94.02% |
| MobileNet | 80.00% | 83.96% | 87.06% | 87.02% | 85.46% |
| Xception | 77.00% | 73.96% | 77.06% | 77.02% | 75.46% |
| ResNet | 75.00% | 73.96% | 77.06% | 77.02% | 75.46% |

### 9.2 Results shown in the report's figures [FIG]

| Model | Train accuracy (epoch 0 → 9) | Validation accuracy (range; final) | Figure |
|---|---|---|---|
| "Modified DenseNet201" | 0.790 → 0.826 | 0.765–0.791; **≈0.785** | Fig. 6, Fig. 7 |
| MobileNet | 0.700 → 0.783 | 0.709–0.758; **≈0.744** | Fig. 8, Fig. 9 |
| ResNet | 0.682 → 0.775 | 0.698–0.757; **≈0.748** | Fig. 10, Fig. 11 |
| Xception | 0.686 → 0.764 | 0.706–0.747; **≈0.747** | Fig. 12, Fig. 13 |
| GoogleNet | 0.8195 (final evaluation) | **test 0.7089** | screenshots |

### 9.3 Consistency analysis of Table 1

1. **The figures do not support 95%.** The only curves labelled Modified DenseNet201 (Figs. 6 and 7) peak at **≈0.79 validation accuracy**. The text cites "Figures 7 and 8" for 95%, but Fig. 8 is MobileNet.
2. **Some rows are mathematically impossible.** Accuracy is a prevalence-weighted average of sensitivity and specificity, so it must lie between them.
   - MobileNet: 80.00% is **outside** [83.96%, 87.06%], so it cannot come from any confusion matrix.
   - Modified DenseNet201: accuracy 95.00% implies a malignant prevalence of ≈27%, while precision 95.96% with these sensitivity and specificity values implies ≈50%. No single confusion matrix produces all four numbers.
3. **Provenance.** The values match numbers published in the source paper:
   - The "Modified DenseNet201" row (92.16 / 96.05 / 95.96 / 94.02) is identical to the paper's **unmodified DenseNet201** row in its Table 4 (accuracy 94.09%), with accuracy changed to 95.00%.
   - The MobileNet row equals the paper's Modified DenseNet201 row (93.96 / 97.06 / 97.02 / 95.46) **minus 10 points**. The Xception and ResNet rows equal it **minus 20 points**, and those two rows are identical except for accuracy.
   - The paper's numbers come from a different dataset (3,297 balanced ISIC images, §11 C1).
4. **Below the trivial baseline.** On binary HAM10000 with natural class frequencies, always predicting "benign" scores **80.49%**. The reported 75–80% accuracies for MobileNet, Xception and ResNet, and every plotted validation curve, are at or below this baseline. That suggests the plotted runs were 7-class (majority baseline 66.95%) or used a different data composition [N/R].

**Implication:** Table 1 cannot serve as a reproduction target. The plotted curves (≈0.74–0.79 validation accuracy) are the only primary experimental evidence from the original project. The application therefore labels Table 1 **"historical, as printed in the 2024 report"**, shows it separately from measured results, and links to this provenance note.

### 9.4 Source paper results [SRC] (for context only; different dataset)

| Model | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| MobileNetV2 | 90.54% | 89.93% | 91.18% | 91.32% | 90.62% |
| Modified MobileNetV2 | 91.86% | 91.09% | 92.66% | 92.82% | 91.95% |
| DenseNet201 | 94.09% | 92.16% | 96.05% | 95.96% | 94.02% |
| Modified DenseNet201 | 95.50% | 93.96% | 97.06% | 97.02% | 95.46% |

---

## 10. Missing implementation details

| # | Missing detail | Reconstruction |
|---|---|---|
| M1 | Contrast-stretching algorithm and parameters | Linear percentile stretch (2nd–98th percentile → full range) [STD] |
| M2 | Input resolution for pretrained models | 224 × 224 [SRC]/[STD] |
| M3 | Normalisation | Each backbone's own ImageNet normalisation (mean/std; [-1, 1] for Xception) [STD] |
| M4 | Aspect-ratio handling | Direct resize (squash) 600×450 → 224×224, as Keras `load_img(target_size=...)` does [STD] |
| M5 | Augmentation probabilities and multiplicity | Online: flip p = 0.5; angle drawn from {0, ±15, ±45}; Gaussian noise p = 0.25 [STD] |
| M6 | Noise intensity scale | Variance 0.1 on [0, 1] intensities (σ ≈ 0.316), configurable [STD] |
| M7 | Activation, padding and stride of additional convs | ReLU, `same`, stride 1 [STD] |
| M8 | Dense Layer 1 width, dropout | 256 ReLU, no dropout [STD] |
| M9 | Learning rates, schedule, early stopping, monitored metric | §7 [STD] |
| M10 | Which backbone layers were frozen or fine-tuned | §7 [STD] |
| M11 | Class-imbalance strategy | Minority oversampling [STD] |
| M12 | Split grouping and stratification | Lesion-grouped, diagnosis-stratified 64/16/20 [FIG ratio]/[STD] |
| M13 | Random seeds | 42 [STD] |
| M14 | Grad-CAM target layer and implementation | CONV2, the last conv before GAP [STD] |
| M15 | ResNet depth, Xception input size | ResNet50; 224 px for all ImageNet backbones [STD] |
| M16 | Number of classes in the plotted runs | [N/R]; the reconstruction is binary [DOC] |
| M17 | Which split Table 1 was computed on | [N/R] |
| M18 | Contrast-stretching and augmentation example images (Fig. 1 caption) | [N/R] (missing from the report) |
| M19 | Original trained weights and notebook | [N/R] (lost) |

---

## 11. Contradictions and inconsistencies

| # | Contradiction | Resolution |
|---|---|---|
| C1 | **Dataset.** The report says HAM10000 (~10,000 images, 7 classes). The source paper used Kaggle "Skin Cancer: Malignant vs. Benign" (3,297 ISIC images, 1,800 benign / 1,497 malignant). The report's own Figs. 1–3 and split sizes match HAM10000. | Use **HAM10000** [DOC]/[DATA]. |
| C2 | Report Table 1 contradicts the report's own curves (≈0.79 vs 95%). | Documented in §9.3. Neither is a target. |
| C3 | Table 1 rows are internally impossible (accuracy outside the sensitivity–specificity range; incompatible prevalences). | Documented in §9.3. |
| C4 | The text cites the wrong figures: "Figures 7 and 8 ... modified DenseNet201" (Fig. 8 is MobileNet); "Figures 12 and 13 ... ResNet ≈ 75%" (Figs. 12–13 are Xception); the text gives ResNet ≈ 77% while Table 1 gives Xception 77% and ResNet 75%. | Model identity taken from the figure captions. |
| C5 | "20% of data were misled as malignant" treats total error as if it were the false-positive rate. | Reconstruction reports FP and FN separately. |
| C6 | The Fig. 1 caption claims contrast-stretching and augmentation examples, but Fig. 1 is a statistics pair-plot. | Examples are missing (M18). |
| C7 | "Two distinct Deep Learning models are evaluated" (copied from the paper's DenseNet201 + MobileNetV2), yet five architectures are compared. | All five implemented. |
| C8 | "MobileNet" (text, Howard et al. citation) vs "MobileNetV2" (Fig. 4). | MobileNetV2. |
| C9 | The "Table 2" architecture breakdown is referenced but absent. | Recovered from the paper's Table 2 [SRC]. |
| C10 | "NLTK Toolkit" listed as a requirement, while TensorFlow/Keras is not listed. | NLTK ignored. |
| C11 | Conclusion: three conv layers added "at the culmination of each model", while comparison models are presented as standard. | Configurable; standard by default. |
| C12 | The paper's Table 2 misdescribes DenseNet201 pooling layers. | Standard pretrained topology kept. |
| C13 | The paper is internally inconsistent: Modified DenseNet201 sensitivity is 96.96% in the text but 93.96% in Table 4 and the abstract; specificity is 97.03% in the abstract but 97.06% in the table. | Recorded only. |
| C14 | The paper split *after* augmentation (70/20/10 of 7,327 augmented images), so augmented copies of the same image can appear in train and test. | **Not replicated:** split first, augment only the training set. |

---

## 12. Assumptions required for reconstruction

1. **Framework: PyTorch 2.10 + torchvision 0.25 + timm**, approved by the user. TensorFlow 2.20 segfaults on import in this Anaconda environment (conda pyarrow conflict, which Streamlit needs), and on Python 3.13 it cannot use the M1 GPU. PyTorch's MPS backend runs DenseNet201 at ≈54 img/s on this machine. The architectures are framework-independent, and torchvision's DenseNet201 is topologically identical to `keras.applications.DenseNet201`. Keras' DenseNet preprocessing ("torch" mode: /255 then ImageNet mean/std) is identical to torchvision's. The timm `legacy_xception` weights are a port of the original Keras Xception weights.
2. **Softmax and loss.** Models output 2 logits. `CrossEntropyLoss` (log-softmax plus NLL) is mathematically identical to Keras softmax with `categorical_crossentropy`. Probabilities shown in the app are `softmax(logits)`.
3. **Contrast stretching** is preprocessing, not augmentation. It is applied to every image (train, validation, test and uploaded images).
4. **Augmentation** is applied online to the training set only. Each training draw independently samples the documented operations, which is equivalent in expectation to the paper's offline enlargement and avoids its leakage (C14).
5. **Binary labels** follow §2.3 (verified).
6. **Lesion-level grouping** is enforced in every split, as the user's requirements demand. This makes measured performance more honest and probably lower than any leaky protocol.
7. **Backbone BatchNorm layers stay in inference mode** during fine-tuning (Keras transfer-learning convention).
8. Hardware: Apple M1, 8 GB unified memory. Batch size 32 fits for every model.

---

## 13. Proposed implementation architecture

### 13.1 Repository layout

The suggested layout is kept, but core code lives inside one importable package, `skincancer/`. Top-level package names such as `models` or `training` would collide with other installed packages, and a single package makes `streamlit run app/main.py`, the CLI scripts and the tests import the same code.

```
skin_cancer_cell/
├── app/                              # Streamlit UI only (no ML logic)
│   ├── main.py                       # entry point: theme, navigation
│   ├── pages/  home.py · prediction.py · model_comparison.py · project_information.py
│   └── components/  theme.py · ui.py · charts.py · state.py
├── skincancer/                       # core library
│   ├── config.py                     # dataclass config + YAML loading (single source of truth)
│   ├── data/      metadata.py · validation.py · splits.py · dataset.py
│   ├── preprocessing/  contrast_stretching.py · augmentation.py · image_loader.py · transforms.py
│   ├── models/    base.py · modified_densenet201.py · mobilenet.py · resnet.py · xception.py · googlenet.py · registry.py
│   ├── training/  trainer.py · callbacks.py · checkpointing.py
│   ├── evaluation/  metrics.py · confusion_matrix.py · visualization.py · evaluate.py
│   ├── explainability/  gradcam.py
│   ├── inference/  predictor.py
│   └── utils/     seed.py · logging.py · device.py
├── scripts/  download_ham10000.py · prepare_data.py · train.py · evaluate.py · train_all.sh
├── configs/default.yaml
├── data/  (HAM10000 images + metadata; splits/)        ← git-ignored
├── checkpoints/<model>/{best.pt,last.pt}               ← git-ignored
├── results/<model>/{metrics.json, history.json, predictions.csv, *.png}, results/summary.json
├── tests/
├── requirements.txt · README.md · RECONSTRUCTION_SPECIFICATION.md · .gitignore
```

### 13.2 Data flow
```
HAM10000_metadata.csv ─► metadata.py (binary mapping) ─► validation.py (missing or corrupt files)
        └► splits.py (StratifiedGroupKFold by lesion_id, 64/16/20, seed 42) ─► data/splits/*.csv
image ─► image_loader (RGB, EXIF-safe) ─► resize ─► contrast stretch ─► [train only: flip / rotate / noise]
      ─► backbone normalisation ─► tensor ─► model ─► softmax ─► {Benign, Malignant}
                                                      └► Grad-CAM (target = CONV2) ─► heatmap and overlay
```

### 13.3 Model registry

| Key | Architecture | Input | Pretrained | Additional convs | Phases |
|---|---|---|---|---|---|
| `modified_densenet201` | DenseNet201 + CONV1_1/CONV1_2/CONV2 + head | 224 | ImageNet | yes | transfer → fine-tune |
| `densenet201` | DenseNet201 + head (ablation) | 224 | ImageNet | no | transfer → fine-tune |
| `mobilenet_v2` | MobileNetV2 + head | 224 | ImageNet | no (configurable) | transfer → fine-tune |
| `resnet50` | ResNet50 + head | 224 | ImageNet | no | transfer → fine-tune |
| `xception` | Xception + head | 224 | ImageNet | no | transfer → fine-tune |
| `googlenet` | Report's Inception v1 (§6.3) | 64 | none (from scratch) | n/a | single phase, 10 epochs |

### 13.4 Training and evaluation
- `python scripts/prepare_data.py`: validate images, build and save the lesion-grouped split, write a data report.
- `python scripts/train.py --model modified_densenet201 [--resume]`: phase-aware training. Each epoch saves `last.pt` (model, optimiser, scheduler, phase, epoch, history, RNG state); `best.pt` holds the best validation loss. Training stops early on validation-loss plateau.
- `python scripts/evaluate.py --model modified_densenet201` (or `--all`): test-set metrics with lesion-level bootstrap CIs, confusion matrix, ROC curve, curves, per-diagnosis sensitivity; updates `results/summary.json`, including paired bootstrap comparisons of the primary model against every other model.
- `python scripts/validate_gradcam.py --model modified_densenet201`: Grad-CAM deletion and model-randomisation tests (§14.4).

### 13.5 Application
Streamlit multipage app with four pages: **Home** (overview), **Prediction** (upload, model choice, results, Grad-CAM), **Model Comparison** (measured metrics only, plus a separately labelled historical table) and **Project Information**. Models are cached with `st.cache_resource`. Uploaded images are processed in memory and never written to disk. Medical and research disclaimers appear on every page that shows a prediction.

---

## 14. Measured results of the reconstruction

Everything in this section was measured by `scripts/evaluate.py` on 2026-11-18. It is not taken from the 2024 report. The test split holds **2,014 images (390 malignant) from 1,495 lesions**. No test lesion appears in training or validation, and the test split was not used for checkpoint selection, early stopping or threshold choice. Decision rule: softmax argmax (threshold 0.5). Hardware: Apple M1 (MPS), PyTorch 2.10, seed 42, one run per model. Raw values: `results/<model>/metrics.json`, `results/<model>/predictions.csv`, `results/summary.json`.

### 14.1 Test-set metrics

| Model | Best checkpoint (by val ROC-AUC) | ROC-AUC [95% CI] | Balanced acc. | Sensitivity [95% CI] | Specificity [95% CI] | Accuracy [95% CI] | Precision | F1 | TP / FN / FP / TN |
|---|---|---|---|---|---|---|---|---|---|
| **Modified DenseNet201** | epoch 17 (fine-tuning ep. 8) | **0.897** [0.877–0.916] | 81.4% | 84.1% [79.9–88.4] | 78.8% [76.2–81.1] | 79.8% [77.5–81.9] | 48.7% | 0.617 | 328 / 62 / 345 / 1279 |
| DenseNet201, unmodified (ablation) | epoch 14 (fine-tuning ep. 4) | 0.904 [0.885–0.923] | 81.8% | 83.3% [78.9–87.4] | 80.2% [77.6–82.4] | 80.8% [78.6–82.9] | 50.2% | 0.627 | 325 / 65 / 322 / 1302 |
| ResNet50 | epoch 16 (fine-tuning ep. 7) | 0.895 [0.874–0.914] | 80.5% | 78.2% [73.1–83.1] | 82.8% [80.3–85.1] | 81.9% [79.7–84.0] | 52.1% | 0.626 | 305 / 85 / 280 / 1344 |
| MobileNetV2 | epoch 17 (fine-tuning ep. 7) | 0.889 [0.868–0.910] | 80.2% | 74.9% [69.7–80.2] | 85.5% [83.0–87.6] | 83.4% [81.3–85.7] | 55.3% | 0.636 | 292 / 98 / 236 / 1388 |
| Xception | epoch 12 (fine-tuning ep. 2) | 0.867 [0.845–0.888] | 78.4% | 80.3% [75.6–84.5] | 76.5% [74.1–79.1] | 77.3% [75.0–79.5] | 45.1% | 0.577 | 313 / 77 / 381 / 1243 |
| GoogleNet (report notebook, from scratch, 64 px) | epoch 10 of 10 | 0.830 [0.805–0.854] | 75.0% | 86.2% [82.4–89.8] | 63.9% [61.2–66.8] | 68.2% [65.8–70.8] | 36.4% | 0.512 | 336 / 54 / 586 / 1038 |

Confidence intervals: 95% percentile bootstrap, 1,000 lesion-level resamples. Majority-class ("always benign") accuracy on this test set: **80.6%**.

### 14.2 Paired comparison with the primary model

Δ = Modified DenseNet201 minus the comparison model, on the same test images. Paired lesion-level bootstrap, 2,000 resamples, two-sided bootstrap p-value. No multiple-comparison correction is applied. Sensitivity, specificity and balanced-accuracy differences are in percentage points.

| Comparison model | Δ ROC-AUC [95% CI] | p | Δ Balanced acc. [95% CI] | Δ Sensitivity [95% CI] | Δ Specificity [95% CI] |
|---|---|---|---|---|---|
| DenseNet201, unmodified | −0.007 [−0.017, +0.003] | 0.143 | −0.3 [−2.1, +1.4] | +0.8 [−2.5, +3.8] | −1.4 [−3.1, +0.1] |
| ResNet50 | +0.002 [−0.013, +0.018] | 0.783 | +0.9 [−1.5, +3.4] | +5.9 [+1.5, +10.4] | −4.0 [−6.0, −2.0] |
| MobileNetV2 | +0.008 [−0.009, +0.025] | 0.384 | +1.3 [−1.5, +4.2] | +9.2 [+4.1, +14.4] | −6.7 [−8.8, −4.4] |
| Xception | **+0.030** [+0.013, +0.047] | < 0.001 | +3.0 [+0.5, +5.6] | +3.8 [−0.5, +8.4] | +2.2 [+0.1, +4.3] |
| GoogleNet | **+0.067** [+0.043, +0.090] | < 0.001 | +6.4 [+3.7, +9.2] | −2.1 [−6.8, +2.7] | +14.8 [+12.3, +17.4] |

### 14.3 Per-diagnosis behaviour

Fraction of test images predicted *malignant*. For mel, bcc and akiec this is the per-diagnosis sensitivity; for the benign diagnoses it is the false-positive rate.

| Model | mel | bcc | akiec | nv | bkl | df | vasc |
|---|---|---|---|---|---|---|---|
| Modified DenseNet201 | 80.2% | 89.0% | 89.7% | 16.3% | 50.7% | 54.5% | 20.6% |
| DenseNet201, unmodified | 76.1% | 92.0% | 94.1% | 14.2% | 52.7% | 54.5% | 26.5% |
| ResNet50 | 68.5% | 89.0% | 94.1% | 11.4% | 51.2% | 59.1% | 20.6% |
| MobileNetV2 | 62.2% | 91.0% | 92.6% | 8.7% | 48.8% | 59.1% | 14.7% |
| Xception | 73.9% | 86.0% | 92.6% | 17.2% | 60.0% | 72.7% | 20.6% |
| GoogleNet | 81.5% | 91.0% | 94.1% | 28.3% | 82.4% | 81.8% | 38.2% |
| *test images* | *222* | *100* | *68* | *1,363* | *205* | *22* | *34* |

Benign keratosis-like lesions (bkl) are the main source of false positives for every model: about half are flagged as malignant. Under dermoscopy, benign keratoses are known to resemble melanoma. Dermatofibroma and vascular lesions have very few test images, so their rates are imprecise.

### 14.4 Grad-CAM validation (Modified DenseNet201)

`scripts/validate_gradcam.py`, 200 test images (100 benign, 100 malignant), target layer `additional.conv2` (CONV2, 64 × 7 × 7). Output: `results/modified_densenet201/gradcam_validation.json`.

| Area blurred | Drop in predicted-class probability: Grad-CAM regions | Same, random regions (3 draws) | Images where Grad-CAM region > random |
|---|---|---|---|
| 10% | 42.8 points | 9.3 points | 65.0% |
| 20% | 46.8 points | 18.0 points | 67.0% |
| 40% | 46.4 points | 32.0 points | 62.5% |

Model-randomisation test (Adebayo et al., 2018): re-initialising the added convolution layers and the head changes the maps almost completely (mean Spearman ρ = 0.11, median 0.08). Grad-CAM therefore reflects what the trained layers learned rather than image structure alone.

### 14.5 Interpretation and comparison with the report

1. **The original claim is not reproduced.** The report states 95% accuracy for Modified DenseNet201 and a clear lead over MobileNet (80%), Xception (77%) and ResNet (75%). Measured, the pretrained models lie between 77.3% and 83.4% accuracy and between 0.867 and 0.904 ROC-AUC, and the primary model is not the most accurate. §9.3 already showed that Table 1 is internally inconsistent and copied from the source paper, so this is expected.
2. **The report's own plots are reproduced.** Its Modified DenseNet201 curves end at ≈0.785 validation accuracy (§9.2). The reconstruction reaches 81.4% validation and 79.8% test accuracy. The reported GoogleNet test accuracy of 70.89% compares with 68.2% here. This supports reading the plots, not Table 1, as the original project's real outcome.
3. **The architectural modification has no measurable benefit on HAM10000.** Against the identical pipeline without the three extra convolutions, ΔROC-AUC = −0.007 (95% CI −0.017 to +0.003) and Δbalanced accuracy = −0.3 points. The source paper reports +1.4 points of accuracy for the modification on its own balanced 3,297-image dataset (§9.4). That gain is not seen here.
4. **Pretrained models are statistically tied.** Modified DenseNet201, DenseNet201, ResNet50 and MobileNetV2 do not differ in ROC-AUC (all CIs of the paired difference include 0). Xception is significantly weaker. GoogleNet, trained from scratch at 64 × 64 px exactly as in the report's notebook, is the weakest. Its best epoch was its last, so it had not converged within the notebook's 10 epochs.
5. **Sensitivity and specificity differences reflect operating points.** At the fixed 0.5 threshold, Modified DenseNet201 is the most sensitive pretrained model and catches the most melanomas (80.2%), but with lower specificity than DenseNet201, ResNet50 and MobileNetV2. These differences are significant at 0.5, while ranking quality (ROC-AUC) is not. Any of these models could be moved to a different operating point by choosing a threshold on the validation set. This was deliberately not done, to keep the original argmax decision rule.

### 14.6 Limitations of these measurements

- One training run per model (seed 42). Seed-to-seed variation was not measured; it may be of the same order as the smaller differences above, which is another reason not to rank the tied models.
- Hyper-parameters the report does not document (§10, §12) were reconstructed once and not tuned per model. A per-model search might change the ranking among the tied models.
- One dataset (HAM10000, dermatoscopic images, mostly fair skin). No external validation set was used.
- Precision is about 0.45–0.55 at the 0.5 threshold, a consequence of class-balanced training and 19% prevalence. Roughly every second "malignant" prediction is a benign lesion.
