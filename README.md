# A 272-Layer PyramidNet for CIFAR-10 Under Five Million Parameters

- **Course**: NYU Tandon Deep Learning, Fall 2026 — Project 1
- **Authors**: Jingyu He, Hongda Wu
- **Kaggle Team**: `Hongda Wu`
- **Report Document**: [`Project1_Report.docx`](./Project1_Report.docx)
- **Assignment Specification**: [`project1.pdf`](./project1.pdf)

---

## 1. Execution Introduction of this Repository

### 1.1 Repository Organization and File Functionalities

```text
Project1_Result/
├── README.md                                              # Repository execution guide and report summary
├── Project1_Report.docx                                   # Final project report (Word document)
├── project1.pdf                                           # Official assignment prompt and grading rubric
└── Archive/                                               # Self-contained training, launch, and evaluation notebooks
    ├── A_resnet.ipynb                                     # Family A: BasicBlock ResNet ([3,5,3] × [64,128,256])
    ├── B_wrn.ipynb                                        # Family B: Pre-activation Wide ResNet (WRN-16/22/28)
    ├── C_seresnet.ipynb                                   # Family C: Squeeze-and-Excitation ResNet (SE r=16/8, IBN)
    ├── E_pyramid.ipynb                                    # Family E baseline: Basic-block PyramidNet-110 (α=98)
    ├── E_pyramiad_HW.ipynb                                # Family E optimizer screen: SAM & ASAM on PyramidNet-110 (T106/T111/T112)
    ├── F_shake.ipynb                                      # Family F: Shake-Shake-26 2×40d (1,800-epoch schedule)
    ├── G_resnext.ipynb                                    # Family G: ResNeXt-29 32×4d (grouped convolutions)
    ├── E_trivialaug_ibn.ipynb                             # Family E ablation: TrivialAugmentWide, IBN-a, MixStyle, DropPath
    ├── E_pyramid_distill.ipynb                            # Family E ablation: Knowledge distillation from multi-model ensemble
    ├── E_pyramid_t106_cutmix.ipynb                        # Family E ablation: CutMix regularization on T106 recipe
    ├── E_pyramid_t125_queue.ipynb                         # Family E queue: CutMix × schedule length, reflect pad, label smoothing
    ├── E_pyramid_t106_shakedrop.ipynb                     # Family E core template: Bottleneck PyramidNet + ShakeDrop + ASAM
    ├── E_pyramid_t106_shakedrop_T123_launch.ipynb         # Launch copy (T123): Bottleneck PyramidNet-164 (α=100), SGD 1,800 epochs
    ├── E_pyramid_t106_shakedrop_T127_launch.ipynb         # Launch copy (T127): Bottleneck PyramidNet-110 (α=128), ASAM 900ep, WD 5e-4, LS 0.1
    ├── E_pyramid_t106_shakedrop_T128_launch.ipynb         # Launch copy (T128): Bottleneck PyramidNet-110 (α=128), ASAM 900ep, WD 5e-5, LS 0
    ├── E_pyramid_t106_shakedrop_T129_launch.ipynb         # Launch copy (T129): Bottleneck PyramidNet-74 (α=162), ASAM 900ep
    ├── E_pyramid_t106_shakedrop_T130_launch.ipynb         # Launch spec (T130): 45k/5k Bottleneck PyramidNet-272 (α=72)
    ├── E_pyramid_t106_shakedrop_T131_launch.ipynb         # Launch copy (T131): Full-data 50k refit, PyramidNet-272 (α=72, ReLU), ASAM 900ep
    ├── E_pyramid_t106_shakedrop_T132_launch.ipynb         # Launch spec (T132): Full-data 50k PyramidNet-272 with label smoothing 0.1
    ├── E_pyramid_t106_shakedrop_T133_launch.ipynb         # Launch copy (T133): Full-data 50k refit, PyramidNet-218 (α=84, ReLU, LS 0)
    ├── E_pyramid_t106_shakedrop_T134_launch.ipynb         # Launch copy (T134): 45k/5k held-out PyramidNet-218 (α=84, ReLU), ASAM 900ep
    ├── E_pyramid_t106_shakedrop_T135_launch.ipynb         # Launch copy (T135): Full-data 50k refit, PyramidNet-218 (α=84, ReLU, LS 0.1)
    ├── E_pyramid_t106_shakedrop_T137_launch.ipynb         # ★ Final Model Launch (T137): Full-data 50k PyramidNet-272 (α=72, SiLU), ASAM 900ep
    ├── E_pyramid_t106_shakedrop_T138_launch.ipynb         # Launch copy (T138): Full-data 50k PyramidNet-218 (α=84, SiLU), ASAM 900ep
    ├── E_T123_checkpoint_exports.ipynb                    # Saved-checkpoint fp32 submission exporter for T123
    ├── E_T128_checkpoint_exports.ipynb                    # Saved-checkpoint fp32 submission exporter for T128
    ├── Q05_shift_tta.ipynb                                # Evaluation: Multi-crop shift test-time augmentation (10-view & 18-view TTA)
    ├── Q08_shiftval.ipynb                                 # ★ Evaluation: 13-corruption × 3-severity shifted validation benchmark (Q08)
    ├── Q10_bn_adapt_balance.ipynb                         # ★ Inference: Test-time BatchNorm adaptation & Sinkhorn/exact class balancing (Q10)
    ├── Q10_T133_checkpoint_exports.ipynb                  # Q10 BN-adapted & class-balanced test exporter for full-data run T133
    └── Q10_T135_checkpoint_exports.ipynb                  # Q10 BN-adapted & class-balanced test exporter for full-data run T135
```

All notebooks in [`Archive/`](./Archive/) share a unified directory convention across environments (`RUNTIME_ENV = "auto" | "local" | "colab" | "runpod"`):
- **Local**: Reads competition data from `project1/data`, caches decoded `uint8` arrays in `project1/cache`, and writes outputs to `project1/result/{checkpoints,configs,evaluations,exports,logs,manifests,plots}/<RUN_ID>/`.
- **Google Colab**: Mounts Google Drive and uses `MyDrive/DL_Project1/{data,result}` with local SSD cache `/content/cache` (automatically backed up to `MyDrive/DL_Project1/cache.zip`).
- **RunPod**: Uses `/workspace/project1/{data,cache,result}` on the persistent volume and packs a verified `<RUN_ID>_result.zip` archive at the end of execution.

---

### 1.2 Detailed Execution Guide for Core Notebooks

#### A. Final Model Training & Primary Submission Generation: [`Archive/E_pyramid_t106_shakedrop_T137_launch.ipynb`](./Archive/E_pyramid_t106_shakedrop_T137_launch.ipynb)
This notebook trains our final 272-layer pre-activation bottleneck PyramidNet with SiLU activations (`T137`, `4,943,934` trainable parameters) on all `50,000` CIFAR-10 images (`VAL_SIZE = 0`) for `900` epochs using ASAM ($\rho = 1.0, \eta = 0.01$), and exports the baseline single-view and flip-TTA test submissions—including our **Primary Submission** (`last-ep900-raw-plain_submission.csv`, `0.918` Public / `0.899` Private).

1. **Data Preparation**:
   - Place the competition dataset (`train_labels.csv`, `test.csv`, `sample_submission.csv`, and image directories) or the pre-built `cache.zip` (SHA-256 fingerprint `42749a486a13184f0f96dab13293950f0120c357ddf1e6b1ef6a3bd3a9ce5fde`) under `data/` or `cache/` in the active runtime root (`/workspace/project1` on RunPod, `MyDrive/DL_Project1` on Colab, or `project1/` locally).
2. **Configuration (Section 1 / Cell 2)**:
   - All hyperparameters are pre-configured for `T137`:
     - Architecture: `BLOCK = "bottleneck"`, `DEPTH = 272`, `ALPHA = 72`, `ACTIVATION = "silu"`, `SHAKEDROP = 0.5`, `DROP_PATH = 0.0`.
     - Data & Augmentation: `VAL_SIZE = 0` (all 50,000 images), `AUG = "trivialaug"`, `CROP_PADDING_MODE = "constant"`, `AUTOAUG_ERASE = "cutout"`, `CUTOUT_SIZE = 16`.
     - Optimization: `EPOCHS = 900`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `NESTEROV = True`, `WEIGHT_DECAY = 5e-5`, `LABEL_SMOOTHING = 0.0`, `SAM = True`, `SAM_ADAPTIVE = True`, `SAM_RHO = 1.0`, `SAM_ETA = 0.01`, `EMA = True` (`0.999`).
     - Hardware: `AMP = "bf16"`, `COMPILE = "reduce-overhead"` (automatically gated by a numeric preflight check against eager/default mode; falls back to `"default"` if needed).
   - *Optional timing check*: Set `BENCHMARK_STEPS = 200` to benchmark steady-state seconds/epoch and projected wall-clock completion time without creating a run directory, then reset `BENCHMARK_STEPS = 0`.
   - *Resuming an interrupted run*: Set `RESUME_RUN = "<RUN_ID>"` and `RUN_NAME = None` to resume seamlessly from the latest saved `<RUN_ID>_last-ckpt.pt`.
3. **Execution (Run All Cells, Sections 1–9)**:
   - **Sections 2–4** detect the runtime environment, verify GPU/CUDA architecture support, validate the `50,000` train / `2,000` test image cache, compute full-training-set channel normalization statistics, and instantiate `PyramidNet-272` (verifying `4,943,934 <= 5,000,000` trainable parameters).
   - **Section 5–6** run the 900-epoch ASAM training loop (replaying identical ShakeDrop masks across both gradient passes and updating BN running stats once per step) and save training curves (`result/plots/<RUN_ID>/`) and epoch logs (`result/logs/<RUN_ID>/<RUN_ID>_log.csv`).
   - **Sections 7–9** run `fp32` inference on `<RUN_ID>_last-ckpt.pt` and execute **Section 9 (`Extra exports`)**, which exports all four baseline test variants (`last-ep900-raw-plain`, `last-ep900-raw-flip`, `last-ep900-ema-plain`, `last-ep900-ema-flip`) as validated 2,000-row Kaggle CSVs and probability `.npz` files under `result/evaluations/test/<RUN_ID>/`, verifies SHA-256 manifests, and packages `<RUN_ID>_result.zip`.
   - *Inference-only re-export*: If `<RUN_ID>_last-ckpt.pt` already exists, set `RESUME_RUN = "<RUN_ID>"` (or set `EXPORT_RUN_ID = "<RUN_ID>"` in Section 9) and run to regenerate all submission CSVs without retraining.

---

#### B. Corrupted Validation Benchmark (`Q08`): [`Archive/Q08_shiftval.ipynb`](./Archive/Q08_shiftval.ipynb)
This evaluation-only notebook builds our deterministic, version-locked shifted validation set (`Q08_shiftval-v1`, SHA-256 `3f12d750b28139569b8912c175dce67dfaadaa471afc838d9f4af62538b797d1`) on the `5,000` held-out validation images (`VAL_SIZE = 5000, SPLIT_SEED = 0`) and evaluates saved checkpoints from any model family (`A`, `B`, `C`, `E`, `F`, `G`) across **13 corruptions $\times$ 3 severities** grouped into four equally weighted families (`photometric`, `sharpness`, `noise`, `compression`).

1. **Configuration (Section 1 / Cell 2)**:
   - `Q08_SOURCE = "auto"`: Loads the pre-generated `Q08_shiftval-v1.npz` if present, or regenerates all 39 corruption arrays directly from the raw `uint8` validation images and verifies their exact SHA-256 fingerprint.
   - `Q08_RUNS`: List of 45k/5k run IDs to evaluate (searches `RESULT_DIR/checkpoints/` first, then any read-only `Q08_EXTRA_RESULT_ROOTS`). Note that 50k full-data refits (`VAL_SIZE = 0`) are excluded because their training set includes the 5,000 validation images.
   - `Q08_CHECKPOINTS = ("best", "last")` and `Q08_WEIGHTS = ("model", "ema")`: Evaluates both raw and EMA weights at the best and final checkpoints.
2. **Execution (Run All Cells, Sections 1–8)**:
   - **Sections 2–3** verify the competition dataset cache and confirm that the 5,000 stratified validation indices match the exact training split hash (`7957d060fcc34aefa5a43c70e31433927db55526494f67a9f7da553059be6767`).
   - **Section 4** constructs or loads the 13 corruptions (`brightness_up/down`, `contrast_up/down`, `saturate_up/down`, `gamma_bright/dark`, `gaussian_blur`, `sharpen`, `gaussian_noise`, `jpeg`, `pixelate`) across severities 1–3 and verifies their hashes.
   - **Section 5** loads isolated per-family model namespaces (`MODEL_SOURCES['A'..'G']`) copied verbatim from the six family training notebooks so every checkpoint is rebuilt with its exact architecture fingerprint.
   - **Sections 6–8** first verify that each checkpoint reproduces its stored clean validation accuracy within `Q08_REPRO_TOLERANCE_PP = 0.10` in `fp32`, then evaluate all 39 corruption sets under single-view (`plain`) and horizontal-flip (`flip`) inference, compute per-family and 4-family equal-weighted `Q08` scores, print the severity drop table (Section 7), and optionally evaluate cross-run ensemble diversity (Section 8). Per-run JSON/NPZ evaluations and aggregate CSV summaries (`*_Q08_shiftval-v1_summary.csv`) are saved to `result/evaluations/validation/<RUN_ID>/` and `result/reports/`.

---

#### C. Test-Time BatchNorm Adaptation & Class Balancing (`Q10`): [`Archive/Q10_bn_adapt_balance.ipynb`](./Archive/Q10_bn_adapt_balance.ipynb)
This evaluation and inference notebook (`TASK_ID = "T136"`) implements the two instructor-permitted post-processing methods—**BatchNorm statistics calibration/blending** and **class-balanced prediction** (Sinkhorn scaling and exact equal-count assignment)—along with optional multi-crop shift TTA (`shift5x2` 10-view and `grid9x2` 18-view from [`Q05_shift_tta.ipynb`](./Archive/Q05_shift_tta.ipynb)). It produces both the Table 6 ablation on `T134` and our **Secondary Submission** (`11-06-17-33_E_pyramid-ver-45-T137-silu-d272full_q10-bnmix025-last-ep900-ema-plain-bal_submission.csv`, `0.915` Public / `0.900` Private).

1. **Configuration (Section 1 / Cell 2)**:
   - `Q10_RUNS`: List of run IDs to evaluate or export:
     - For development/ablation (clean validation + `Q08` severity 1 + test): include 45k/5k runs such as `T134` (`11-05-00-20_E_pyramid-ver-43-T134-asam-d218`), `T128`, `T129`, `T122`, `T123`, `T106`.
     - For final full-data submission export: include `T137` (`11-06-04-18_E_pyramid-ver-45-T137-silu-d272full`) or `T138` (`11-06-04-27_E_pyramid-ver-46-T138-silu-d218full`). The notebook automatically detects `VAL_SIZE = 0` from the checkpoint and runs test-domain adaptation and export only.
   - `Q10_CHECKPOINTS = ("best", "last")` and `Q10_WEIGHTS = ("ema",)` (append `"model"` to also adapt raw weights).
   - `BN_BLEND_ALPHAS = (0.25, 0.5, 0.75)` (test and clean validation blend weights $a$ on target-domain BN stats) and `Q08_BLEND_ALPHAS = (0.5,)` (severity-1 `Q08` blend weight).
   - `BN_STATS_BATCH_SIZE = 1000` and `BN_STATS_INCLUDE_FLIP = False` (passes unflipped target-domain images in equal consecutive batches to compute target BN means and variances with `momentum = None`).
   - `Q10_VIEWS = ("plain", "flip")`: Evaluates 1-view (`plain`) and 2-view horizontal-flip (`flip`) predictions; add `"shift5x2"` (10 views) and `"grid9x2"` (18 views) for multi-crop shift TTA.
2. **Execution (Run All Cells, Sections 1–9)**:
   - **Sections 2–5** verify the competition dataset cache, rebuild the exact `PyramidNet` architecture (supporting basic/bottleneck blocks, DropPath/ShakeDrop, and ReLU/SiLU) from checkpoint metadata, and (for 45k/5k checkpoints when `Q08_EVAL = True`) regenerate and hash-verify the locked `Q08` severity-1 corruption sets.
   - **Sections 6–7** define the post-processing pipeline:
     - *BN Adaptation (`bn_target_statistics` & `bn_blend_statistics`)*: Deep-copies the model, resets all BN running stats (`momentum = None`), puts **only** BN layers in `train()` mode (keeping ShakeDrop/DropPath in `eval()`), accumulates exact target-domain means and variances over unflipped batches, and blends $(1 - a)\cdot\text{source} + a\cdot\text{target}$ for `source` ($a = 0$), `mix025` ($a = 0.25$), `mix050` ($a = 0.50$), `mix075` ($a = 0.75$), and `full` ($a = 1.0$).
     - *Class Balancing*: Applies `sinkhorn_balance` (deterministic `float64` iterative proportional fitting of the $N \times 10$ probability matrix to uniform class marginals within `BALANCE_TOL = 1e-6`, followed by argmax) and `exact_balance` (linear sum assignment enforcing exactly 200 test images per class or 500 validation images per class).
   - **Section 8** executes an automated stand-in self-test on two synthetic PyramidNets before reading any checkpoint to verify that the source model is never mutated, only BN layers enter training mode during the statistics pass, and blending math is exact.
   - **Section 9** verifies that each checkpoint's `source` variant reproduces its stored training-time validation/test probabilities (`BASELINE_PROB_ATOL = 5e-3`), evaluates all `BN variant × view × balancing` combinations, prints summary tables (including exact McNemar $p$-values), and writes all submission CSVs (`*_submission.csv` for unbalanced, `*-bal_submission.csv` for Sinkhorn, and `*-balx_submission.csv` for exact balancing), probability `.npz` archives, and `*_Q10_bn-adapt-v1_summary.csv` reports to `result/evaluations/test/<RUN_ID>/` and `result/reports/`.

---

## 2. Report Summary

### Task, Constraints, and Distribution Shift
This project builds a residual convolutional classifier for CIFAR-10 (`50,000` labeled $32 \times 32$ RGB training images across 10 classes and `2,000` unlabeled test images) from scratch under a strict limit of **5,000,000 trainable parameters**, without external data or pretrained weights. Early experiments revealed a substantial gap between clean held-out validation accuracy and Kaggle public leaderboard scores: our initial basic PyramidNet reached `97.22%` on validation at its submitted checkpoint but scored `0.889` on the public leaderboard.

To investigate this gap, we compared all `45,000` development training images against the `2,000` competition test images (Table 1). Test images exhibited higher mean RGB intensity (`+0.015` to `+0.022`), larger RGB standard deviations (`+0.014` to `+0.019`), higher within-image grayscale contrast (`0.20635` vs. `0.19635`), stronger neighbor edge intensity differences (`0.06077` vs. `0.05656`), and lower HSV saturation (`0.25975` vs. `0.27344`). We therefore paired a deterministic stratified `45,000 / 5,000` train/validation split (evaluated at the final epoch using `EMA + horizontal flip` and paired exact McNemar tests) with a locked corrupted validation benchmark (**`Q08`**) applying 13 CIFAR-10-C-style corruptions across four equally weighted families: *photometric*, *sharpness*, *noise*, and *compression*.

### Architecture Exploration and Depth Scaling
We first screened six residual network families under SGD and cosine annealing while keeping the initial feature map at $32 \times 32$ (Table 2):
- **BasicBlock ResNet** (`[3,5,3] × [64,128,256]`, `4.92M` params, 200 ep): `96.62%` val / `0.872` public.
- **Wide ResNet `WRN-22-4`** (`4.30M` params, 200 ep): `96.84%` val / `0.871` public.
- **SE-ResNet** (`[3,5,3] + SE r=16`, `4.96M` params, 200 ep): `97.08%` val (highest in screen) / `0.865` public (lowest in screen).
- **ResNeXt-29 `32×4d`** (`4.78M` params, 200 ep): `96.40%` val / `0.872` public.
- **Shake-Shake-26 `2×40d`** (`4.56M` params, 1,800 ep): `96.72%` val / `0.873` public.
- **Basic PyramidNet-110** ($\alpha = 98$, `4.96M` params, 300 ep): `96.84%` val / **`0.889` public**.

Selecting PyramidNet for its gradual channel growth and parameter-free zero-padded/average-pooled shortcuts, we transitioned to **pre-activation bottleneck blocks** ($1 \times 1, 3 \times 3, 1 \times 1$) with **ShakeDrop** regularization and traded width ($\alpha$) for depth to remain near `4.9M` parameters (Figure 1). Under a matched 900-epoch ASAM recipe on the 45k/5k split, increasing depth from **110 layers (`T128`, `98.00%`)** to **218 layers (`T134`, `98.40%`)** classified 20 more validation images correctly (`32` errors corrected vs. `12` introduced, exact McNemar $p = 0.004$) while increasing compute from `12.3` to `19.6` A100 GPU-hours. Notably, **16 of the 20 net gains** came from the two hardest animal classes: `dog` (`12` corrected vs. `2` introduced = `+10`) and `cat` (`8` corrected vs. `2` introduced = `+6`). For the full 50,000-image training set, we scaled further to a narrower **272-layer bottleneck PyramidNet ($\alpha = 72$)**, training `T131` with ReLU (`28.9` hours on an A100) and `T137` with SiLU (`12.9` hours on a B200).

On basic PyramidNet-110, `DropPath = 0.1` improved validation accuracy from `97.18%` (`T76`) to `97.60%` (`T77`), whereas wider pre-pooling heads (`-0.12 pp`), multi-stage pooling (`-0.24 pp`), IBN, and MixStyle gave no consistent gains. In a 300-epoch optimizer screen (`T106` SGD `97.48%`, `T111` SAM `97.60%`, `T112` ASAM `97.54%`), initial differences were modest, but as deeper bottleneck PyramidNets were pushed toward their performance limits, a clear ordering emerged (`ASAM > SAM > SGD`).

### Training Recipe and Prediction-Time Processing
Our final training recipe (Table 3) combines:
- **Augmentation**: $32 \times 32$ random crop (zero-padding 4) + horizontal flip $\to$ `TrivialAugmentWide` $\to$ training-split normalization $\to$ $16 \times 16$ `Cutout` (`AutoAugment` and `TrivialAugmentWide` finished within `0.3 pp`; `Mixup` and `CutMix` reduced clean accuracy; ensemble knowledge distillation lowered clean validation from `97.36%` to `97.18%` and `Q08` from `89.0` to `88.5`).
- **Optimization & Regularization**: **ASAM** ($\rho = 1.0, \eta = 0.01$) over Nesterov SGD (momentum `0.9`, batch size `128`, initial LR `0.1`, 5-epoch linear warm-up + 900-epoch cosine decay to `0`, weight decay $5 \times 10^{-5}$, zero label smoothing, `ShakeDrop` $p_L = 0.5$, and `EMA` decay `0.999` in `bf16` with `channels-last` and `torch.compile`). Replaying identical ShakeDrop masks across both ASAM passes and reducing weight decay from $5 \times 10^{-4}$ to $5 \times 10^{-5}$ while removing $0.1$ label smoothing (`T128` vs. `T127`) corrected 30 errors and introduced 19.
- **Raw vs. EMA Trajectory (Figure 2)**: On the 218-layer run, EMA led raw weights by `2.54 pp` at epoch 50 (`93.02%` vs. `90.48%`), narrowed to `0.28 pp` at epoch 450, and converged by epoch 900 (`98.38%` raw vs. `98.36%` EMA), motivating retaining both weight sets at inference.

At prediction time, we evaluated two instructor-permitted techniques (Table 6):
1. **BatchNorm Adaptation**: Recomputing BN means and variances across equal-size batches of unflipped target images (only BN layers in `train` mode) and blending with stored training statistics via weight $a$.
2. **Class Balancing**: **Sinkhorn scaling** (equalizing class probability mass before argmax) or **exact balancing** (200 images per class on test).

On `T134` (`98.40%` clean / `96.53%` mild `Q08` with training BN stats), **BN blend $a = 0.5$ + Sinkhorn** raised mild `Q08` accuracy to **`97.39%` (`+0.86 pp`)** and clean accuracy to `98.42%`, while Sinkhorn alone achieved the highest clean accuracy (`98.54%`) and full BN replacement ($a = 1.0$) degraded mild `Q08` to `95.97%` (`-0.56 pp`). Gains from `blend a = 0.5 + Sinkhorn` were concentrated in **noise (`+2.54 pp`)** and **compression (`+0.81 pp`)** over photometric (`+0.055 pp`) and sharpness (`+0.04 pp`). Across five other held-out runs (`T106`, `T122`, `T123`, `T128`, `T129`), the same combination gained **`+0.97 pp`** on average on mild `Q08`, whereas full BN replacement lost **`-1.18 pp`**.

### Final Architecture, Leaderboard Results, and Conclusions
The final **`PyramidNet-272` ($\alpha = 72$, SiLU, ShakeDrop — Table 4)** has a $3 \times 3$ conv stem (`16` channels) followed by $N = 3$ stages of 30 pre-activation bottleneck blocks each (`BN → 1×1 → BN → SiLU → 3×3 → BN → SiLU → 1×1 → BN → ShakeDrop` + zero-padded shortcut, with $2 \times 2$ avg-pool downsampling at Stages 2 and 3), expanding bottleneck inner widths from `17` to `88` channels and stage outputs across `68 → 160` ($32 \times 32$), `164 → 256` ($16 \times 16$), and `260 → 352` ($8 \times 8$), followed by `BN → SiLU → 8×8 global avg-pool → Linear(352 → 10)` for **`4,943,934` trainable parameters**.

| Selected Submission (`T137`, Epoch 900, 1 View) | Weights | Prediction-Time Processing | Public | Private | Combined Test (`2,000` Images) |
| :--- | :--- | :--- | ---: | ---: | ---: |
| **Primary** | `raw` | None (single forward pass) | **0.918** | **0.899** | **1,817 / 2,000** (`90.85%`) |
| **Secondary** | `EMA` | BN blend $a = 0.25$ + Sinkhorn balancing | **0.915** | **0.900** | **1,815 / 2,000** (`90.75%`) |

Our two selected submissions (Table 5, Figure 3) disagreed on **54 of the 2,000 test images**: `raw` weights led by 3 images on the public split (`918` vs. `915`), while adapted `EMA` weights led by 1 image on the private split (`900` vs. `899`). Across the full 2,000-image test benchmark (public + private combined), the unadapted `raw` weights of `T137` correctly identified **`1,817 / 2,000` images**, outperforming the BN-blended and class-balanced EMA submission (`1,815 / 2,000`). Under matched single-view adapted EMA inference, 272-layer SiLU (`T137`, `0.915`) edged out 218-layer SiLU (`T138`, `0.913`) by 2 public images, and SiLU achieved slightly higher raw and variant-averaged `Q08` scores than ReLU (`T131`, `0.916` with exact balancing). Because single-seed comparisons and 1,000-image leaderboard splits carry $0.1\text{ pp}$ per-image granularity, future work would replicate the 110- vs. 218-layer ASAM comparison across multiple seeds, evaluate 218 vs. 272 layers directly on the held-out split, and lock in post-processing choices on development data before full-data refitting.
