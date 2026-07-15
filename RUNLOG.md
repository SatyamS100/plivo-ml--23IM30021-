# RUNLOG

## Run 1: Silence-Only Baseline (Given)
- **Honest OOF Validation Scores**:
  - English OOF Delay: 1600 ms (at 0.0% interrupted turns, threshold=1.0, delay=1600 ms)
  - Hindi OOF Delay: 1600 ms
  - Pooled AUC: 0.514
- **In-Sample Train-Fit Scores**: Same as above (no fit performed).
- **Changes**: None (pure silence-based baseline).
- **Rationale**: Acts as the baseline starting point. The agent waits out the full 1.6s timeout at the end of turns because it has no predictive model.

---

## Run 2: Starter Weak Model (Given)
- **Honest OOF Validation Scores**:
  - English OOF Delay: 1305 ms (at 5.0% interrupted turns, threshold=0.55, delay=600 ms)
  - Hindi OOF Delay: 850 ms (at 5.0% interrupted turns, threshold=0.05, delay=850 ms)
  - Overall OOF Delay (Pooled): 1194 ms
  - Pooled OOF AUC: 0.5825 (fold std: 0.0633)
- **In-Sample Train-Fit Scores**:
  - English Train-Fit Delay: 1190 ms (AUC: 0.596)
- **Changes**: Trained Logistic Regression on 3 basic features (local energy, final pitch, context duration).
- **Rationale**: A weak starter script demonstrating how features are computed from the last 1.5s window. Note that the 1190ms figure reported previously was the starter model evaluated on its own training set. Under honest 5-fold CV, the starter model actually scores 1305 ms on English.

---

## Run 3: Causal Feature Engineering v1 (35 Features) + Model Comparison
- **Cross-Validation Setup**: 5-fold GroupKFold. Dataset size: 200 turns (100 English, 100 Hindi) with 496 pauses. Fold size: exactly 40 validation turns and 99-100 validation pauses per fold. Each fold's validation predictions come strictly from a model retrained fresh on the other 4 folds.
- **Honest OOF Validation Scores**:
  - **RandomForest_d8**: Overall OOF Delay = 1105 ms, English OOF Delay = 1300 ms, Hindi OOF Delay = 850 ms, OOF AUC = 0.6628 (fold std: 0.0593, fold AUCs: `[0.6538, 0.6881, 0.7102, 0.7169, 0.5547]`)
  - **SVC_rbf**: Overall OOF Delay = 1134 ms, English OOF Delay = 1248 ms, Hindi OOF Delay = 850 ms, OOF AUC = 0.6860 (fold std: 0.0369, fold AUCs: `[0.6713, 0.6508, 0.7301, 0.7354, 0.6542]`)
  - **LogisticRegression (Scaled)**: Overall OOF Delay = 1154 ms, OOF AUC = 0.6593 (fold std: 0.0415, fold AUCs: `[0.6162, 0.6928, 0.6695, 0.7258, 0.6233]`)
  - **LightGBM**: Overall OOF Delay = 1184 ms, OOF AUC = 0.6585 (fold std: 0.0422, fold AUCs: `[0.6517, 0.6462, 0.7008, 0.7157, 0.5975]`)
- **Changes**: Extracted ZCR, spectral centroid, spectral roll-off, and high-to-low frequency ratios. Normalized energy and pitch relative to speaker/turn-level baselines.
- **Rationale**: Normalization addresses speaker and volume variability causally. Standard scaling resolved Logistic Regression convergence issues.

---

## Run 4: Causal Feature Engineering v2 (44 Features) + Model Selection & Verification
- **Cross-Validation Setup**: 5-fold Stratified Group K-Fold. Fold size: exactly 40 validation turns and 98-101 validation pauses per fold. Models retrained strictly from scratch on training partitions to predict on validation partitions.
- **Features added**: Pitch delta (slope of F0 over final 200ms of causal audio) and breath detector (ZCR spike and energy drop over final 100ms of causal audio). No language flag is added.

### Side-by-Side Mode Comparison (SVC_rbf)
- **Mode A: Full 44 Features**
  - **Overall OOF AUC**: **0.6749** (fold std: **0.0376**, fold AUCs: `[0.7045, 0.6552, 0.7250, 0.6915, 0.6193]`)
  - **English OOF Delay**: **1163 ms** (at threshold=0.45) — **Beats starter OOF English delay of 1305 ms by 142 ms!**
  - **Hindi OOF Delay**: **850 ms** (at threshold=0.05)
  - **Average Language Delay**: **1006.5 ms**
- **Mode B: Pruned (Top 15 Features Selected inside Fold Splits)**
  - **Overall OOF AUC**: **0.6699** (fold std: **0.0147**, fold AUCs: `[0.6537, 0.6599, 0.6674, 0.6962, 0.6737]`)
  - **English OOF Delay**: **1230 ms** (at threshold=0.50)
  - **Hindi OOF Delay**: **850 ms** (at threshold=0.05)
  - **Average Language Delay**: **1040.0 ms**

### In-Sample Sanity Check / Train-Fit Scores (SVC_rbf_FullFeatures)
*These scores are evaluated on the exact same folders the final model was trained on, serving strictly as an end-to-end pipeline sanity check rather than a generalization performance estimate:*
- **English Train-Fit Delay**: **770 ms** (AUC = 0.884)
- **Hindi Train-Fit Delay**: **731 ms** (AUC = 0.884)

- **Changes**: Added advanced micro-acoustic features (ZCR spike, local energy drop, breath detector, final-200ms pitch slope) and changed CV to `StratifiedGroupKFold`. Feature selection (Lasso/RF) was implemented inside the folds to avoid leakage.
- **Rationale**: The side-by-side comparison shows that the model performs significantly better using the full 44-feature set (1163 ms English Delay) compared to pruning down to the top 15 features (1230 ms). The full-feature RBF SVC model was trained on all combined data and saved to `best_model.pkl`.
