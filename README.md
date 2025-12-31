# Pull Request Merge Predictor

# Quick Start

```bash
pip install -r requirements.txt

# Run with curated subset (default)
python -m src.main --use-curated

# Or run with 'all_' datasets (no commit_details/review_comments support)
python -m src.main --no-use-curated
```

# Project Layout

```
src/
  config.py              # global flags, paths, seed
  data_io.py             # loading + basic normalization
  features.py            # feature engineering & modeling table build
  model.py               # preprocessing, training, coef extraction
  evaluation.py          # metrics, plots, artifact saving
  main.py                # end-to-end training
results/
  figs/                  # saved figures (created at runtime)
  tables/                # saved CSVs (created at runtime)
```



## Overview

This project addresses the task of predicting whether a GitHub Pull Request (PR) will be merged or closed without merging.

The pipeline covers:

• Dataset loading (curated PR data from HuggingFace).

• Feature engineering (text, time, patch statistics, repository metadata).

• Model training with **Logistic Regression.**

• Evaluation and visualization of results.



## Dataset

We use the AIDev dataset hosted on HuggingFace.
Two variants are supported:

• **Curated subset:** includes commit details and review comments.

• **Full dataset:** larger but without commit/review features.

Data paths are configurable in src/config.py


## Features

### Main groups of features:

• **PR text & time:** title length, body length, created hour/weekday.

• **Patch stats:** lines added, lines deleted, files changed, presence of test files.

• **Metadata:** task type, agent.

• **Repository signals:** star count.

## Methodology

**Train/test split:** 80/20 with stratification.

**Preprocessing:** numeric passthrough, categorical encoded with OneHotEncoder.

**Classifier:** Logistic Regression with class balancing.

**Evaluation metrics:** Accuracy, Precision, Recall, F1, ROC-AUC.

### Proposed Method

<img width="999" height="572" alt="Untitled Diagram drawio" src="https://github.com/user-attachments/assets/04b7b773-95ce-4029-81f5-6aaa9f244a9e" />


## Results

**Class balance (test split):**

• Merged: ~4,800

• Not merged: ~1,400

**Performance:**

• ROC-AUC: ~0.69

• Accuracy: ~71%

• Precision (merged): 0.86

• Recall (merged): 0.75

**Key findings:**

• PRs containing tests are more likely to be merged.

• Agents such as OpenAI Codex correlate positively with merges.

• Repository popularity (stars) has weak but noticeable influence.


## Reproducibility

• Fixed seed (42) ensures repeatability.

• All outputs are regenerated with one command.

• Results and features are stored in CSVs for transparency.

## Outputs

### Tables - results/tables

• class_balance_full.csv – overall label distribution

• class_balance_train.csv – label distribution in training split

• class_balance_test.csv – label distribution in test split

• test_predictions_logit.csv – model predictions (y_true, y_prob, y_pred)

• logit_feature_importance.csv – feature importance with coefficients, absolute values, and odds ratios

• sample_model_table.csv – a 1,000-row sample of the modeling table used for training

### Figures - results/figs

• class_balance_test.png – visualization of class distribution in test split
<img width="630" height="470" alt="class_balance_test" src="https://github.com/user-attachments/assets/45cda87d-d57c-4a02-be9a-cf49160afa98" />


• roc_curve_logit.png – ROC curve of logistic regression on test data
<img width="462" height="470" alt="roc_curve_logit" src="https://github.com/user-attachments/assets/893fa349-380b-4fdd-b9d3-556b4e80b91f" />


• logit_top_coefs.png – top-N feature weights by absolute coefficient value
<img width="790" height="790" alt="logit_top_coefs" src="https://github.com/user-attachments/assets/c1b31b19-6ac5-43d1-b008-37927a9589a8" />


##                                         ------------- End -------------
