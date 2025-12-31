# 20 Newsgroups Classification

This project focuses on text classification using the **20 Newsgroups dataset**. It explores a BiLSTM-based architecture with multiple attention mechanisms and provides both quantitative evaluation and detailed analysis of model behavior. The repository is structured to clearly separate **baseline models** and the
**proposed attention-based model**.

---

## Repository Structure

### Main Training & Evaluation (Proposed Model)

**train.py**  
  Trains the classification model and automatically saves the best checkpoint
  based on validation loss.

- **`test.py`**  
  Evaluates the trained model and produces performance metrics along with
  visualizations such as confusion matrices and plots.

- **`model.py`**  
  Contains the core model architecture:
  - BiLSTM encoder  
  - Word-level attention  
  - Top-K attention selection  
  - Cross-attention mechanism  
  - Final classification layer  

- **`data_pipeline.py`**  
  Handles all data-related processing, including dataset loading, sentence
  splitting, tokenization, vocabulary construction, and loading **GloVe**
  embeddings.

- **`config.py`**  
  Central place for defining hyperparameters, file paths, and experiment settings.

- **`best_model_by_val_loss.pt`**  
  Saved model checkpoint corresponding to the lowest validation loss.

---

### Baseline Models

The **`Base/`** directory contains baseline implementations used for comparison
with the proposed attention-based model. These baselines provide reference
performance and help quantify the improvements achieved by more advanced
architectural choices.

Typical baseline components include:
- Simpler neural architectures
- Models without attention mechanisms
- Standard training and evaluation pipelines

---

### Analysis

All analysis scripts generate their outputs inside the **`attention_reports/`**
directory.

- **`analysis-1.py`**  
  Visualizes attention behavior, including Top-K word attention, cross-attention
  heatmaps, and sentence-level importance.

- **`analysis-2.py`**  
  Performs error analysis such as per-class performance metrics, confusion
  matrices, and inspection of misclassified examples.

- **`analysis-3.py`**  
  Studies attention dynamics, including Top-K sensitivity and entropy-based
  statistical analysis.

- **`analysis-4.py`**  
  Investigates failure modes, focusing on model biases and compounding error
  patterns.

---

### Outputs

- **`attention_reports/`**  
  Contains all generated artifacts:
  - JSON summaries  
  - CSV files  
  - Visualization plots  

- **Evaluation files**
  - `test_metrics.csv`
  - `confusion_matrix_test.csv`


---

## How to Run

1. Train the base model:
   ```bash
   cd Base
   python train.py
   ```

2. Test the base model:
   ```bash
   python test.py
   ```

3. Train the proposed model:
   ```bash
   cd ../Proposed
   python train.py
   ```

4. Test the proposed model:
   ```bash
   python test.py
   ```

---

## Analysis

Run analysis scripts from the `Proposed` directory after model evaluation.

1. Attention visualization:
   ```bash
   python analysis-1.py
   ```

2. Error analysis:
   ```bash
   python analysis-2.py
   ```

3. Attention sensitivity analysis:
   ```bash
   python analysis-3.py
   ```

4. Failure-mode analysis:
   ```bash
   python analysis-4.py
   ```

All analysis outputs are saved in the `attention_reports/` directory.

