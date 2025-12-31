# 20 Newsgroups Classification

This project focuses on text classification using the **20 Newsgroups dataset**. It explores a BiLSTM-based architecture with multiple attention mechanisms and provides both quantitative evaluation and detailed analysis of model behavior. The repository is structured to clearly separate **baseline models** and the
**proposed attention-based model**.

---


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

### Outputs

- **`attention_reports/`**  
  Contains all generated artifacts:
  - JSON summaries  
  - CSV files  
  - Visualization plots  

- **Evaluation files**
  - `test_metrics.csv`
  - `confusion_matrix_test.csv`

