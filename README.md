# 🌸 Project 2: Data Classification Using AI
**DecodeLabs — Artificial Intelligence Internship (Batch 2026)**

## 📌 Overview
This project builds a basic **classification model** using the **Iris dataset** (150 samples, 3 species, 4 features), following the **INPUT → PROCESS → OUTPUT** pipeline taught in the training kit.

**Algorithm used:** K-Nearest Neighbors (KNN)

**Pipeline steps covered:**
1. Load and understand the dataset
2. Feature scaling (StandardScaler)
3. Train-test split (80% / 20%, shuffled, stratified)
4. Choose the best K (elbow method)
5. Train the model (Instantiate → Fit → Predict)
6. Evaluate with a Confusion Matrix
7. Evaluate with Accuracy, Precision, Recall, and F1 Score
8. Predict on a brand-new, unseen sample

## 📁 Files in This Project

| File | Description |
|---|---|
| `Project2_Iris_Classification.ipynb` | Full project in **one file** — explanations + code + already-run outputs (tables & graphs). **Recommended for submission.** |
| `Project2_Iris_Classification.html` | Same notebook exported as a webpage — open in any browser, no installation needed. |
| `Project2_Iris_Classification.pdf` | Same notebook exported as a PDF — ready to print or upload. |
| `iris_classification.py` | The same pipeline as a plain Python script (no notebook). |
| `confusion_matrix.png` | Confusion matrix chart (standalone image). |
| `k_selection_elbow.png` | Elbow-method chart used to pick the best K value (standalone image). |
| `README.md` | This file. |

## ▶️ How to Run

### Option A — View only (no setup needed)
Just open **`Project2_Iris_Classification.html`** or **`Project2_Iris_Classification.pdf`** in a browser/PDF reader. Everything is already executed and visible — nothing to install.

### Option B — Run the Notebook (`.ipynb`) yourself
1. Install Python 3.9+ if you don't have it: https://www.python.org/downloads/
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn notebook
   ```
3. Open a terminal in the folder containing `Project2_Iris_Classification.ipynb` and run:
   ```bash
   jupyter notebook Project2_Iris_Classification.ipynb
   ```
   (This opens Jupyter in your browser.)
   
   **OR** upload the `.ipynb` file to [Google Colab](https://colab.research.google.com/) — no installation needed there.
4. Once it's open, click **Run → Run All Cells** (or press `Shift + Enter` on each cell one by one) to execute the whole pipeline from top to bottom.

### Option C — Run the plain Python script (`iris_classification.py`)
1. Install Python 3.9+ (see link above).
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
3. Open a terminal in the folder containing `iris_classification.py` and run:
   ```bash
   python iris_classification.py
   ```
4. Results will print in the terminal, and two chart images (`k_selection_elbow.png`, `confusion_matrix.png`) will be saved in the same folder.

## 📊 Result Summary
- **Best K (via elbow method):** 1
- **Accuracy:** 96.67%
- **F1 Score (macro):** 0.9666
- Model correctly classified 29 out of 30 test samples.

## 🛠️ Requirements
- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn
- jupyter / notebook (only needed to *run* the `.ipynb`, not to view the `.html`/`.pdf`)

## 👤 Submission Note
Remember to replace `(Your Name Here)` at the top of the notebook/PDF with your actual name before submitting.

---
*DecodeLabs | www.decodelabs.tech | decodelabs.tech@gmail.com*
