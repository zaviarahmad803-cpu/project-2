"""
=====================================================================
 DecodeLabs - Artificial Intelligence Internship
 Project 2: Data Classification Using AI
 Goal: Build a basic classification model using a small dataset (Iris)
 Algorithm Used: K-Nearest Neighbors (KNN)
 Framework Followed: INPUT -> PROCESS -> OUTPUT (as per PPT)
=====================================================================
"""

# ---------------------------------------------------------------
# STEP 0: IMPORT LIBRARIES
# ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

print("=" * 70)
print("PROJECT 2: DATA CLASSIFICATION USING AI  |  DecodeLabs Internship")
print("=" * 70)

# ---------------------------------------------------------------
# STEP 1 (INPUT): LOAD AND UNDERSTAND THE DATASET
# "Raw Material: The Iris Benchmark" -> 150 samples, 3 classes, 4 features
# ---------------------------------------------------------------
iris = load_iris()
X = iris.data                     # features: sepal length/width, petal length/width
y = iris.target                   # labels: 0=Setosa, 1=Versicolor, 2=Virginica
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print("\n--- STEP 1: DATASET OVERVIEW ---")
print(f"Total samples : {df.shape[0]}")
print(f"Total features: {len(feature_names)}")
print(f"Classes       : {list(target_names)}")
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution (balanced dataset check):\n", df["species"].value_counts())
print("\nBasic statistics:\n", df.describe())

# ---------------------------------------------------------------
# STEP 2 (PROCESS - Gatekeeper Rule): FEATURE SCALING
# StandardScaler -> Mean = 0, Variance = 1 (removes bias from raw scale)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- STEP 2: FEATURE SCALING APPLIED (StandardScaler) ---")
print("Before scaling (first row):", X[0])
print("After  scaling (first row):", np.round(X_scaled[0], 3))

# ---------------------------------------------------------------
# STEP 3 (PROCESS - Structural Integrity): TRAIN-TEST SPLIT
# Shuffle first to remove order bias, then split 80% train / 20% test
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,
    random_state=42,   # reproducibility
    shuffle=True,       # randomize before splitting (removes order bias)
    stratify=y           # keeps class balance equal in train & test
)

print("\n--- STEP 3: TRAIN-TEST SPLIT (80/20) ---")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples : {X_test.shape[0]}")

# ---------------------------------------------------------------
# STEP 4 (PROCESS - Tuning the Engine): CHOOSE OPTIMAL "K"
# Test multiple K values and find the "elbow" (lowest error rate)
# ---------------------------------------------------------------
error_rates = []
k_values = range(1, 21)

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    pred_temp = knn_temp.predict(X_test)
    error_rates.append(np.mean(pred_temp != y_test))

best_k = k_values[np.argmin(error_rates)]

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), error_rates, marker="o", linestyle="--", color="steelblue")
plt.axvline(best_k, color="orangered", linestyle=":", label=f"Optimal K = {best_k}")
plt.title("Tuning the Engine: Choosing K (Error Rate vs K Value)")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/k_selection_elbow.png", dpi=150)
plt.close()

print(f"\n--- STEP 4: OPTIMAL K FOUND VIA ELBOW METHOD ---")
print(f"Best K value = {best_k} (lowest error rate = {min(error_rates):.4f})")

# ---------------------------------------------------------------
# STEP 5 (PROCESS - The Workflow: Scikit-Learn): TRAIN THE MODEL
# INSTANTIATE -> FIT -> PREDICT
# ---------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=best_k)   # INSTANTIATE
model.fit(X_train, y_train)                        # FIT (memorize the map)
predictions = model.predict(X_test)                # PREDICT (apply logic)

print(f"\n--- STEP 5: MODEL TRAINED (KNeighborsClassifier, n_neighbors={best_k}) ---")

# ---------------------------------------------------------------
# STEP 6 (OUTPUT - Diagnostic Tool): CONFUSION MATRIX
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Iris Classification (KNN)")
plt.tight_layout()
plt.savefig("/home/claude/confusion_matrix.png", dpi=150)
plt.close()

print("\n--- STEP 6: CONFUSION MATRIX ---")
print(cm)

# ---------------------------------------------------------------
# STEP 7 (OUTPUT - Strategic Trade-offs): FINAL METRICS
# Accuracy alone can be a "mirage" -> also check Precision, Recall, F1
# ---------------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="macro")
recall = recall_score(y_test, predictions, average="macro")
f1 = f1_score(y_test, predictions, average="macro")

print("\n--- STEP 7: MODEL EVALUATION (OUTPUT) ---")
print(f"Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nFull Classification Report:\n")
print(classification_report(y_test, predictions, target_names=target_names))

# ---------------------------------------------------------------
# STEP 8: TEST WITH A NEW / UNSEEN SAMPLE (real-world simulation)
# ---------------------------------------------------------------
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # looks like a Setosa
new_sample_scaled = scaler.transform(new_sample)
new_prediction = model.predict(new_sample_scaled)

print("\n--- STEP 8: PREDICTION ON A BRAND-NEW SAMPLE ---")
print(f"Input features : {new_sample[0]}")
print(f"Predicted class: {target_names[new_prediction[0]]}")

print("\n" + "=" * 70)
print("PROJECT 2 PIPELINE COMPLETE. Files saved:")
print(" - k_selection_elbow.png")
print(" - confusion_matrix.png")
print("=" * 70)
