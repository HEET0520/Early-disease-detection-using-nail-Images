import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier

# --- Configuration ---
# NOTE: Updated file name to match the file you provided
FEATURES_FILE = "data/features_weighted_avg.csv" 
MODEL_DIR = "models" 

# Output file paths
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
PCA_PATH = os.path.join(MODEL_DIR, "pca_model.joblib")
BINARY_SVM_PATH = os.path.join(MODEL_DIR, "binary_svm_model.joblib")
MULTI_SVM_PATH = os.path.join(MODEL_DIR, "multiclass_svm_model.joblib")

# Mapping used for printing/reporting only (from your notebook logic)
# This mapping assumes 'Healthy Nail' is label 3.
INT_TO_LABEL_NAME_MAPPING = {
    0: "Acral Lentiginous Melanoma",
    1: "Bluish Nail",
    2: "Clubbing",
    3: "Healthy Nail",
    4: "Koilonychia",
    5: "Median nail",
    6: "Nail Pitting",
    7: "Onychogryphosis",
    8: "Onychomycosis",
    9: "Yellow Nails"
}
HEALTHY_LABEL_INT = 3

def train():
    print(f"Loading features from {FEATURES_FILE}...")
    try:
        df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"Error: Features file not found at '{FEATURES_FILE}'")
        print("Please ensure the file is in the correct 'data/' folder.")
        return

    # --- SIMPLIFIED DATA LOADING ---
    # Your file already has the integer label in a column named 'Label'.
    
    # 1. Separate features (X) and labels (y)
    X = df.drop(columns=['Label']) # All Feature_X columns
    y = df['Label']
    
    X = X.fillna(0) # Fill NaNs (as done in your notebook)
    
    print(f"Total samples after loading: {len(X)}")

    # --- 1. Standard Scaler & PCA (from cell 16 & 22) ---
    print("\nTraining StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    print("Training PCA...")
    # Using n_components=149 (as found in your notebook output)
    pca = PCA(n_components=149) 
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, PCA_PATH)
    print(f"PCA model saved to {PCA_PATH}")
    print(f"Original features: {X_scaled.shape[1]}, PCA features: {X_pca.shape[1]}")

    # --- 2. Binary Classification (Healthy vs. Diseased) (from cell 22) ---
    print("\nStarting Binary Classification (Healthy vs. Diseased)...")
    
    # Healthy = 0, Diseased = 1
    y_binary = (y != HEALTHY_LABEL_INT).astype(int) 
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_pca, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    print("Tuning Binary SVM with GridSearchCV...")
    binary_svm_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01, 0.001],
        'kernel': ['rbf', 'poly']
    }
    binary_grid = GridSearchCV(
        SVC(probability=True, class_weight='balanced'), 
        binary_svm_grid, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy'
    )
    binary_grid.fit(X_train_bin, y_train_bin)

    print(f"✅ Best Binary SVM Params: {binary_grid.best_params_}")
    best_binary_svm = binary_grid.best_estimator_
    
    y_pred_binary = best_binary_svm.predict(X_test_bin)
    print("\n📊 Binary Classification Report (Healthy=0, Diseased=1):")
    print(classification_report(y_test_bin, y_pred_binary, target_names=["Healthy", "Diseased"]))
    
    joblib.dump(best_binary_svm, BINARY_SVM_PATH)
    print(f"Binary SVM model saved to {BINARY_SVM_PATH}")

    # --- 3. Multiclass Classification (Diseased Only) (from cell 22) ---
    print("\nStarting Multiclass Classification (Diseased Types)...")
    
    X_diseased = X_pca[y_binary == 1]
    y_diseased = y[y_binary == 1] # Original disease labels
    
    # Filter map to only include the disease names
    disease_labels = sorted(y_diseased.unique())
    disease_label_names = [INT_TO_LABEL_NAME_MAPPING[idx] for idx in disease_labels]
    
    print(f"Training on {len(disease_labels)} disease classes: {disease_labels}")

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_diseased, y_diseased, test_size=0.2, random_state=42, stratify=y_diseased
    )

    print("Tuning Multiclass One-vs-Rest SVM...")
    multi_svm_grid = {
        'estimator__C': [1, 10, 100],
        'estimator__gamma': ['scale', 0.01, 0.001],
        'estimator__kernel': ['rbf', 'poly']
    }
    
    ovr_svm = OneVsRestClassifier(SVC(probability=True, class_weight='balanced'))
    
    multi_grid = GridSearchCV(
        ovr_svm, 
        multi_svm_grid, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy'
    )
    multi_grid.fit(X_train_d, y_train_d)

    print(f"✅ Best Multiclass SVM Params: {multi_grid.best_params_}")
    best_multi_svm = multi_grid.best_estimator_

    y_pred_multi = best_multi_svm.predict(X_test_d)
    print("\n📊 Multiclass Classification Report (Disease Types):")
    print(classification_report(y_test_d, y_pred_multi, labels=disease_labels, target_names=disease_label_names))

    joblib.dump(best_multi_svm, MULTI_SVM_PATH)
    print(f"Multiclass SVM model saved to {MULTI_SVM_PATH}")
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    train()