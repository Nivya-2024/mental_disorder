import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.combine import SMOTEENN

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler

def load_and_preprocess():
    df = pd.read_csv("data/depression_anxiety.csv")
    print("Available columns:", df.columns.tolist())
    df = df.dropna()

    X = df[['phq_score', 'gad_score', 'epworth_score', 'sleepiness', 'age', 'gender', 'bmi']]
    y = df['depression_severity']
    return X, y

def train_qsvm_rf():
    X, y = load_and_preprocess()

    # Feature engineering
    X['mental_risk_score'] = X['phq_score'] + X['gad_score'] + X['epworth_score']
    X['sleep_adjusted_score'] = X['sleepiness'] / (X['epworth_score'] + 1)
    X.columns = X.columns.astype(str)

    # Encode gender
    X['gender'] = X['gender'].map({'male': 0, 'female': 1})

    # Map severity labels to binary
    severity_map = {
        "None": 0,
        "Mild": 0,
        "Moderate": 1,
        "Moderately severe": 1,
        "Severe": 1
    }
    y_binary = y.map(severity_map)
    mask = y_binary.notna()
    X = X[mask]
    y_binary = y_binary[mask]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, y_binary)
    importances = rf.feature_importances_
    top_indices = importances.argsort()[-8:]
    X_selected = X_scaled[:, top_indices]

    # Dimensionality reduction
    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X_selected)

    # Balance data
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_pca, y_binary)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Quantum kernel setup
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    # Train QSVM
    qsvc = QSVC(quantum_kernel=kernel, C=10)
    qsvc.fit(X_train, y_train)

    # Evaluate
    y_pred = qsvc.predict(X_test)
    print("📊 Classification Report (QSVM + RF):")
    print(classification_report(y_test, y_pred))
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("📈 Additional Metrics:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))

if __name__ == "__main__":
    train_qsvm_rf()