import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.svm import SVC
import numpy as np

# Load ECG-only data
df = pd.read_csv('data/ECG_AP1.csv')
X = df.drop(columns=['label'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

selector = SelectKBest(score_func=f_classif, k=6)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Quantum kernel setup
feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='full')
backend = Aer.get_backend("statevector_simulator")
qi = QuantumInstance(backend=backend)
qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=qi)

# Kernel matrices
kernel_train = qkernel.evaluate(x_vec=X_train)
kernel_test = qkernel.evaluate(x_vec=X_test, y_vec=X_train)

# Train QSVM
clf = SVC(kernel='precomputed', C=1.0)
clf.fit(kernel_train, y_train)
y_pred = clf.predict(kernel_test)

# Evaluation
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("✅ Precision:", round(precision_score(y_test, y_pred), 4))
print("✅ Recall:", round(recall_score(y_test, y_pred), 4))
print("✅ F1 Score:", round(f1_score(y_test, y_pred), 4))