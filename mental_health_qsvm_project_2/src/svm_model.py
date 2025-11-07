import pandas as pd
import numpy as np
from sklearn.svm import SVC
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

from preprocessing import load_and_preprocess  # Your existing function

def train_svm():
    # Load preprocessed data
    X_raw, y_raw = load_and_preprocess()

    # Assign column names
    column_names = ['phq_score', 'gad_score', 'epworth_score', 'sleepiness', 'age', 'gender', 'bmi']
    X = pd.DataFrame(X_raw, columns=column_names)
    y = pd.Series(y_raw)

    # Feature engineering
    X['mental_risk_score'] = X['phq_score'] + X['gad_score'] + X['epworth_score']
    X['sleep_adjusted_score'] = X['sleepiness'] / (X['epworth_score'] + 1)
    X.columns = X.columns.astype(str)

    # Binary classification target
    y_binary = y.apply(lambda val: 0 if val <= 2 else 1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection using Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_scaled, y_binary)
    importances = rf.feature_importances_
    top_indices = importances.argsort()[-6:]
    X_selected = X_scaled[:, top_indices]

    # Dimensionality reduction
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_selected)

    # Balance data
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_pca, y_binary)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Classical SVM with RBF kernel
    svm = SVC(kernel='rbf', C=5, gamma='scale')  # C=5 to match QSVM
    svm.fit(X_train, y_train)

    # Evaluate
    y_pred = svm.predict(X_test)
    print("📊 Classification Report:")
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
    train_svm()