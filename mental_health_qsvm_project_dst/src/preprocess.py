import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_fused_data(fused_path="data/fused_eeg_ecg.csv"):
    # Load fused data
    df = pd.read_csv(fused_path)

    # Encode gender (Female=0, Male=1)
    df["Gender_x"] = df["Gender_x"].map({"Female": 0, "Male": 1})

    # Create binary stress label (AC1 = 1, EO = 0)
    df["Stress_Label"] = df["Segment"].map({"AC1": 1, "EO": 0})
    y = df["Stress_Label"]

    # Drop non-numeric columns before scaling
    drop_cols = ["Subject", "Gender_y", "Segment", "Segment_x", "Segment_y", "Stress_Label"]
    X = df.drop(columns=drop_cols)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Balance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_fused_data()
    print("✅ Preprocessing complete")
    print("🔢 Train shape:", X_train.shape)
    print("🔢 Test shape:", X_test.shape)
    print("🎯 Class distribution in train:", pd.Series(y_train).value_counts().to_dict())