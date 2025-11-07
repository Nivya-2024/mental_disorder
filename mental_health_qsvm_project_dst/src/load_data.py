import pandas as pd
import os

def load_eeg_features(data_dir="data/raw"):
    df = pd.read_csv(os.path.join(data_dir, "Ratio of Alpha _ Beta Power.csv"), skiprows=1)
    df = df.drop(index=0).reset_index(drop=True)
    df.rename(columns={"Unnamed: 0": "Subject", "Unnamed: 1": "Gender"}, inplace=True)
    df = df[["Subject", "Gender", "EO", "AC1", "AC2"]]
    df_melted = df.melt(id_vars=["Subject", "Gender"], value_vars=["EO", "AC1", "AC2"],
                        var_name="Segment", value_name="EEG_Feature")
    df_melted.dropna(inplace=True)
    df_melted["Subject"] = df_melted["Subject"].astype(str).str.extract(r"(\d+)")[0]
    return df_melted

def load_clean_ecg(data_dir="data/raw"):
    ecg = pd.read_csv(os.path.join(data_dir, "ECG_AC1.csv"))
    ecg.columns = ecg.columns.str.strip()
    if "Subject NO." in ecg.columns:
        ecg.rename(columns={"Subject NO.": "Subject"}, inplace=True)
    ecg["Subject"] = ecg["Subject"].astype(str).str.extract(r"(\d+)")[0]
    ecg["Segment"] = "AC1"
    ecg = ecg.loc[:, ~ecg.columns.str.contains("Unnamed")]
    ecg.dropna(inplace=True)
    return ecg

if __name__ == "__main__":
    print("📁 Files in raw folder:", os.listdir("data/raw"))

    eeg = load_eeg_features()
    ecg = load_clean_ecg()

    # AC1 = stress (label 1)
    eeg_ac1 = eeg[eeg["Segment"] == "AC1"].copy()
    eeg_ac1["Stress_Label"] = 1
    fused_ac1 = pd.merge(eeg_ac1, ecg, on=["Subject"], how="inner")
    fused_ac1["Segment"] = "AC1"

    # EO = non-stress (label 0)
    eeg_eo = eeg[eeg["Segment"] == "EO"].copy()
    eeg_eo["Stress_Label"] = 0
    fused_eo = pd.merge(eeg_eo, ecg, on=["Subject"], how="inner")
    fused_eo["Segment"] = "EO"

    # Combine both
    fused_all = pd.concat([fused_ac1, fused_eo], ignore_index=True)

    print("🧬 Fused shape:", fused_all.shape)
    print("🧬 Columns:", fused_all.columns.tolist())
    print("🔍 Preview:")
    print(fused_all.head())

    # Save for preprocessing
    os.makedirs("data", exist_ok=True)
    fused_all.to_csv("data/fused_eeg_ecg.csv", index=False)
    print("💾 Fused data saved to data/fused_eeg_ecg.csv")