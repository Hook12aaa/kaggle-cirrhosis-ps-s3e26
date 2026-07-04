"""Stateless feature engineering for the cirrhosis (PBC) survival task.

engineer_features(df) adds derived columns and returns the augmented frame.
Row count is never changed. Every transform is per-row and stateless so it
applies identically to train, test, and the appended original UCI rows.
Stateful encodings (target/frequency encoding) live in the model harness.
"""

import numpy as np
import pandas as pd

REQUIRED = [
    "N_Days", "Drug", "Age", "Sex", "Ascites", "Hepatomegaly", "Spiders",
    "Edema", "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage",
]

NUMERIC_SOURCE = [
    "N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper",
    "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin", "Stage",
]

ENGINEERED_COLUMNS = [
    "Age_years", "Bilirubin_log1p", "Cholesterol_log1p", "Copper_log1p",
    "Alk_Phos_log1p", "Tryglicerides_log1p", "SGOT_log1p", "N_Days_log1p",
    "Edema_ord", "Edema_diuretic_resistant", "decompensation_count",
    "Bili_Albumin_ratio", "AST_Platelet_ratio", "AlkPhos_Bili_ratio",
    "SGOT_AlkPhos_ratio", "Bili_x_Copper", "Chol_minus_Tryg_log",
    "cholestasis_composite", "mayo_risk_score", "Stage4_flag",
    "Stage_x_Bili", "Stage_x_Platelet",
]


def _safe_denom(s):
    return s.replace(0, np.nan)


def engineer_features(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"engineer_features missing required columns: {missing}")

    out = df.copy()
    for col in NUMERIC_SOURCE:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["Age_years"] = out["Age"] / 365.25

    for col in ["Bilirubin", "Cholesterol", "Copper", "Alk_Phos",
                "Tryglicerides", "SGOT", "N_Days"]:
        out[f"{col}_log1p"] = np.log1p(out[col])

    edema_ord_map = {"N": 0, "S": 1, "Y": 2}
    out["Edema_ord"] = out["Edema"].map(edema_ord_map)
    out["Edema_diuretic_resistant"] = (out["Edema"] == "Y").astype(int)

    out["decompensation_count"] = (
        (out["Ascites"] == "Y").astype(int)
        + (out["Hepatomegaly"] == "Y").astype(int)
        + (out["Spiders"] == "Y").astype(int)
        + out["Edema"].isin(["S", "Y"]).astype(int)
    )

    out["Bili_Albumin_ratio"] = out["Bilirubin"] / _safe_denom(out["Albumin"])
    out["AST_Platelet_ratio"] = out["SGOT"] / _safe_denom(out["Platelets"])
    out["AlkPhos_Bili_ratio"] = out["Alk_Phos"] / _safe_denom(out["Bilirubin"])
    out["SGOT_AlkPhos_ratio"] = out["SGOT"] / _safe_denom(out["Alk_Phos"])

    out["Bili_x_Copper"] = np.log1p(out["Bilirubin"]) * np.log1p(out["Copper"])
    out["Chol_minus_Tryg_log"] = np.log1p(out["Cholesterol"]) - np.log1p(out["Tryglicerides"])
    out["cholestasis_composite"] = (
        np.log1p(out["Bilirubin"]) + np.log1p(out["Alk_Phos"]) + np.log1p(out["Copper"])
    )

    edema_score = out["Edema"].map({"N": 0.0, "S": 0.5, "Y": 1.0})
    out["mayo_risk_score"] = (
        0.871 * np.log(out["Bilirubin"])
        - 2.53 * np.log(out["Albumin"])
        + 0.039 * out["Age_years"]
        + 2.38 * np.log(out["Prothrombin"])
        + 0.859 * edema_score
    )

    out["Stage4_flag"] = (out["Stage"] == 4).astype(int)
    out["Stage_x_Bili"] = out["Stage"] * np.log1p(out["Bilirubin"])
    out["Stage_x_Platelet"] = out["Stage"] * out["Platelets"]

    return out
