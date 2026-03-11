#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate supplementary-material tables and sensitivity analyses
from FINAL_MANUSCRIPT_DATA.csv.zip

Outputs:
- CSV files for each table
- One consolidated TXT report
- Data-driven values only where possible
"""

import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# =============================================================================
# CONFIG
# =============================================================================

INPUT_FILE = "FINAL_MANUSCRIPT_DATA.csv.zip"
OUTPUT_DIR = "supplementary_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ARCH_ORDER = ["A", "B", "C", "D", "E"]
ARCH_LABELS = {
    "A": "A: Young",
    "B": "B: Thin-Fat",
    "C": "C: CKD-DM",
    "D": "D: Female",
    "E": "E: Standard",
}
CALC_RX_COLS = {
    "FRS-CVD": "rx_frs",
    "PCE": "rx_pce",
    "PREVENT": "rx_prevent",
    "QRISK3": "rx_qrisk3",
    "WHO": "rx_who",
    "SCORE2": "rx_score2",
    "LAI": "rx_lai",
}
RISK_COLS = {
    "FRS-CVD": "frs_cvd",
    "PCE": "pce",
    "PREVENT": "prevent",
    "QRISK3 (Indian)": "qrisk3_indian",
    "QRISK3 (White)": "qrisk3_white",
    "WHO SEAR-D": "who_ish",
    "SCORE2/DM": "score2",
    "LAI 2023": "lai",
}
PREDICTED_OUTCOME = {
    "FRS-CVD": "General CVD",
    "PCE": "Hard ASCVD",
    "PREVENT": "Total CVD",
    "QRISK3 (Indian)": "CVD events",
    "QRISK3 (White)": "CVD events",
    "WHO SEAR-D": "Fatal+NF CVD",
    "SCORE2/DM": "Fatal+NF CVD",
    "LAI 2023": "ASCVD (India)",
}

PREVALENCE_WEIGHTS = {
    "A": 0.05,
    "B": 0.10,
    "C": 0.08,
    "D": 0.15,
    "E": 0.62,
}


# =============================================================================
# HELPERS
# =============================================================================

def read_final_dataset(path: str) -> pd.DataFrame:
    if path.endswith(".zip"):
        return pd.read_csv(path, compression="zip")
    return pd.read_csv(path)


def fmt1(x):
    """One decimal with Lancet mid-dot."""
    return f"{x:.1f}".replace(".", "·")


def fmt2(x):
    """Two decimals with Lancet mid-dot."""
    return f"{x:.2f}".replace(".", "·")


def mean_sd(series: pd.Series) -> str:
    return f"{fmt1(series.mean())} ± {fmt1(series.std(ddof=1))}"


def pct(series: pd.Series) -> str:
    return fmt1(series.mean() * 100)


def ordered_group_keys(df):
    return ARCH_ORDER + ["Overall"]


def subset_by_arch(df, arch):
    return df if arch == "Overall" else df[df["archetype"] == arch]


def save_csv(df: pd.DataFrame, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    return path


def write_txt(lines, filename="SUPPLEMENTARY_DERIVED_REPORT.txt"):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# =============================================================================
# ST1: Detailed baseline characteristics
# =============================================================================

def build_st1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    diabetic = df["dm"] == 1

    parameters = [
        ("UACR, mg/g", lambda d: mean_sd(d["uacr"])),
        ("DM diagnosis age, years", lambda d: mean_sd(d.loc[d["dm"] == 1, "dm_diag_age"]) if (d["dm"] == 1).any() else "NA"),
        ("Atrial fibrillation, %", lambda d: pct(d["af"])),
        ("Rheumatoid arthritis, %", lambda d: pct(d["ra"])),
        ("On statin, %", lambda d: pct(d["on_statin"])),
        ("Family history, %", lambda d: pct(d["fhx"])),
    ]

    for param_name, func in parameters:
        row = {"Parameter": param_name}
        for key in ordered_group_keys(df):
            d = subset_by_arch(df, key)
            row[ARCH_LABELS.get(key, key)] = func(d)
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.rename(columns={"Overall": "Overall"})


# =============================================================================
# ST2: Mean 10-year ASCVD risk estimates
# =============================================================================

def build_st2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calc_name, col in RISK_COLS.items():
        row = {"Calculator": calc_name}
        for key in ordered_group_keys(df):
            d = subset_by_arch(df, key)
            row[ARCH_LABELS.get(key, key)] = fmt1(d[col].mean())
        row["Predicted outcome"] = PREDICTED_OUTCOME[calc_name]
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# ST3: Cardioprotective drug eligibility among diabetic patients
# =============================================================================

def build_st3(df: pd.DataFrame) -> pd.DataFrame:
    dfdm = df[df["dm"] == 1].copy()
    rows = []
    for fw, col in CALC_RX_COLS.items():
        row = {"Framework": fw}
        total_n = int(dfdm[col].sum())
        for key in ordered_group_keys(dfdm):
            d = subset_by_arch(dfdm, key)
            row[ARCH_LABELS.get(key, key)] = fmt1(d[col].mean() * 100)
        row["N eligible"] = total_n
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# ST4 / ST5 / ST8: Manual scaffolds
# =============================================================================

def build_st4_manual() -> pd.DataFrame:
    return pd.DataFrame([
        ["Age and sex", "✓", "✓", "✓", "✓", "✓", "✓", "✓"],
        ["Total cholesterol", "✓", "✓", "—", "—", "✓", "✓", "✓"],
        ["HDL-C", "✓", "✓", "✓", "✓", "—", "✓", "✓"],
        ["Non-HDL-C", "—", "—", "✓", "—", "—", "✓", "✓"],
        ["TC/HDL ratio", "—", "—", "—", "✓", "—", "—", "—"],
        ["Systolic BP", "✓", "✓", "✓", "✓", "✓", "✓", "✓"],
        ["BP treatment", "✓", "✓", "✓", "✓", "—", "—", "✓"],
        ["Smoking", "✓", "✓", "✓", "✓", "✓", "✓", "✓"],
        ["Diabetes", "✓", "✓", "✓", "✓", "✓", "DM ext", "✓"],
        ["HbA1c", "—", "—", "opt", "—", "—", "DM ext", "✓"],
        ["eGFR / CKD", "—", "—", "✓", "✓", "—", "DM ext", "✓"],
        ["Family history", "—", "—", "—", "✓", "—", "—", "✓"],
        ["BMI / obesity", "—", "—", "✓", "✓", "—", "—", "✓"],
        ["Ethnicity", "✓*", "—", "—", "✓", "—", "—", "✓"],
        ["Atrial fibrillation", "—", "—", "—", "✓", "—", "—", "—"],
    ], columns=["Parameter", "PCE", "FRS-CVD", "PREVENT", "QRISK3", "WHO", "SCORE2", "LAI"])


def build_st5_manual() -> pd.DataFrame:
    return pd.DataFrame([
        ["PCE", "2013", "United States", "~24,000", "White + Black", "None", "Hard ASCVD", "40–79"],
        ["FRS-CVD", "2008", "United States", "~8,000", "White", "None", "General CVD", "30–74"],
        ["PREVENT", "2023", "United States", "~3·3M", "Diverse US", "Minimal", "Total CVD", "30–79"],
        ["QRISK3", "2017", "United Kingdom", "~10M", "UK primary care", "Yes (Indian)", "CVD events", "25–84"],
        ["WHO", "2019", "Global (modelled)", "N/A", "Regional models", "SEAR-D region", "Fatal+NF CVD", "40–80"],
        ["SCORE2", "2021", "Europe", "~680,000", "European", "None", "Fatal+NF CVD", "40–69"],
        ["SCORE2-DM", "2023", "Europe", "~229,000", "European T2D", "None", "Fatal+NF CVD", "40–69"],
        ["LAI 2023", "2023", "India (consensus)", "N/A", "Indian", "100%", "ASCVD (India)", "40+"],
    ], columns=["Calculator", "Year", "Geography", "N derivation", "Primary ethnicity", "SA data", "Outcome", "Age range"])


def build_st8_manual(df: pd.DataFrame) -> pd.DataFrame:
    e = df[df["archetype"] == "E"]
    rows = [
        ["Age, years", mean_sd(e["age"]), "40–70 range", "ICMR-INDIAB", "Within range", "OPD age distribution"],
        ["Male, %", pct(e["sex"]), "58–65%", "Indian OPD data", "Match", "Referral bias accounted"],
        ["TC, mg/dL", mean_sd(e["tc"]), "176·7 ± 42·1", "FitHeart", "Close", "Higher TC intended (OPD)"],
        ["HDL-C, mg/dL", mean_sd(e["hdl"]), "43·2 ± 11·7", "FitHeart", "Match", ""],
        ["TG, mg/dL", mean_sd(e["tg"]), "162·3 ± 106·7", "FitHeart", "Match", ""],
        ["LDL-C, mg/dL", mean_sd(e["ldl"]), "110·5 ± 34·0", "FitHeart", "Close", "Friedewald-derived"],
        ["SBP, mm Hg", mean_sd(e["sbp"]), "130 ± 18", "CARRS", "Match", ""],
        ["BMI, kg/m²", mean_sd(e["bmi"]), "25·0", "ICMR-INDIAB", "Match", ""],
        ["DM, %", pct(e["dm"]), "11·4–22%", "ICMR-INDIAB", "Within range", "OPD enrichment"],
        ["Smoking, %", pct(e["smoking"]), "~24% male", "NFHS-5", "Match", ""],
        ["eGFR", mean_sd(e["egfr"]), "~90", "CARRS", "Match", ""],
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Synthetic (E)", "Published reference", "Source", "Concordance", "Note"])


# =============================================================================
# ST6: Pairwise Cohen's kappa
# =============================================================================

def build_st6(df: pd.DataFrame) -> pd.DataFrame:
    names = list(CALC_RX_COLS.keys())
    mat = pd.DataFrame(index=names, columns=names, dtype=object)

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j < i:
                mat.loc[n1, n2] = "—"
            else:
                k = cohen_kappa_score(df[CALC_RX_COLS[n1]], df[CALC_RX_COLS[n2]])
                mat.loc[n1, n2] = fmt3(k)
    mat.insert(0, "Framework", mat.index)
    return mat.reset_index(drop=True)


def fmt3(x):
    return f"{x:.3f}".replace(".", "·")


# =============================================================================
# ST7: QRISK3 ethnicity toggle
# =============================================================================

def build_st7(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in ARCH_ORDER + ["Overall"]:
        d = subset_by_arch(df, key)
        indian_mean = d["qrisk3_indian"].mean()
        white_mean = d["qrisk3_white"].mean()
        rx_ind = (d["qrisk3_indian"] >= 10).mean() * 100
        rx_white = (d["qrisk3_white"] >= 10).mean() * 100
        rows.append({
            "Archetype": key,
            "Indian mean": f"{fmt1(indian_mean)}%",
            "White mean": f"{fmt1(white_mean)}%",
            "Δ (pp)": fmt1(indian_mean - white_mean),
            "Ratio": fmt2(indian_mean / white_mean) if white_mean != 0 else "NA",
            "Rx Indian": f"{fmt1(rx_ind)}%",
            "Rx White": f"{fmt1(rx_white)}%",
            "Δ Rx (pp)": fmt1(rx_ind - rx_white),
            "N": len(d),
        })
    return pd.DataFrame(rows)


# =============================================================================
# S3.1: Uniform threshold harmonisation
# =============================================================================

def build_s31_uniform_threshold(df: pd.DataFrame, threshold=7.5):
    risk_cols = {
        "FRS-CVD": "frs_cvd",
        "PCE": "pce",
        "PREVENT": "prevent",
        "QRISK3": "qrisk3_indian",
        "WHO": "who_ish",
        "SCORE2": "score2",
        "LAI": "lai",
    }

    rx_uniform = pd.DataFrame(index=df.index)
    for name, col in risk_cols.items():
        rx_uniform[name] = (df[col] >= threshold).astype(int)

    sum_rx = rx_uniform.sum(axis=1)
    discordance = ((sum_rx > 0) & (sum_rx < len(risk_cols))).mean() * 100

    eligibility = {name: rx_uniform[name].mean() * 100 for name in risk_cols}
    return eligibility, discordance


# =============================================================================
# S3.2: Prevalence-weighted discordance
# =============================================================================

def build_s32_prevalence_weighted(df: pd.DataFrame):
    if "discordant" not in df.columns:
        calcs = list(CALC_RX_COLS.values())
        sum_rx = df[calcs].sum(axis=1)
        df = df.copy()
        df["discordant"] = ((sum_rx > 0) & (sum_rx < len(calcs))).astype(int)

    arch_discordance = df.groupby("archetype")["discordant"].mean() * 100
    weighted = sum(arch_discordance[a] * PREVALENCE_WEIGHTS[a] for a in PREVALENCE_WEIGHTS)
    return arch_discordance, weighted


# =============================================================================
# S3.3: QRISK3 statin sensitivity in Archetype C
# =============================================================================

def build_s33_qrisk3_statin_sensitivity(df: pd.DataFrame):
    c = df[df["archetype"] == "C"].copy()
    c_free = c[c["on_statin"] == 0]
    c_treated = c[c["on_statin"] == 1]

    out = {
        "statin_free_n": len(c_free),
        "statin_free_mean": c_free["qrisk3_indian"].mean(),
        "statin_free_rx": (c_free["qrisk3_indian"] >= 10).mean() * 100,
        "statin_treated_n": len(c_treated),
        "statin_treated_mean": c_treated["qrisk3_indian"].mean(),
        "statin_treated_rx": (c_treated["qrisk3_indian"] >= 10).mean() * 100,
        "delta_mean_pp": c_treated["qrisk3_indian"].mean() - c_free["qrisk3_indian"].mean(),
    }
    return out


# =============================================================================
# MAIN REPORT
# =============================================================================

def main():
    df = read_final_dataset(INPUT_FILE)

    # Ensure discordant available
    if "discordant" not in df.columns:
        calcs = list(CALC_RX_COLS.values())
        sum_rx = df[calcs].sum(axis=1)
        df["discordant"] = ((sum_rx > 0) & (sum_rx < len(calcs))).astype(int)

    # Build tables
    st1 = build_st1(df)
    st2 = build_st2(df)
    st3 = build_st3(df)
    st4 = build_st4_manual()
    st5 = build_st5_manual()
    st6 = build_st6(df)
    st7 = build_st7(df)
    st8 = build_st8_manual(df)

    # Sensitivity analyses
    s31_eligibility, s31_discordance = build_s31_uniform_threshold(df, threshold=7.5)
    s32_arch, s32_weighted = build_s32_prevalence_weighted(df)
    s33 = build_s33_qrisk3_statin_sensitivity(df)

    # Save CSVs
    save_csv(st1, "ST1_detailed_baseline.csv")
    save_csv(st2, "ST2_mean_risk_estimates.csv")
    save_csv(st3, "ST3_diabetes_drug_eligibility.csv")
    save_csv(st4, "ST4_calculator_parameters_manual.csv")
    save_csv(st5, "ST5_derivation_cohorts_manual.csv")
    save_csv(st6, "ST6_pairwise_kappa.csv")
    save_csv(st7, "ST7_qrisk3_ethnicity_toggle.csv")
    save_csv(st8, "ST8_external_validation_manual.csv")

    # Consolidated TXT
    lines = []
    lines.append("SUPPLEMENTARY MATERIAL - DATA-DRIVEN DERIVATIONS")
    lines.append("=" * 80)

    lines.append("\nS2. ST1 Detailed Baseline Characteristics")
    lines.append(st1.to_string(index=False))

    lines.append("\nS2. ST2 Mean 10-Year ASCVD Risk Estimates (%)")
    lines.append(st2.to_string(index=False))

    lines.append("\nS2. ST3 Cardioprotective Drug Eligibility (%) Among Diabetic Patients")
    lines.append(st3.to_string(index=False))

    lines.append("\nS2. ST4 Clinical Parameters Used by Each Calculator (manual scaffold)")
    lines.append(st4.to_string(index=False))

    lines.append("\nS2. ST5 Derivation Cohort Characteristics (manual scaffold)")
    lines.append(st5.to_string(index=False))

    lines.append("\nS2. ST6 Pairwise Cohen's Kappa")
    lines.append(st6.to_string(index=False))

    lines.append("\nS2. ST7 QRISK3 Ethnicity Toggle")
    lines.append(st7.to_string(index=False))

    lines.append("\nS2. ST8 External Validation Against Published Indian Data (manual scaffold)")
    lines.append(st8.to_string(index=False))

    lines.append("\nS3.1 Effect of Uniform Threshold Harmonisation (7.5% for all seven calculators)")
    for name, val in s31_eligibility.items():
        lines.append(f"{name}: {fmt1(val)}%")
    lines.append(f"Overall discordance under harmonised threshold: {fmt1(s31_discordance)}%")

    lines.append("\nS3.2 Prevalence-Weighted Discordance")
    for arch in ARCH_ORDER:
        lines.append(f"Archetype {arch}: {fmt1(s32_arch[arch])}%")
    lines.append(f"Prevalence-weighted overall discordance: {fmt1(s32_weighted)}%")

    lines.append("\nS3.3 QRISK3 Statin-Exclusion Sensitivity for Archetype C")
    lines.append(
        f"Statin-free patients (N={s33['statin_free_n']}): "
        f"mean QRISK3 Indian risk {fmt1(s33['statin_free_mean'])}%, "
        f"high-risk {fmt1(s33['statin_free_rx'])}%"
    )
    lines.append(
        f"Statin-treated patients (N={s33['statin_treated_n']}): "
        f"mean QRISK3 Indian risk {fmt1(s33['statin_treated_mean'])}%, "
        f"high-risk {fmt1(s33['statin_treated_rx'])}%"
    )
    lines.append(f"Difference in mean risk: {fmt1(s33['delta_mean_pp'])} percentage points")

    report_path = write_txt(lines)

    print("Done.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Main report: {report_path}")


if __name__ == "__main__":
    main()