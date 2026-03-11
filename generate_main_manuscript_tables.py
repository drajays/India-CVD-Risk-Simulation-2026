#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# =========================================================
# CONFIG
# =========================================================
INPUT_FILE = "FINAL_MANUSCRIPT_DATA.csv.zip"   # or FINAL_MANUSCRIPT_DATA.csv
OUTPUT_PREFIX = "india_cvd_manuscript_stats"

ARCH_ORDER = ["A", "B", "C", "D", "E"]
ARCH_LABELS_T1 = {
    "A": "A: Young",
    "B": "B: Thin-Fat",
    "C": "C: CKD-DM",
    "D": "D: Female",
    "E": "E: Standard",
}
ARCH_LABELS_T2 = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
}

# =========================================================
# HELPERS
# =========================================================
def mdot(s: str) -> str:
    return str(s).replace(".", "·")

def fmt_mean_sd(x, digits=1):
    return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}"

def fmt_pct(x, digits=1):
    return f"{100 * x.mean():.{digits}f}"

def fmt_num(x, digits=1):
    return f"{x:.{digits}f}"

def pretty_table(df: pd.DataFrame, title: str) -> str:
    return f"{title}\n" + df.to_string(index=False) + "\n"

def safe_read_csv(path):
    if path.endswith(".zip"):
        return pd.read_csv(path, compression="zip")
    return pd.read_csv(path)

# =========================================================
# LOAD DATA
# =========================================================
df = safe_read_csv(INPUT_FILE)

required_cols = [
    "archetype", "age", "sex", "tc", "hdl", "ldl", "tg", "non_hdl", "sbp",
    "bmi", "hba1c", "egfr", "dm", "smoking", "fhx", "bp_rx", "on_statin",
    "frs_cvd", "pce", "prevent", "qrisk3_indian", "qrisk3_white",
    "who_ish", "score2", "lai"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---------------------------------------------------------
# Create treatment columns if absent
# ---------------------------------------------------------
if "rx_frs" not in df.columns:
    df["rx_frs"] = (df["frs_cvd"] >= 10.0).astype(int)
if "rx_pce" not in df.columns:
    df["rx_pce"] = (df["pce"] >= 7.5).astype(int)
if "rx_prevent" not in df.columns:
    df["rx_prevent"] = (df["prevent"] >= 7.5).astype(int)
if "rx_qrisk3" not in df.columns:
    df["rx_qrisk3"] = (df["qrisk3_indian"] >= 10.0).astype(int)
if "rx_who" not in df.columns:
    df["rx_who"] = (df["who_ish"] >= 10.0).astype(int)
if "rx_score2" not in df.columns:
    df["rx_score2"] = np.where(
        df["age"] < 50,
        (df["score2"] >= 5.0).astype(int),
        (df["score2"] >= 10.0).astype(int),
    )
if "rx_lai" not in df.columns:
    df["rx_lai"] = (df["lai"] >= 10.0).astype(int)

calc_cols = ["rx_frs", "rx_pce", "rx_prevent", "rx_qrisk3", "rx_who", "rx_score2", "rx_lai"]

if "discordant" not in df.columns:
    s = df[calc_cols].sum(axis=1)
    df["discordant"] = ((s > 0) & (s < len(calc_cols))).astype(int)

# =========================================================
# TABLE 1
# =========================================================
table1_specs = [
    ("Age, years",               lambda d: fmt_mean_sd(d["age"])),
    ("Male, %",                  lambda d: fmt_pct(d["sex"] == 1)),
    ("Total cholesterol, mg/dL", lambda d: fmt_mean_sd(d["tc"])),
    ("HDL-C, mg/dL",             lambda d: fmt_mean_sd(d["hdl"])),
    ("LDL-C, mg/dL",             lambda d: fmt_mean_sd(d["ldl"])),
    ("Triglycerides, mg/dL",     lambda d: fmt_mean_sd(d["tg"])),
    ("Non-HDL-C, mg/dL",         lambda d: fmt_mean_sd(d["non_hdl"])),
    ("SBP, mm Hg",               lambda d: fmt_mean_sd(d["sbp"])),
    ("BMI, kg/m²",               lambda d: fmt_mean_sd(d["bmi"])),
    ("HbA1c, %",                 lambda d: fmt_mean_sd(d["hba1c"])),
    ("eGFR, mL/min/1·73 m²",     lambda d: fmt_mean_sd(d["egfr"])),
    ("Type 2 diabetes, %",       lambda d: fmt_pct(d["dm"] == 1)),
    ("Current smoking, %",       lambda d: fmt_pct(d["smoking"] == 1)),
    ("Family history, %",        lambda d: fmt_pct(d["fhx"] == 1)),
    ("On antihypertensive, %",   lambda d: fmt_pct(d["bp_rx"] == 1)),
    ("On statin, %",             lambda d: fmt_pct(d["on_statin"] == 1)),
]

table1_rows = []
for label, func in table1_specs:
    row = {"Parameter": label}
    for a in ARCH_ORDER:
        row[ARCH_LABELS_T1[a]] = mdot(func(df[df["archetype"] == a]))
    row["Overall"] = mdot(func(df))
    table1_rows.append(row)

table1 = pd.DataFrame(table1_rows)

# =========================================================
# TABLE 2
# =========================================================
table2_specs = [
    ("FRS-CVD (≥10%)",    "rx_frs"),
    ("PCE (≥7·5%)",       "rx_pce"),
    ("PREVENT (≥7·5%)",   "rx_prevent"),
    ("QRISK3 (≥10%)",     "rx_qrisk3"),
    ("WHO SEAR-D (≥10%)", "rx_who"),
    ("SCORE2",            "rx_score2"),
    ("LAI (≥10%)",        "rx_lai"),
    ("Discordance (%)",   "discordant"),
]

table2_rows = []
for label, col in table2_specs:
    row = {"Framework": label}
    for a in ARCH_ORDER:
        row[ARCH_LABELS_T2[a]] = mdot(fmt_pct(df.loc[df["archetype"] == a, col]))
    row["Overall"] = mdot(fmt_pct(df[col]))
    table2_rows.append(row)

table2 = pd.DataFrame(table2_rows)

# =========================================================
# TABLE 3
# =========================================================
df["lai_high_risk"] = (df["lai"] >= 10.0).astype(int)

X = pd.DataFrame({
    "Diabetes mellitus": df["dm"].astype(int),
    "Current smoking": df["smoking"].astype(int),
    "Age ≥50 years": (df["age"] >= 50).astype(int),
    "SBP ≥140 mm Hg": (df["sbp"] >= 140).astype(int),
    "Triglycerides ≥150 mg/dL": (df["tg"] >= 150).astype(int),
    "Low HDL-C (<40 M / <45 F)": (
        ((df["sex"] == 1) & (df["hdl"] < 40)) |
        ((df["sex"] == 0) & (df["hdl"] < 45))
    ).astype(int),
})

y = df["lai_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=2026, stratify=y
)

clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_train, y_train)

coef = clf.coef_[0].copy()
coef_pos = np.maximum(coef, 0)
base_coef = np.min(coef_pos[coef_pos > 0])
weights = np.round(2 * coef_pos / base_coef).astype(int)

raw_probs = clf.predict_proba(X_test)[:, 1]
raw_auc = roc_auc_score(y_test, raw_probs)

point_scores = X_test.dot(weights)
score_auc = roc_auc_score(y_test, point_scores)
preds = (point_scores >= 10).astype(int)
acc = accuracy_score(y_test, preds)

rationale_map = {
    "Diabetes mellitus": "Strongest independent predictor; 2–4× ASCVD risk",
    "Current smoking": "~2× relative risk; 24% male prevalence in India",
    "Age ≥50 years": "Inflection in Indian CVD event rates",
    "SBP ≥140 mm Hg": "Stage 2 hypertension threshold",
    "Triglycerides ≥150 mg/dL": "Indian atherogenic dyslipidaemia marker",
    "Low HDL-C (<40 M / <45 F)": "Present in 66·9% of Indian adults",
}

table3_points = pd.DataFrame({
    "Clinical Variable": list(X.columns),
    "Points": [f"+{int(w)}" for w in weights],
    "Rationale": [rationale_map[c] for c in X.columns],
})

table3_perf = pd.DataFrame({
    "Holdout Performance (N = 30,000)": [
        "Area under ROC curve (AUC)",
        "Accuracy",
        "Proposed use",
    ],
    "Value": [
        mdot(f"{score_auc:.2f}"),
        mdot(f"{100 * acc:.1f}%"),
        "Bedside heuristic for LAI consensus; requires prospective validation",
    ]
})

# =========================================================
# QRISK3 SUMMARY
# =========================================================
overall_indian = df["qrisk3_indian"].mean()
overall_white = df["qrisk3_white"].mean()

qrisk_arch = (
    df.groupby("archetype")[["qrisk3_indian", "qrisk3_white"]]
      .mean()
      .round(1)
      .reset_index()
)

qrisk_rx = (
    df.assign(
        rx_qrisk3_indian=(df["qrisk3_indian"] >= 10).astype(int),
        rx_qrisk3_white=(df["qrisk3_white"] >= 10).astype(int),
    )
    .groupby("archetype")[["rx_qrisk3_indian", "rx_qrisk3_white"]]
    .mean()
    .mul(100)
    .round(1)
    .reset_index()
)

qrisk_summary = pd.DataFrame({
    "Metric": [
        "Overall QRISK3 Indian mean",
        "Overall QRISK3 White mean",
        "Sentence for manuscript",
    ],
    "Value": [
        mdot(f"{overall_indian:.1f}%"),
        mdot(f"{overall_white:.1f}%"),
        f"The QRISK3 ‘Indian’ ethnicity designation yielded "
        f"{'lower' if overall_indian < overall_white else 'higher'} mean risk "
        f"({mdot(f'{overall_indian:.1f}')}%) than the ‘White’ designation "
        f"({mdot(f'{overall_white:.1f}')}%)."
    ]
})

# =========================================================
# ADDITIONAL MAIN MANUSCRIPT STATS
# =========================================================
# Statin eligibility range across 7 frameworks
framework_means = {
    "FRS-CVD": 100 * df["rx_frs"].mean(),
    "PCE": 100 * df["rx_pce"].mean(),
    "PREVENT": 100 * df["rx_prevent"].mean(),
    "QRISK3": 100 * df["rx_qrisk3"].mean(),
    "WHO": 100 * df["rx_who"].mean(),
    "SCORE2": 100 * df["rx_score2"].mean(),
    "LAI": 100 * df["rx_lai"].mean(),
}
statin_min_name = min(framework_means, key=framework_means.get)
statin_max_name = max(framework_means, key=framework_means.get)
statin_min_val = framework_means[statin_min_name]
statin_max_val = framework_means[statin_max_name]

# Diabetic drug eligibility range among diabetic patients
df_dm = df[df["dm"] == 1].copy()
dm_framework_means = {
    "FRS-CVD": 100 * df_dm["rx_frs"].mean(),
    "PCE": 100 * df_dm["rx_pce"].mean(),
    "PREVENT": 100 * df_dm["rx_prevent"].mean(),
    "QRISK3": 100 * df_dm["rx_qrisk3"].mean(),
    "WHO": 100 * df_dm["rx_who"].mean(),
    "SCORE2": 100 * df_dm["rx_score2"].mean(),
    "LAI": 100 * df_dm["rx_lai"].mean(),
}
dm_min_name = min(dm_framework_means, key=dm_framework_means.get)
dm_max_name = max(dm_framework_means, key=dm_framework_means.get)
dm_min_val = dm_framework_means[dm_min_name]
dm_max_val = dm_framework_means[dm_max_name]

india_diabetes_adults_m = 101.0
eligibility_swing_pct_points = dm_max_val - dm_min_val
eligibility_swing_millions = india_diabetes_adults_m * eligibility_swing_pct_points / 100.0

main_stats = pd.DataFrame({
    "Metric": [
        "Overall treatment-decision discordance",
        "Statin eligibility minimum",
        "Statin eligibility maximum",
        "Diabetic cardioprotective drug eligibility minimum",
        "Diabetic cardioprotective drug eligibility maximum",
        "Eligibility swing across 101M adults with T2D",
        "Raw logistic model AUC",
        "Integer point-score AUC",
        "Integer point-score accuracy",
    ],
    "Value": [
        mdot(f"{100 * df['discordant'].mean():.1f}%"),
        f"{mdot(f'{statin_min_val:.1f}')}% ({statin_min_name})",
        f"{mdot(f'{statin_max_val:.1f}')}% ({statin_max_name})",
        f"{mdot(f'{dm_min_val:.1f}')}% ({dm_min_name})",
        f"{mdot(f'{dm_max_val:.1f}')}% ({dm_max_name})",
        f"{mdot(f'{eligibility_swing_millions:.1f}')} million",
        mdot(f"{raw_auc:.3f}"),
        mdot(f"{score_auc:.3f}"),
        mdot(f"{100 * acc:.1f}%"),
    ]
})

# =========================================================
# SAVE FILES
# =========================================================
table1.to_csv(f"{OUTPUT_PREFIX}_table1.csv", index=False)
table2.to_csv(f"{OUTPUT_PREFIX}_table2.csv", index=False)
table3_points.to_csv(f"{OUTPUT_PREFIX}_table3_points.csv", index=False)
table3_perf.to_csv(f"{OUTPUT_PREFIX}_table3_performance.csv", index=False)
qrisk_arch.to_csv(f"{OUTPUT_PREFIX}_qrisk3_by_archetype.csv", index=False)
qrisk_rx.to_csv(f"{OUTPUT_PREFIX}_qrisk3_treatment_by_archetype.csv", index=False)
qrisk_summary.to_csv(f"{OUTPUT_PREFIX}_qrisk3_summary.csv", index=False)
main_stats.to_csv(f"{OUTPUT_PREFIX}_main_stats.csv", index=False)

with pd.ExcelWriter(f"{OUTPUT_PREFIX}_all_tables.xlsx", engine="openpyxl") as writer:
    table1.to_excel(writer, sheet_name="Table1", index=False)
    table2.to_excel(writer, sheet_name="Table2", index=False)
    table3_points.to_excel(writer, sheet_name="Table3_Points", index=False)
    table3_perf.to_excel(writer, sheet_name="Table3_Performance", index=False)
    qrisk_arch.to_excel(writer, sheet_name="QRISK3_By_Archetype", index=False)
    qrisk_rx.to_excel(writer, sheet_name="QRISK3_Tx_By_Archetype", index=False)
    qrisk_summary.to_excel(writer, sheet_name="QRISK3_Summary", index=False)
    main_stats.to_excel(writer, sheet_name="Main_Stats", index=False)

# =========================================================
# TXT REPORT
# =========================================================
txt_path = f"{OUTPUT_PREFIX}_report.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("INDIA-CVD MANUSCRIPT STATS REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write("MAIN MANUSCRIPT HEADLINE STATS\n")
    f.write(main_stats.to_string(index=False))
    f.write("\n\n")

    f.write("QRISK3 SUMMARY\n")
    f.write(qrisk_summary.to_string(index=False))
    f.write("\n\n")

    f.write("QRISK3 BY ARCHETYPE (MEAN RISK)\n")
    f.write(qrisk_arch.to_string(index=False))
    f.write("\n\n")

    f.write("QRISK3 TREATMENT BY ARCHETYPE (%)\n")
    f.write(qrisk_rx.to_string(index=False))
    f.write("\n\n")

    f.write(pretty_table(table1, "TABLE 1. Baseline Characteristics"))
    f.write("\n")
    f.write(pretty_table(table2, "TABLE 2. Statin Eligibility (%)"))
    f.write("\n")
    f.write(pretty_table(table3_points, "TABLE 3A. India-CVD Triage Score: Point Assignment"))
    f.write("\n")
    f.write(pretty_table(table3_perf, "TABLE 3B. Holdout Performance"))
    f.write("\n")

# =========================================================
# PRINT TO SCREEN
# =========================================================
print("\nMAIN MANUSCRIPT HEADLINE STATS\n")
print(main_stats.to_string(index=False))

print("\nQRISK3 SUMMARY\n")
print(qrisk_summary.to_string(index=False))

print("\nQRISK3 BY ARCHETYPE\n")
print(qrisk_arch.to_string(index=False))

print("\nQRISK3 TREATMENT BY ARCHETYPE (%)\n")
print(qrisk_rx.to_string(index=False))

print("\nTABLE 1\n")
print(table1.to_string(index=False))

print("\nTABLE 2\n")
print(table2.to_string(index=False))

print("\nTABLE 3A. India-CVD Triage Score: Point Assignment\n")
print(table3_points.to_string(index=False))

print("\nTABLE 3B. Holdout Performance\n")
print(table3_perf.to_string(index=False))

print("\nSaved files:")
print(f"  {OUTPUT_PREFIX}_table1.csv")
print(f"  {OUTPUT_PREFIX}_table2.csv")
print(f"  {OUTPUT_PREFIX}_table3_points.csv")
print(f"  {OUTPUT_PREFIX}_table3_performance.csv")
print(f"  {OUTPUT_PREFIX}_qrisk3_by_archetype.csv")
print(f"  {OUTPUT_PREFIX}_qrisk3_treatment_by_archetype.csv")
print(f"  {OUTPUT_PREFIX}_qrisk3_summary.csv")
print(f"  {OUTPUT_PREFIX}_main_stats.csv")
print(f"  {OUTPUT_PREFIX}_all_tables.xlsx")
print(f"  {OUTPUT_PREFIX}_report.txt")