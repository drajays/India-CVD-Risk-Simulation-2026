#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Supplementary Material 2 tables and derived values
for:
- S4 India-CVD Triage Score
- S5 One-Year Budget Impact Scenario
from FINAL_MANUSCRIPT_DATA.csv.zip

Outputs:
- S4a_logistic_coefficients.csv
- S4b_holdout_performance.csv
- S4c_risk_stratification_full.csv
- S4d_auc_against_multiple_targets.csv
- S4e_auc_by_archetype.csv
- S5_budget_impact.csv
- SUPPLEMENTARY2_DERIVED_REPORT.txt
"""

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_curve,
)

# =========================================================
# CONFIG
# =========================================================
INPUT_FILE = "FINAL_MANUSCRIPT_DATA.csv.zip"
OUTPUT_PREFIX = ""
RANDOM_STATE = 2026

# Budget scenario assumptions
DIABETES_POPULATION_INDIA = 101_000_000
INCREMENTAL_COST_PER_YEAR_INR = 9_000  # dapagliflozin add-on assumption
# If you want to change the budget assumption, edit here only.

# =========================================================
# HELPERS
# =========================================================
def fmt1(x):
    return f"{x:.1f}".replace(".", "·")

def fmt3(x):
    return f"{x:.3f}".replace(".", "·")

def pct(x):
    return 100 * x

def safe_auc(y_true, y_score):
    # returns NaN if only one class present
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)

def add_triage_variables(df):
    out = df.copy()
    out["diabetes"] = out["dm"].astype(int)
    out["smoker"] = out["smoking"].astype(int)
    out["age_over_50"] = (out["age"] >= 50).astype(int)
    out["sbp_over_140"] = (out["sbp"] >= 140).astype(int)
    out["tg_over_150"] = (out["tg"] >= 150).astype(int)
    out["hdl_low"] = (
        ((out["sex"] == 1) & (out["hdl"] < 40)) |
        ((out["sex"] == 0) & (out["hdl"] < 45))
    ).astype(int)
    out["lai_high_risk"] = (out["lai"] >= 10.0).astype(int)
    return out

def derive_triage_model(df):
    df = add_triage_variables(df)

    feature_cols = [
        "diabetes",
        "smoker",
        "age_over_50",
        "sbp_over_140",
        "tg_over_150",
        "hdl_low",
    ]
    X = df[feature_cols]
    y = df["lai_high_risk"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    coef = clf.coef_[0]
    intercept = clf.intercept_[0]

    # Same locked derivation logic as manuscript
    pos_coef = np.maximum(coef, 0)
    base_coef = np.min(pos_coef[pos_coef > 0])
    weights = np.round(2 * pos_coef / base_coef).astype(int)

    train_df = df.loc[idx_train].copy()
    test_df = df.loc[idx_test].copy()

    # Raw logistic model predictions
    raw_probs = clf.predict_proba(X_test)[:, 1]
    raw_auc = roc_auc_score(y_test, raw_probs)

    # Integer score
    test_scores = X_test.dot(weights)
    score_auc = roc_auc_score(y_test, test_scores)

    # manuscript threshold
    thresh_score = 10
    pred_class = (test_scores >= thresh_score).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred_class).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    acc = accuracy_score(y_test, pred_class)
    kappa = cohen_kappa_score(y_test, pred_class)

    # Youden J threshold in score units
    fpr, tpr, thresholds = roc_curve(y_test, test_scores)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = thresholds[best_idx]

    # attach scores back
    df["triage_score"] = df[feature_cols].dot(weights)
    df["triage_high_risk_10"] = (df["triage_score"] >= 10).astype(int)

    # ORs
    odds_ratios = np.exp(coef)

    s4a = pd.DataFrame({
        "Variable": [
            "Diabetes mellitus",
            "Current smoking",
            "Age ≥50 years",
            "SBP ≥140 mm Hg",
            "Triglycerides ≥150 mg/dL",
            "Low HDL-C",
            "Intercept",
        ],
        "Coefficient": list(coef) + [intercept],
        "Odds Ratio": list(odds_ratios) + [np.nan],
        "Points": list(weights) + [np.nan],
        "Max possible": list(weights) + [np.nan],
    })

    s4b = pd.DataFrame({
        "Metric": [
            "Training set",
            "Holdout test set",
            "Raw logistic model AUC",
            "Integer point score AUC",
            "ΔAUC from integer rounding",
            "Optimal threshold (Youden’s J)",
            "Sensitivity (at ≥10)",
            "Specificity (at ≥10)",
            "Positive predictive value",
            "Negative predictive value",
            "Overall accuracy",
            "Cohen’s κ",
        ],
        "Value": [
            f"N = {len(train_df):,}",
            f"N = {len(test_df):,}",
            raw_auc,
            score_auc,
            raw_auc - score_auc,
            best_threshold,
            sensitivity,
            specificity,
            ppv,
            npv,
            acc,
            kappa,
        ]
    })

    return {
        "df": df,
        "feature_cols": feature_cols,
        "weights": weights,
        "coef": coef,
        "intercept": intercept,
        "clf": clf,
        "train_idx": idx_train,
        "test_idx": idx_test,
        "s4a": s4a,
        "s4b": s4b,
        "raw_auc": raw_auc,
        "score_auc": score_auc,
        "acc": acc,
        "kappa": kappa,
    }

# =========================================================
# LOAD
# =========================================================
df = pd.read_csv(INPUT_FILE, compression="zip")

# sanity checks
required_cols = [
    "age", "sex", "tc", "hdl", "ldl", "tg", "sbp", "bmi",
    "dm", "smoking", "lai", "archetype",
    "rx_frs", "rx_pce", "rx_prevent", "rx_qrisk3", "rx_who", "rx_score2", "rx_lai"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# add discordant if absent
calc_cols = ["rx_frs", "rx_pce", "rx_prevent", "rx_qrisk3", "rx_who", "rx_score2", "rx_lai"]
if "discordant" not in df.columns:
    s = df[calc_cols].sum(axis=1)
    df["discordant"] = ((s > 0) & (s < len(calc_cols))).astype(int)

# diabetic drug eligibility same operational rule as manuscript
df_dm = df[df["dm"] == 1].copy()

# =========================================================
# S4: TRIAGE SCORE
# =========================================================
model_out = derive_triage_model(df)
df = model_out["df"]
weights = model_out["weights"]
feature_cols = model_out["feature_cols"]

# ---------- Table S4a ----------
s4a = model_out["s4a"].copy()

clinical_rationale = {
    "Diabetes mellitus": "Strongest predictor; 2–4× ASCVD risk multiplier",
    "Current smoking": "~2× relative risk; 24% male prevalence in India (NFHS-5)",
    "Age ≥50 years": "Inflection in Indian CVD event rates; exposes age-bias of Western tools",
    "SBP ≥140 mm Hg": "Stage 2 hypertension threshold",
    "Triglycerides ≥150 mg/dL": "Indian atherogenic dyslipidaemia marker",
    "Low HDL-C": "Present in 66·9% of Indian adults (ICMR-INDIAB-17)",
    "Intercept": "",
}
s4a["Clinical rationale"] = s4a["Variable"].map(clinical_rationale)

# ---------- Table S4b ----------
s4b = model_out["s4b"].copy()

# ---------- Table S4c ----------
def risk_category(score):
    if 0 <= score <= 5:
        return "Low"
    elif 6 <= score <= 9:
        return "Moderate"
    elif 10 <= score <= 13:
        return "High"
    else:
        return "Very High"

df["triage_category"] = df["triage_score"].apply(risk_category)

cat_order = ["Low", "Moderate", "High", "Very High"]
score_range_map = {
    "Low": "0–5",
    "Moderate": "6–9",
    "High": "10–13",
    "Very High": "14–22",
}

s4c = (
    df.groupby("triage_category")
      .agg(
          N=("triage_category", "size"),
          lai_high_pct=("lai_high_risk", "mean"),
          mean_lai=("lai", "mean"),
          mean_pce=("pce", "mean"),
      )
      .reindex(cat_order)
      .reset_index()
)
s4c["Score"] = s4c["triage_category"].map(score_range_map)
s4c["Prevalence"] = s4c["N"] / len(df)
s4c = s4c[["triage_category", "Score", "N", "Prevalence", "lai_high_pct", "mean_lai", "mean_pce"]]
s4c.columns = ["Category", "Score", "N", "Prevalence", "LAI ≥10%", "Mean LAI", "Mean PCE"]

# ---------- Table S4d ----------
# Use holdout set only
holdout = df.loc[model_out["test_idx"]].copy()
score_holdout = holdout["triage_score"]

targets = {
    "LAI 2023 (derivation target)": ("lai", "≥10·0%", (holdout["lai"] >= 10).astype(int)),
    "SCORE2/SCORE2-DM": ("score2", "≥10% (age-strat)", holdout["rx_score2"].astype(int)),
    "ACC/AHA PCE": ("pce", "≥7·5%", (holdout["pce"] >= 7.5).astype(int)),
    "FRS-CVD": ("frs_cvd", "≥10%", (holdout["frs_cvd"] >= 10).astype(int)),
    "WHO SEAR-D": ("who_ish", "≥10%", (holdout["who_ish"] >= 10).astype(int)),
    "QRISK3 (Indian)": ("qrisk3_indian", "≥10%", (holdout["qrisk3_indian"] >= 10).astype(int)),
    "AHA PREVENT": ("prevent", "≥7·5%", (holdout["prevent"] >= 7.5).astype(int)),
}
s4d_rows = []
for name, (_, thr, y_bin) in targets.items():
    auc_val = safe_auc(y_bin, score_holdout)
    s4d_rows.append({
        "Target": name,
        "Threshold": thr,
        "AUC": auc_val,
    })
s4d = pd.DataFrame(s4d_rows)

# ---------- Table S4e ----------
s4e_rows = []
arch_label_map = {
    "A": "A: Young/Premature",
    "B": "B: Thin-Fat MetSyn",
    "C": "C: CKD-Diabetic",
    "D": "D: Post-Menopausal",
    "E": "E: Standard OPD",
}
for arch in ["A", "B", "C", "D", "E"]:
    sub = holdout[holdout["archetype"] == arch].copy()
    auc_val = safe_auc(sub["lai_high_risk"], sub["triage_score"])
    s4e_rows.append({
        "Archetype": arch_label_map[arch],
        "N (holdout)": len(sub),
        "AUC": auc_val,
        "LAI-high %": sub["lai_high_risk"].mean(),
    })
s4e = pd.DataFrame(s4e_rows)

# =========================================================
# S5: BUDGET IMPACT
# =========================================================
framework_map = {
    "WHO SEAR-D": "rx_who",
    "PCE": "rx_pce",
    "PREVENT": "rx_prevent",
    "LAI": "rx_lai",
    "QRISK3": "rx_qrisk3",
    "FRS-CVD": "rx_frs",
    "SCORE2": "rx_score2",
}

s5_rows = []
for fw_name, col in framework_map.items():
    eligible_pct = df_dm[col].mean()
    n_treated = eligible_pct * DIABETES_POPULATION_INDIA
    n_deferred = DIABETES_POPULATION_INDIA - n_treated
    incremental_cost_b = n_treated * INCREMENTAL_COST_PER_YEAR_INR / 1e9
    deferred_savings_b = n_deferred * INCREMENTAL_COST_PER_YEAR_INR / 1e9

    s5_rows.append({
        "Framework": fw_name,
        "Eligible %": eligible_pct,
        "N treated (M)": n_treated / 1e6,
        "Incremental cost (INR B/yr)": incremental_cost_b,
        "N deferred (M)": n_deferred / 1e6,
        "Deferred cost savings (INR B/yr)": deferred_savings_b,
        "Net budget (INR B/yr)": incremental_cost_b,
    })

s5 = pd.DataFrame(s5_rows)

# =========================================================
# TEXT REPORT
# =========================================================
report_lines = []

report_lines.append("SUPPLEMENTARY MATERIAL 2 — DATA-DRIVEN DERIVED REPORT")
report_lines.append("=" * 72)
report_lines.append("")

report_lines.append("S4.3 Table S4a — Logistic coefficients and point weights")
report_lines.append("-" * 72)
for _, r in s4a.iterrows():
    if r["Variable"] == "Intercept":
        report_lines.append(
            f"{r['Variable']}: coefficient={r['Coefficient']:.3f}"
        )
    else:
        report_lines.append(
            f"{r['Variable']}: coefficient={r['Coefficient']:.3f}, "
            f"OR={r['Odds Ratio']:.1f}, points=+{int(r['Points'])}"
        )
report_lines.append(f"Total possible score range: 0–{int(np.sum(weights))} points")
report_lines.append("")

report_lines.append("S4.4 Table S4b — Holdout performance")
report_lines.append("-" * 72)
for _, r in s4b.iterrows():
    val = r["Value"]
    if isinstance(val, (float, np.floating)):
        if "threshold" in r["Metric"].lower():
            out = f"{val:.0f}"
        elif "AUC" in r["Metric"] or "κ" in r["Metric"]:
            out = f"{val:.3f}"
        else:
            out = f"{100*val:.1f}%" if val <= 1 else f"{val}"
    else:
        out = str(val)
    report_lines.append(f"{r['Metric']}: {out}")
report_lines.append("")

report_lines.append("S4.5 Table S4c — Risk stratification")
report_lines.append("-" * 72)
for _, r in s4c.iterrows():
    report_lines.append(
        f"{r['Category']} ({r['Score']}): "
        f"N={int(r['N']):,}, prevalence={100*r['Prevalence']:.1f}%, "
        f"LAI≥10={100*r['LAI ≥10%']:.1f}%, mean LAI={r['Mean LAI']:.1f}%, "
        f"mean PCE={r['Mean PCE']:.1f}%"
    )
report_lines.append("")

report_lines.append("S4.6 Table S4d — Triage score AUC against multiple calculator targets")
report_lines.append("-" * 72)
for _, r in s4d.iterrows():
    report_lines.append(f"{r['Target']}: AUC={r['AUC']:.3f} at threshold {r['Threshold']}")
report_lines.append("")

report_lines.append("S4.7 Table S4e — Triage score AUC by archetype")
report_lines.append("-" * 72)
for _, r in s4e.iterrows():
    report_lines.append(
        f"{r['Archetype']}: N={int(r['N (holdout)']):,}, "
        f"AUC={r['AUC']:.3f}, LAI-high={100*r['LAI-high %']:.1f}%"
    )
report_lines.append("")

report_lines.append("S5.2 Table S5 — One-year budget impact")
report_lines.append("-" * 72)
for _, r in s5.iterrows():
    report_lines.append(
        f"{r['Framework']}: eligible={100*r['Eligible %']:.1f}%, "
        f"treated={r['N treated (M)']:.1f}M, "
        f"incremental cost={r['Incremental cost (INR B/yr)']:.0f}B INR/yr, "
        f"deferred={r['N deferred (M)']:.1f}M"
    )

lowest = s5.loc[s5["Eligible %"].idxmin()]
highest = s5.loc[s5["Eligible %"].idxmax()]
delta_patients_m = highest["N treated (M)"] - lowest["N treated (M)"]
delta_budget_b = highest["Incremental cost (INR B/yr)"] - lowest["Incremental cost (INR B/yr)"]

report_lines.append("")
report_lines.append("Budget swing summary")
report_lines.append(
    f"Lowest-eligibility framework: {lowest['Framework']} ({100*lowest['Eligible %']:.1f}%, {lowest['N treated (M)']:.1f}M treated)"
)
report_lines.append(
    f"Highest-eligibility framework: {highest['Framework']} ({100*highest['Eligible %']:.1f}%, {highest['N treated (M)']:.1f}M treated)"
)
report_lines.append(
    f"Difference: {delta_patients_m:.1f}M patients, {delta_budget_b:.0f}B INR/year"
)

report_text = "\n".join(report_lines)

# =========================================================
# SAVE CSVs
# =========================================================
s4a.to_csv(f"{OUTPUT_PREFIX}S4a_logistic_coefficients.csv", index=False)
s4b.to_csv(f"{OUTPUT_PREFIX}S4b_holdout_performance.csv", index=False)
s4c.to_csv(f"{OUTPUT_PREFIX}S4c_risk_stratification_full.csv", index=False)
s4d.to_csv(f"{OUTPUT_PREFIX}S4d_auc_against_multiple_targets.csv", index=False)
s4e.to_csv(f"{OUTPUT_PREFIX}S4e_auc_by_archetype.csv", index=False)
s5.to_csv(f"{OUTPUT_PREFIX}S5_budget_impact.csv", index=False)

with open(f"{OUTPUT_PREFIX}SUPPLEMENTARY2_DERIVED_REPORT.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print("\nSaved:")
print(" - S4a_logistic_coefficients.csv")
print(" - S4b_holdout_performance.csv")
print(" - S4c_risk_stratification_full.csv")
print(" - S4d_auc_against_multiple_targets.csv")
print(" - S4e_auc_by_archetype.csv")
print(" - S5_budget_impact.csv")
print(" - SUPPLEMENTARY2_DERIVED_REPORT.txt")