# India-CVD Risk Simulation & Triage Score

This repository contains the full computational framework and synthetic dataset for the manuscript: **"Treatment Decision Discordance Across Eight ASCVD Risk Calculators in 100,000 Synthetic Indian Patients: An Archetype-Based Treatment-Consequence Analysis for Statin and Cardiometabolic Drug Selection"**.

## 📂 Repository Contents

### Datasets (Locked & Final)
* **`FINAL_MANUSCRIPT_DATA.csv`**: The locked 100,000-patient synthetic dataset containing all baseline parameters, 8 continuous risk scores, and 7 binary treatment classifications. Anchored to ICMR-INDIAB-17 metrics.
* **`SCORE2_Diabetes_Batch_Upload_v3.csv`**: The formatted input file used for the ESC SCORE2-Diabetes web calculator.
* **`score2_diabetes_results.csv`**: Raw output from the ESC SCORE2-Diabetes web calculator (41,633 patients, 0 errors).

### Core Analysis Code & Tools
* **`ascvd_simulation.py`**: Executable Python script for the consolidated analytical pipeline.
* **`ascvd_simulation.ipynb`**: Interactive Jupyter/Colab Notebook containing the complete analysis pipeline, Triage Score derivation, and figure generation.
* **`India_CVD_Simulation_Phase1.py`**: Phase 1 analysis pipeline (patient generation + 7 continuous calculator implementations).
* **`India_CVD_Simulation_Phase2.py`**: Phase 2 analysis pipeline (SCORE2-DM merge, treatment mapping, discordance calculation, and Triage Score derivation).
* **`India_CVD_Triage_Score_Calculator.html`**: Standalone interactive web calculator for the derived Triage Score (HTML/CSS/JavaScript, zero dependencies).
* **`requirements.txt`**: Python environment dependencies.

## 🚀 How to Reproduce

1. **Clone this repository** to your local machine.
2. **Install dependencies:** `pip install -r requirements.txt`.
3. **Execute:** Run the consolidated analysis (`python ascvd_simulation.py`), or execute the staged pipeline (`Phase1` then `Phase2`) to replicate the exact ESC SCORE2-Diabetes batch workflow.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
