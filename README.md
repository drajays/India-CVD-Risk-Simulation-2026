# India-CVD Risk Simulation & Triage Score

This repository contains the full computational framework and synthetic dataset for the manuscript: 
**"Treatment Decision Discordance Across Eight ASCVD Risk Calculators in 100,000 Synthetic Indian Patients: An Archetype-Based Treatment-Consequence Analysis for Statin and Cardiometabolic Drug Selection"**

## 📂 Repository Contents

### Core Analysis
- `ascvd_simulation.py`: Executable Python script for the consolidated analytical pipeline.
- `ascvd_simulation.ipynb`: Interactive Jupyter/Colab Notebook containing the complete analysis pipeline, Triage Score derivation, and figure generation.
- `requirements.txt`: Python environment dependencies.

### Staged Pipeline & Extended Files (Full Reproduction)
- `India_CVD_Simulation_Phase1.py`: Phase 1 analysis pipeline (patient generation + 7 continuous calculator implementations).
- `India_CVD_Simulation_Phase2.py`: Phase 2 analysis pipeline (SCORE2-DM merge, treatment mapping, discordance calculation, and Triage Score derivation).
- `SCORE2_Diabetes_Batch_Upload.csv`: The formatted input file used for the ESC SCORE2-Diabetes web calculator.
- `SCORE2_Diabetes_Results.xlsx`: Raw output from the ESC SCORE2-Diabetes web calculator (41,633 patients, 0 errors).

### Datasets & Clinical Tools
- `FINAL_MANUSCRIPT_DATA.csv.zip` / `India_CVD_Simulation_Data.csv`: The locked 100,000-patient synthetic dataset containing all baseline parameters, 8 continuous risk scores, and 7 binary treatment classifications. Anchored to ICMR-INDIAB-17 metrics.
- `India_CVD_Triage_Score_Calculator.html`: Standalone interactive web calculator for the derived Triage Score (HTML/CSS/JavaScript, zero dependencies). Can be opened directly in any web browser.

## 🚀 How to Reproduce
1. Clone this repository.
2. Ensure you have the dependencies installed: `pip install -r requirements.txt`.
3. Unzip the dataset into the main directory.
4. Run the consolidated analysis (`python ascvd_simulation.py`), or execute the staged pipeline (`Phase1` then `Phase2`) to replicate the exact ESC SCORE2-Diabetes batch workflow.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
