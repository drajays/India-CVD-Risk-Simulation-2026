#India-CVD Risk Simulation & Triage Score
This repository contains the full computational framework for the manuscript: "Treatment Decision Discordance Across Eight ASCVD Risk Calculators in 100,000 Synthetic Indian Patients: An Archetype-Based Treatment-Consequence Analysis for Statin and Cardiometabolic Drug Selection".

#Zenodo Data Repository (Download Data Here)
Due to GitHub file size limits, the large 100,000-patient synthetic datasets are hosted permanently on Zenodo. You must download the data file from Zenodo to run the analysis scripts.

DOI Link: https://doi.org/10.5281/zenodo.18949324

FINAL_MANUSCRIPT_DATA.csv.zip: The locked 100,000-patient synthetic dataset containing all baseline parameters, 8 continuous risk scores, and 7 binary treatment classifications. Anchored to ICMR-INDIAB-17 metrics.

indian_ascvd_100k_v3_base.csv.zip: The base synthetic patient cohort generated prior to the SCORE2-Diabetes web calculator merge.

#GitHub Repository Contents
Core Analysis Code & Scripts
ascvd_simulation.py: The complete Python analysis pipeline (patient generation, calculator implementations, and Triage Score derivation).

generate_main_manuscript_tables.py: Script to computationally reproduce the main manuscript statistics, tables, and figure data.

generate_supplementary_1.py: Script to computationally reproduce Supplementary Tables ST1–ST8 and S3 sensitivity analyses.

generate_supplementary_2.py: Script to computationally reproduce the Triage Score validation metrics and S5 budget impact extrapolations.

Clinical Implementation Tools
india_cvd_triage_score_calculator.html: Standalone interactive web calculator for the derived Triage Score (HTML/CSS/JavaScript, zero dependencies).

Intermediate ESC SCORE2 Data
SCORE2_Diabetes_Batch_Upload_v3.csv: The formatted input file used for the ESC SCORE2-Diabetes web calculator.

score2_diabetes_results.csv: Raw output from the ESC SCORE2-Diabetes web calculator (41,633 diabetic patients, 0 errors).

Environment
requirements.txt: Python environment dependencies (e.g., pandas, numpy, scikit-learn, rpy2).

#How to Reproduce the Study
Clone this repository to your local machine.

Install dependencies: Run pip install -r requirements.txt.

Download the Data: Download FINAL_MANUSCRIPT_DATA.csv.zip from the Zenodo link above and place it in the root directory of this cloned repository.

Execute the reproduction scripts: * Run python generate_main_manuscript_tables.py to recreate the core findings.

Run python generate_supplementary_1.py and python generate_supplementary_2.py to recreate the supplementary reports.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
