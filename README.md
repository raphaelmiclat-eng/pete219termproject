# README.md
"""
# PETE 219 Term Project
### Well Log Analysis • Poro-Perm Analysis • X-ray CT Image Analysis

This repository contains the complete workflow for **Tasks 1–3** of the PETE 219 Group Project. The project integrates:

- Well log processing and interpretation
- Core porosity–permeability analysis
- Machine learning classification and regression
- X-ray CT image segmentation and pore-scale property estimation

All tasks were implemented entirely in **Python**.

---

## Software & Libraries Used
- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-image
- scikit-learn
- seaborn (optional)

---

# Task 1 — Well Log Analysis

## 1.1 – Data Import & Cleaning
**Objective:** Prepare the raw well log data for analysis.

**Steps**
- Imported the CSV file
- Verified presence of DEPTH
- Computed descriptive statistics
- Identified missing values
- Detected extreme outliers (percentile method)
- Exported cleaned dataset

**Output**
- cleaned_well_logs.csv
- Summary statistics and missing-value report

---

## 1.2 – Well Log Visualization
**Objective:** Create standard petrophysical plots and interpret reservoir quality.

**Plots**
- Log tracks: GR, RHOB, CNPOR, RILD (log scale)
- Histograms for each log
- Neutron–Density crossplot
- Correlation heatmap
- GR variable-fill track

**Interpretation Summary**
- GR identifies sand–shale alternations
- Neutron–Density logs confirm porosity patterns
- Resistivity highlights potential hydrocarbon zones
- Correlation matrix
