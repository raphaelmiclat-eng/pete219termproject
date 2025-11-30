PETE 219 Term Project
Well Log Analysis • Poro-Perm Analysis • X-ray CT Image Analysis

This repository contains the complete workflow for Tasks 1–3 of the PETE 219 Group Project. The project integrates:

Well log processing and interpretation

Core porosity–permeability analysis

Machine learning classification and regression

X-ray CT image segmentation and pore-scale property estimation

All tasks were implemented entirely in Python.

Software & Libraries Used

Python 3.x

numpy

pandas

matplotlib

scikit-image

scikit-learn

seaborn (optional)

Task 1 — Well Log Analysis
1.1 – Data Import & Cleaning

Objective: Prepare the raw well log data for analysis.

Steps Performed

Imported the CSV file

Verified presence of DEPTH

Computed descriptive statistics

Identified missing values

Detected extreme outliers (percentile method)

Exported the cleaned dataset

Output

cleaned_well_logs.csv

Summary statistics and missing-value report

1.2 – Well Log Visualization

Objective: Create standard petrophysical plots and interpret reservoir quality.

Plots Generated

Log tracks: GR, RHOB, CNPOR, RILD (log scale)

Histograms for each log

Neutron–Density crossplot

Correlation heatmap

GR variable-fill track

Interpretation Summary

GR identifies sand–shale alternations

Neutron–Density logs confirm porosity trends

Resistivity highlights potential hydrocarbon zones

Correlation matrix supports log consistency

1.3 – Initial CT-Based Porosity Estimation

Objective: Estimate porosity using a raw CT slice.

Workflow

Imported grayscale CT

Applied Non-Local Means denoising

Performed Otsu threshold segmentation

Generated binary pore map and labeled pore regions

Calculated porosity and pore-size distribution

Outputs

Binary pore image

Labeled pore regions

Pore size histogram

Porosity estimate

Task 2 — Core Poro-Perm Analysis & Machine Learning
2.1 – Data Cleaning

Objective: Remove invalid and unrealistic measurements.

Steps

Loaded core dataset

Removed negative/invalid values

Removed missing values

Removed outliers

Summarized cleaned data

Output

task2_1_cleaned_poroperm_FINAL.csv

2.2 – Exploratory Data Analysis

Visualizations

Porosity vs Permeability by facies

Porosity & Permeability distributions

Boxplots by facies

Depth trends

P–P probability plots

Key Insights

Channel facies show highest porosity & permeability

Overbank facies have lowest reservoir quality

Porosity generally decreases with depth

Permeability strongly facies-dependent

2.3 – Machine Learning Models

Models Implemented

Linear Regression

K-Means Clustering

Random Forest

Support Vector Machine (SVM)

Artificial Neural Network (ANN)

Outputs

R² values

Silhouette scores

Confusion matrices

Precision, Recall, F1-scores

Findings

Random Forest produced the highest classification accuracy

Linear regression shows moderate poro-perm correlation

ANN limited by dataset size

Task 3 — X-ray CT Image Analysis
3.1 – Preprocessing

Steps

Imported CT image

Applied denoising

Normalized & contrast-enhanced

Smoothed for segmentation

Output

task3_1_preprocessed_ct.png

3.2 – Segmentation & Visualization

Steps

Plotted grayscale CT

Generated pixel-intensity histogram

Performed Otsu segmentation

Produced binary pore map

Labeled connected pore regions

Created pore-size heatmap

Outputs

task3_2_binary_pores.png

task3_2_labeled_pores.png

Pore size heatmap

3.3 – Porosity & Permeability Computation

Computed

Porosity from pore-pixel ratio

Pore-size statistics

Permeability via Kozeny–Carman equation

Key Results

Porosity: ≈ 17.7%

Mean pore diameter: ≈ 16.5 px

Permeability: ≈ 1.24 × 10⁻¹² m² (≈ 1250 mD)

Output

task3_3_ct_results_summary.txt

Integrated Interpretation

The project combines three independent datasets:

Task 1 — Well Logs

→ Identify lithology, porosity variation, and fluid zones

Task 2 — Core + ML

→ Predict reservoir quality using facies and statistical relationships

Task 3 — CT Imaging

→ Direct pore-scale quantification of porosity & permeability

All methods agree on the following:

Channel facies exhibit higher porosity and permeability

Overbank facies show poorer reservoir quality

Trends are consistent across logs, cores, CT images, and ML models

How to Run the Project

Place all CSV files, images, and scripts in the same directory.

Run tasks sequentially:

Task 1.1 → Task 1.2 → Task 1.3  
Task 2.1 → Task 2.2 → Task 2.3  
Task 3.1 → Task 3.2 → Task 3.3


Figures will display automatically, and all output text files will save in the working folder.

Authors — PETE 219 Group Project

Railee Fernandez

Raphael Miclat

Khalifa Sadiq

Ayaan Shaikh

Abdulrahman El Edrisi
