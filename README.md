# pete219termproject

Tasks 1–3: Well Log Analysis, Poro-Perm Analysis, and X-ray CT Image Analysis

This repository contains the complete workflow for Tasks 1–3 of the PETE 219 Group Project. The project integrates well log analysis, core porosity–permeability data analysis, machine learning, and X-ray CT image processing to characterize reservoir properties and estimate porosity and permeability using multiple independent approaches.

All tasks were implemented using Python with standard scientific libraries.

SOFTWARE AND LIBRARIES USED

Python 3.x

numpy

pandas

matplotlib

scikit-image

scikit-learn

seaborn (only for optional plots)

TASK 1: WELL LOG DATA ANALYSIS
Task 1.1 – Data Import and Cleaning
Objective

To load the raw well log dataset, identify missing values and outliers, and prepare a clean dataset for visualization and further analysis.

Steps Performed

Imported the well log CSV file

Verified that the DEPTH column exists

Computed summary statistics (mean, min, max, standard deviation)

Identified missing values in each log

Detected extreme outliers using percentile thresholds

Retained a cleaned dataset for plotting

Output

Cleaned well log dataset:

cleaned_well_logs.csv

Summary statistics and missing-value report printed to the console

Task 1.2 – Well Log Visualization
Objective

To visualize reservoir properties using standard well-log interpretation techniques.

Visualizations Created

Well Log Profiles vs Depth

Gamma Ray (GR)

Bulk Density (RHOB)

Neutron Porosity (CNPOR)

Deep Resistivity (RILD, log scale)

Histograms

Distribution of GR, RHOB, CNPOR, and RILD

Neutron–Density Crossplot

Used for lithology and porosity interpretation

Correlation Heatmap

Shows correlation between GR, RHOB, CNPOR, and RILD

Neutron–Density Infill Plot

Highlights gas-effect and density-dominated zones

Gamma Ray Variable-Fill Track

Clean sand, transitional, and shale zones identified by GR thresholds

Key Interpretation

Sand–shale alternations are clearly visible in GR

Inverse relationship between neutron porosity and density confirms reservoir porosity trends

Resistivity variations indicate potential hydrocarbon-bearing zones

Correlation matrix supports petrophysical consistency

Task 1.3 – Initial X-ray CT Image Porosity Analysis
Objective

To extract pore geometry and porosity directly from a CT image.

Steps Performed

Imported grayscale CT image

Applied Non-Local Means denoising

Used Otsu thresholding for pore–grain segmentation

Converted image to a binary pore map

Computed image-based porosity

Labeled connected pore regions

Generated pore size distribution (equivalent diameter)

Visualized labeled pores

Output

Binary pore image

Labeled pore image

Histogram of pore diameters

Image-based porosity estimate

TASK 2: CORE POROSITY–PERMEABILITY AND MACHINE LEARNING
Task 2.1 – Poro-Perm Data Cleaning
Objective

To clean core porosity–permeability measurements and remove physically unrealistic values.

Steps Performed

Loaded porosity–permeability dataset

Removed negative porosity and permeability values

Identified missing values

Removed extreme outliers using percentile limits

Generated cleaned statistical summary

Output

Final cleaned dataset:

task2_1_cleaned_poroperm_FINAL.csv

Cleaned statistical summary of depth, porosity, and permeability

Task 2.2 – Exploratory Data Analysis
Objective

To visualize relationships between porosity, permeability, facies, and depth.

Visualizations Created

Porosity vs Permeability by facies (scatter plot)

Porosity distribution by facies (histogram)

Permeability distribution by facies (histogram)

Porosity by facies (boxplot)

Permeability by facies (boxplot)

Porosity vs Depth

Permeability vs Depth

P-P plots for porosity and permeability

Key Interpretation

Channel facies show the highest permeability and porosity

Overbank facies show the lowest permeability

Porosity decreases with depth in general

Permeability is strongly influenced by facies type

Task 2.3 – Machine Learning and Regression
Objective

To predict permeability and classify facies using machine learning methods.

Models Used

Linear Regression (Permeability vs Porosity)

K-Means Clustering (unsupervised facies grouping)

Random Forest Classifier

Support Vector Machine (SVM)

Artificial Neural Network (ANN)

Key Outputs

Linear regression R² value and slope

K-means silhouette score

Confusion matrices for each classifier

Precision, recall, and F1-scores

Key Findings

Random Forest produced the highest classification accuracy

Linear regression showed a moderate positive porosity–permeability relationship

ANN performance was limited due to small dataset size

TASK 3: X-RAY CT IMAGE ANALYSIS
Task 3.1 – Data Import and Preprocessing
Objective

To prepare the CT image for quantitative analysis.

Steps Performed

Imported grayscale CT image

Denoised using Non-Local Means filtering

Normalized and contrast-enhanced the image

Smoothed the image for segmentation

Output

Preprocessed CT image saved as:

task3_1_preprocessed_ct.png

Task 3.2 – CT Image Visualization and Segmentation
Objective

To segment pores and visualize pore geometry.

Steps Performed

Displayed grayscale CT image

Generated pixel intensity histogram

Performed Otsu threshold segmentation

Created binary pore map

Labeled connected pore regions

Created pore-size heatmap using equivalent diameters

Output Files

task3_2_binary_pores.png

task3_2_labeled_pores.png

Pore size heatmap figure

Task 3.3 – Quantitative CT-Based Porosity and Permeability
Objective

To compute porosity and estimate permeability from the segmented CT image.

Calculations

Image-based porosity computed as pore pixel fraction

Pore size distribution extracted from labeled regions

Mean and median pore diameter calculated

Permeability estimated using the Kozeny–Carman equation

Pixel size assumed for physical scaling

Key Results

Image-based porosity ≈ 17.7 %

Mean pore diameter ≈ 16.5 pixels

Estimated permeability ≈ 1.24 × 10⁻¹² m² (≈ 1250 mD)

Output File

task3_3_ct_results_summary.txt

OVERALL PROJECT INTEGRATION

Tasks 1–3 provide three independent reservoir characterization methods:

Well logs (Task 1)
→ Lithology, porosity trends, and fluid indicators

Core poro–perm + machine learning (Task 2)
→ Facies-controlled reservoir quality and predictive modeling

X-ray CT imaging (Task 3)
→ Direct pore-scale porosity and permeability estimation

All three methods show consistent trends, with high porosity and permeability in channel-dominated zones and lower quality in overbank facies, validating the integrated workflow.

HOW TO RUN THE PROJECT

Place all CSV image and Python files in the same directory

Run scripts in the following order:

Task 1.1 → Task 1.2 → Task 1.3

Task 2.1 → Task 2.2 → Task 2.3

Task 3.1 → Task 3.2 → Task 3.3

All figures will display automatically

Output text files will be saved in the same folder

AUTHORS

PETE 219 Group Project

Railee Fernandez
Raphael Miclat
Khalifa Sadiq
Ayaan Shaikh
Abdulrahman El Edrisi
