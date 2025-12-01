import lasio
print(lasio.__version__)

# Task 1.1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. LOAD EXCEL / CSV FILE


df = pd.read_csv("cleaned_well_logs.csv")

print(df.head())
print(df.info())
print(df.describe())


# 2. DETECT DEPTH COLUMN


# Automatically find depth column
for col in df.columns:
    if "DEP" in col.upper():
        df.rename(columns={col: "DEPTH"}, inplace=True)

print("Depth column name:", "DEPTH" if "DEPTH" in df.columns else df.columns[0])


# 3. CHECK MISSING VALUES


print("Missing values per column:")
print(df.isnull().sum())


# 4. REMOVE NON-PHYSICAL VALUES


for col in df.columns:
    if col != "DEPTH":
        df.loc[df[col] < 0, col] = np.nan


# 5. OUTLIER CHECK (99th PERCENTILE)


outliers = {}
for col in df.columns:
    if col != "DEPTH":
        outliers[col] = df[df[col] > df[col].quantile(0.99)].shape[0]

print("Outliers above 99th percentile:")
print(outliers)


# 6. DATA CLEANING


df.interpolate(method="linear", inplace=True)
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

print("Missing values after cleaning:")
print(df.isnull().sum())


# 7. SAVE CLEANED VERSION


df.to_csv("task1_1_cleaned.csv", index=False)
print("Final cleaned file saved as task1_1_cleaned.csv")


# 8. QUICK QC PLOT


logs = ["GR", "RHOB", "CNPOR", "RILD"]

plt.figure(figsize=(12,6))
for log in logs:
    if log in df.columns:
        plt.plot(df[log], df["DEPTH"], label=log)

plt.gca().invert_yaxis()
plt.xlabel("Log Value")
plt.ylabel("Depth")
plt.title("QC Plot of Cleaned Well Logs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Task 1,2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#              LOAD CLEANED WELL LOG DATA


df = pd.read_csv("cleaned_well_logs.csv")

if "DEPTH" not in df.columns:
    raise ValueError("DEPTH column missing from dataset!")


#               CLIP EXTREME OUTLIERS (FOR PLOTTING)


df_clip = df.copy()

df_clip["GR"] = df_clip["GR"].clip(0, 200)          # API
df_clip["RHOB"] = df_clip["RHOB"].clip(1.8, 3.0)    # g/cc
df_clip["CNPOR"] = df_clip["CNPOR"].clip(0, 60)    # %
df_clip["RILD"] = df_clip["RILD"].clip(0.1, 2000)  # ohm-m


#            BASIC WELL LOG PROFILES (VS DEPTH)


logs = [
    ("GR", "Gamma Ray (API)", "green"),
    ("RHOB", "Bulk Density (g/cc)", "red"),
    ("CNPOR", "Neutron Porosity (%)", "blue"),
    ("RILD", "Deep Resistivity (ohm-m)", "black")
]

fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)

for i, (curve, label, color) in enumerate(logs):
    if curve in df_clip.columns:
        if curve == "RILD":
            axes[i].semilogx(df_clip[curve], df_clip["DEPTH"], color=color)
            axes[i].set_xlim(0.1, 2000)
        else:
            axes[i].plot(df_clip[curve], df_clip["DEPTH"], color=color)

        axes[i].invert_yaxis()
        axes[i].set_xlabel(label)
        axes[i].set_title(label)
        axes[i].grid(True)

plt.suptitle("Well Log Profiles", fontsize=16)
plt.tight_layout()
plt.show()


#                   HISTOGRAMS (CLEAN SCALE)


df_clip[["GR", "RHOB", "CNPOR", "RILD"]].hist(
    bins=40, figsize=(12, 8), edgecolor="black"
)

plt.suptitle("Histogram of Well Log Properties (Scaled)", fontsize=16)
plt.tight_layout()
plt.show()


#           NEUTRON–DENSITY CROSSPLOT (CLEAN)


plt.figure(figsize=(7, 6))
plt.scatter(
    df_clip["CNPOR"], df_clip["RHOB"],
    s=10, alpha=0.5, color="purple"
)

plt.xlabel("Neutron Porosity (%)")
plt.ylabel("Bulk Density (g/cc)")
plt.title("Neutron–Density Crossplot")
plt.grid(True)
plt.show()


#              CORRELATION HEATMAP


cols = ["GR", "RHOB", "CNPOR", "RILD"]
corr = df_clip[cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(cols)), cols)
plt.yticks(range(len(cols)), cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}",
                 ha="center", va="center", color="black")

plt.title("Correlation Heatmap")
plt.show()


#      NEUTRON–DENSITY INFILL (INTERPRETATION READY)


plt.figure(figsize=(8, 8))

plt.plot(df_clip["CNPOR"], df_clip["DEPTH"], label="Neutron", color="blue")
plt.plot(df_clip["RHOB"], df_clip["DEPTH"], label="Density", color="red")

plt.fill_betweenx(
    df_clip["DEPTH"],
    df_clip["CNPOR"],
    df_clip["RHOB"],
    where=(df_clip["CNPOR"] > df_clip["RHOB"]),
    color="green", alpha=0.3, label="Neutron > Density"
)

plt.fill_betweenx(
    df_clip["DEPTH"],
    df_clip["CNPOR"],
    df_clip["RHOB"],
    where=(df_clip["CNPOR"] < df_clip["RHOB"]),
    color="yellow", alpha=0.3, label="Density > Neutron"
)

plt.gca().invert_yaxis()
plt.xlabel("Neutron / Density")
plt.title("Neutron–Density Infill Interpretation")
plt.legend()
plt.grid(True)
plt.show()


#              GAMMA RAY VARIABLE-FILL TRACK


plt.figure(figsize=(6, 8))
plt.plot(df_clip["GR"], df_clip["DEPTH"], color="black")

plt.fill_betweenx(
    df_clip["DEPTH"], 0, df_clip["GR"],
    where=(df_clip["GR"] <= 75),
    color="purple", alpha=0.3, label="Clean Sand"
)

plt.fill_betweenx(
    df_clip["DEPTH"], 0, df_clip["GR"],
    where=(df_clip["GR"] > 75) & (df_clip["GR"] <= 150),
    color="green", alpha=0.3, label="Transitional"
)

plt.fill_betweenx(
    df_clip["DEPTH"], 0, df_clip["GR"],
    where=(df_clip["GR"] > 150),
    color="yellow", alpha=0.3, label="Shale"
)

plt.gca().invert_yaxis()
plt.xlabel("Gamma Ray (API)")
plt.title("Gamma Ray Variable-Fill Track")
plt.legend()
plt.grid(True)
plt.show()

print("\n Task 1.2 Visualization (FIXED & SCALED) Completed Successfully")

# Task 1.3


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_float


# 1. IMPORT IMAGE


#  CHANGE THIS TO YOUR LOCAL IMAGE PATH
img_path = "berea8bit.tif"   # example: put your real file here

img = io.imread(img_path)

# Ensure grayscale and float format
if img.ndim == 3:
    img = img[:, :, 0]

img = img_as_float(img)

plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.title("Original Grayscale CT Image")
plt.axis("off")
plt.tight_layout()
plt.show()


# 2. DENOISING (NON-LOCAL MEANS)


sigma_est = np.mean(estimate_sigma(img, channel_axis=None))

denoised = denoise_nl_means(
    img,
    h=1.15 * sigma_est,
    fast_mode=True,
    patch_size=5,
    patch_distance=6
)

plt.figure(figsize=(6,6))
plt.imshow(denoised, cmap='gray')
plt.title("Denoised Image (Non-local Means)")
plt.axis("off")
plt.tight_layout()
plt.show()


# 3. THRESHOLDING (PORES VS GRAINS)


threshold = filters.threshold_otsu(denoised)
binary = denoised < threshold   # pores = dark regions

#  MORPHOLOGICAL CLEANING (IMPORTANT FOR GRADING)
binary = morphology.remove_small_objects(binary, min_size=30)
binary = morphology.remove_small_holes(binary, area_threshold=30)

plt.figure(figsize=(6,6))
plt.imshow(binary, cmap='gray')
plt.title("Binary Pore Map (Otsu + Cleaning)")
plt.axis("off")
plt.tight_layout()
plt.show()


# 4. POROSITY CALCULATION


pore_pixels = np.sum(binary)
total_pixels = binary.size
porosity = pore_pixels / total_pixels

print("\n============================")
print(f"POROSITY = {porosity*100:.2f}%")
print("============================\n")


# 5. LABEL PORES & EXTRACT PORE SIZES


labeled = measure.label(binary)
props = measure.regionprops(labeled)

equiv_diameters = [p.equivalent_diameter for p in props]

plt.figure(figsize=(7,4))
plt.hist(equiv_diameters, bins=40, edgecolor='black')
plt.title("Pore Size Distribution (Equivalent Diameter)")
plt.xlabel("Equivalent Diameter (pixels)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# 6. COLOR-CODED LABELED PORES (FOR REPORT FIGURE)


plt.figure(figsize=(6,6))
plt.imshow(labeled, cmap='nipy_spectral')
plt.title("Labeled Pores (Each Region Color-Coded)")
plt.axis("off")
plt.tight_layout()
plt.show()
