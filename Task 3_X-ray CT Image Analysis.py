#Task 3.1

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, exposure
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_float


# 1. IMPORT CT IMAGE


#  CHANGE THIS TO YOUR LOCAL IMAGE PATH
img_path = "berea8bit.tif"    # Example file
img = io.imread(img_path)


# 2. ENSURE GRAYSCALE & FLOAT FORMAT


if img.ndim == 3:                 # If RGB image
    img = img[:, :, 0]            # Convert to grayscale

img = img_as_float(img)           # Convert to float for processing

plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
plt.title("Original CT Image (Grayscale)")
plt.axis("off")
plt.tight_layout()
plt.show()


# 3. NOISE REDUCTION (NON-LOCAL MEANS FILTER)


sigma_est = np.mean(estimate_sigma(img, channel_axis=None))

img_denoised = denoise_nl_means(
    img,
    h=1.15 * sigma_est,
    fast_mode=True,
    patch_size=5,
    patch_distance=6
)

plt.figure(figsize=(6,6))
plt.imshow(img_denoised, cmap="gray")
plt.title("Denoised CT Image (Non-local Means)")
plt.axis("off")
plt.tight_layout()
plt.show()


# 4. INTENSITY NORMALIZATION (CONTRAST STRETCHING)


p2, p98 = np.percentile(img_denoised, (2, 98))
img_normalized = exposure.rescale_intensity(
    img_denoised,
    in_range=(p2, p98)
)

plt.figure(figsize=(6,6))
plt.imshow(img_normalized, cmap="gray")
plt.title("Normalized CT Image (Contrast Enhanced)")
plt.axis("off")
plt.tight_layout()
plt.show()


# 5. EDGE & SMALL NOISE CLEANUP (MEDIAN + MORPHOLOGY)


img_smooth = filters.median(img_normalized, morphology.disk(2))

plt.figure(figsize=(6,6))
plt.imshow(img_smooth, cmap="gray")
plt.title("Smoothed CT Image (Final Preprocessed)")
plt.axis("off")
plt.tight_layout()
plt.show()


# 6. SAVE PREPROCESSED IMAGE FOR TASK 3.2


io.imsave("task3_1_preprocessed_ct.png", img_smooth)

print("\n Task 3.1 Completed Successfully")
print("Preprocessed image saved as: task3_1_preprocessed_ct.png")

#Task 3.2




import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
from skimage.util import img_as_float


# 0. LOAD PREPROCESSED IMAGE FROM TASK 3.1 (RECOMMENDED)


#  Use your preprocessed image from Task 3.1
img_path = "task3_1_preprocessed_ct.png"
img = io.imread(img_path, as_gray=True)

img = img_as_float(img)


# 1. DISPLAY PREPROCESSED GRAYSCALE IMAGE


plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray")
plt.title("Preprocessed CT Image (Grayscale)")
plt.axis("off")
plt.tight_layout()
plt.show()


# 2. PIXEL INTENSITY HISTOGRAM


plt.figure(figsize=(6, 4))
plt.hist(img.ravel(), bins=256, edgecolor="black")
plt.title("Pixel Intensity Histogram")
plt.xlabel("Intensity (0 = black, 1 = white)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# 3. BINARY SEGMENTATION (OTSU THRESHOLD)


thresh = filters.threshold_otsu(img)
binary = img < thresh   # pores = dark regions

#  Morphological cleanup for grading quality
binary = morphology.remove_small_objects(binary, min_size=40)
binary = morphology.remove_small_holes(binary, area_threshold=40)

plt.figure(figsize=(6, 6))
plt.imshow(binary, cmap="gray")
plt.title(f"Binary Pore Map (Otsu Threshold = {thresh:.3f})")
plt.axis("off")
plt.tight_layout()
plt.show()


# 4. LABELED PORE REGIONS (CONNECTED COMPONENTS)


labeled = measure.label(binary, connectivity=2)

plt.figure(figsize=(6, 6))
plt.imshow(labeled, cmap="nipy_spectral")
plt.title("Labeled Pores (Separated Regions)")
plt.axis("off")
plt.tight_layout()
plt.show()


# 5. PORE SIZE HEATMAP (EQUIVALENT DIAMETER)


props = measure.regionprops(labeled)

pore_size_map = np.zeros_like(img, dtype=float)

for region in props:
    d_eq = region.equivalent_diameter
    coords = region.coords
    pore_size_map[coords[:, 0], coords[:, 1]] = d_eq

plt.figure(figsize=(6, 6))
im = plt.imshow(pore_size_map, cmap="inferno")
plt.title("Pore Size Heatmap (Equivalent Diameter in Pixels)")
plt.axis("off")
cbar = plt.colorbar(im)
cbar.set_label("Equivalent Diameter (pixels)")
plt.tight_layout()
plt.show()


# 6. SAVE SEGMENTED OUTPUTS FOR TASK 3.3


io.imsave("task3_2_binary_pores.png", binary.astype(np.uint8) * 255)
io.imsave("task3_2_labeled_pores.png", labeled.astype(np.uint16))

print("\n Task 3.2 Completed Successfully")
print("Binary image saved as: task3_2_binary_pores.png")
print("Labeled pore image saved as: task3_2_labeled_pores.png")

#Task 3.3


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from skimage.util import img_as_float


# 0. LOAD BINARY & LABELED IMAGES FROM TASK 3.2


binary_path = "task3_2_binary_pores.png"
labeled_path = "task3_2_labeled_pores.png"

binary_img = io.imread(binary_path, as_gray=True)
labeled = io.imread(labeled_path)

# make sure binary is 0/1 (pores = 1)
binary = img_as_float(binary_img) > 0.5


# 1. IMAGE-BASED POROSITY


pore_pixels = np.sum(binary)
total_pixels = binary.size
phi = pore_pixels / total_pixels

print("\n============================")
print(f"IMAGE-BASED POROSITY = {phi*100:.2f}%")
print("============================\n")


# 2. PORE SIZE STATISTICS (EQUIVALENT DIAMETER)


props = measure.regionprops(labeled)

equiv_diameters = np.array([p.equivalent_diameter for p in props])

print(f"Number of pores detected = {len(equiv_diameters)}")
print(f"Mean pore diameter (pixels)   = {equiv_diameters.mean():.2f}")
print(f"Median pore diameter (pixels) = {np.median(equiv_diameters):.2f}")
print(f"Std dev of pore diameter      = {equiv_diameters.std():.2f}")

# Pore size histogram
plt.figure(figsize=(7,4))
plt.hist(equiv_diameters, bins=40, edgecolor='black')
plt.title("Pore Size Distribution (Equivalent Diameter)")
plt.xlabel("Equivalent Diameter (pixels)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. KOZENY–CARMAN PERMEABILITY ESTIMATE
#    k = (d^2 * phi^3) / [180 * (1 - phi)^2]

# NOTE: You MUST set the pixel size in meters based on the CT scanner.
# Example: if 1 pixel = 10 micrometers, then:
# pixel_size_m = 10e-6

pixel_size_m = 10e-6   # <-- UPDATE THIS TO YOUR REAL PIXEL SIZE IF KNOWN

# use mean pore diameter as characteristic length scale
d_mean_pixels = equiv_diameters.mean()
d_mean_m = d_mean_pixels * pixel_size_m

k_m2 = (d_mean_m**2 * phi**3) / (180.0 * (1.0 - phi)**2)

print("Kozeny–Carman Permeability Estimate:")
print(f"Mean pore diameter  = {d_mean_pixels:.2f} pixels")
print(f"Pixel size (assumed) = {pixel_size_m:.2e} m/pixel")
print(f"Characteristic d     = {d_mean_m:.2e} m")
print(f"Estimated k          = {k_m2:.3e} m^2")

# Optional: convert to milliDarcy (1 mD ≈ 9.869e-16 m^2)
k_mD = k_m2 / 9.869e-16
print(f"Estimated k          = {k_mD:.1f} mD")


# 4. SAVE SUMMARY TO TEXT FILE (OPTIONAL BUT NICE)


with open("task3_3_ct_results_summary.txt", "w") as f:
    f.write("PETE 219 – Task 3.3 CT Quantitative Results\n")
    f.write("===========================================\n")
    f.write(f"Porosity (image-based): {phi*100:.2f}%\n")
    f.write(f"Number of pores: {len(equiv_diameters)}\n")
    f.write(f"Mean pore diameter (pixels): {d_mean_pixels:.2f}\n")
    f.write(f"Median pore diameter (pixels): {np.median(equiv_diameters):.2f}\n")
    f.write(f"Std dev pore diameter (pixels): {equiv_diameters.std():.2f}\n")
    f.write(f"\nAssumed pixel size: {pixel_size_m:.2e} m/pixel\n")
    f.write(f"Characteristic diameter d: {d_mean_m:.2e} m\n")
    f.write(f"Kozeny–Carman k: {k_m2:.3e} m^2\n")
    f.write(f"Kozeny–Carman k: {k_mD:.1f} mD\n")

print("\n Task 3.3 Completed Successfully")
print("Summary written to: task3_3_ct_results_summary.txt")
