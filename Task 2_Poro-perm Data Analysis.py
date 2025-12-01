#Task 2.1
import pandas as pd
import numpy as np


# 1. LOAD PORO-PERM DATA


df_pp = pd.read_csv("poro_perm_data.csv")

print("\nFirst rows:")
print(df_pp.head())

print("\nDataset Info:")
print(df_pp.info())

print("\nStatistics:")
print(df_pp.describe(include='all'))


# 2. MISSING VALUES CHECK


print("\nMissing values per column:")
print(df_pp.isnull().sum())


# 3. REMOVE NON-PHYSICAL VALUES
# Porosity >= 0
# Permeability >= 0


df_pp.loc[df_pp["Porosity (%)"] < 0, "Porosity (%)"] = np.nan
df_pp.loc[df_pp["Permeability (mD)"] < 0, "Permeability (mD)"] = np.nan


# 4. OUTLIER DETECTION (1%–99%)


outlier_report = {}

for col in ["Depth (ft)", "Porosity (%)", "Permeability (mD)"]:
    q1 = df_pp[col].quantile(0.01)
    q99 = df_pp[col].quantile(0.99)
    outliers = df_pp[(df_pp[col] < q1) | (df_pp[col] > q99)].shape[0]
    outlier_report[col] = outliers

print("\nOutliers detected (outside 1%-99% range):")
print(outlier_report)


# 5. REMOVE EXTREME OUTLIERS


for col in ["Depth (ft)", "Porosity (%)", "Permeability (mD)"]:
    q1 = df_pp[col].quantile(0.01)
    q99 = df_pp[col].quantile(0.99)
    df_pp.loc[df_pp[col] < q1, col] = np.nan
    df_pp.loc[df_pp[col] > q99, col] = np.nan

print("\nAfter removing extreme outliers:")
print(df_pp.isnull().sum())


# 6. CLEAN FILL STRATEGY
# Interpolation + Median fill


df_pp.interpolate(method="linear", inplace=True)

df_pp["Facies"].fillna(method="ffill", inplace=True)

for col in ["Depth (ft)", "Porosity (%)", "Permeability (mD)"]:
    df_pp[col].fillna(df_pp[col].median(), inplace=True)


# 7. ENFORCE PHYSICAL LIMITS (FINAL SAFETY FIX)


df_pp.loc[df_pp["Porosity (%)"] < 0, "Porosity (%)"] = 0
df_pp.loc[df_pp["Porosity (%)"] > 45, "Porosity (%)"] = 45

df_pp.loc[df_pp["Permeability (mD)"] < 0, "Permeability (mD)"] = 0


# 8. FINAL QC


print("\nFinal missing values:")
print(df_pp.isnull().sum())

print("\nFinal cleaned dataset statistics:")
print(df_pp.describe())


# 9. SAVE FINAL CLEAN FILE


df_pp.to_csv("task2_1_cleaned_poroperm_FINAL.csv", index=False)
print("\n Final cleaned file saved as task2_1_cleaned_poroperm_FINAL.csv")

#Task 2.2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


# 1. LOAD CLEANED DATASET


df = pd.read_csv("task2_1_cleaned_poroperm_FINAL.csv")

print(df.head())
print(df.info())

# Clean facies labels (remove quotes)
df["Facies"] = df["Facies"].str.replace("'", "").str.strip()


# 2. POROSITY vs PERMEABILITY CROSSPLOT (FACIES-COLORED)


plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Porosity (%)",
    y="Permeability (mD)",
    hue="Facies",
    style="Facies",
    s=80
)

plt.yscale("log")   # permeability usually plotted in log scale
plt.xlabel("Porosity (%)")
plt.ylabel("Permeability (mD)")
plt.title("Porosity vs Permeability by Facies")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# 3. HISTOGRAMS BY FACIES


facies_list = df["Facies"].unique()

plt.figure(figsize=(12,5))

# Porosity histogram
plt.subplot(1,2,1)
for fac in facies_list:
    plt.hist(
        df[df["Facies"] == fac]["Porosity (%)"],
        bins=20,
        alpha=0.6,
        label=fac
    )
plt.xlabel("Porosity (%)")
plt.ylabel("Frequency")
plt.title("Porosity Distribution by Facies")
plt.legend()
plt.grid(True)

# Permeability histogram
plt.subplot(1,2,2)
for fac in facies_list:
    plt.hist(
        df[df["Facies"] == fac]["Permeability (mD)"],
        bins=20,
        alpha=0.6,
        label=fac
    )
plt.xlabel("Permeability (mD)")
plt.ylabel("Frequency")
plt.title("Permeability Distribution by Facies")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# 4. BOX PLOTS (FACIES COMPARISON)


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(data=df, x="Facies", y="Porosity (%)")
plt.title("Porosity by Facies")
plt.grid(True)

plt.subplot(1,2,2)
sns.boxplot(data=df, x="Facies", y="Permeability (mD)")
plt.yscale("log")
plt.title("Permeability by Facies")
plt.grid(True)

plt.tight_layout()
plt.show()


# 5. P–P (PROBABILITY–PROBABILITY) PLOTS


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
stats.probplot(df["Porosity (%)"], dist="norm", plot=plt)
plt.title("P–P Plot: Porosity")

plt.subplot(1,2,2)
stats.probplot(df["Permeability (mD)"], dist="norm", plot=plt)
plt.title("P–P Plot: Permeability")

plt.tight_layout()
plt.show()


# 6. DEPTH vs POROSITY & PERMEABILITY


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(df["Porosity (%)"], df["Depth (ft)"], c="blue", s=40)
plt.gca().invert_yaxis()
plt.xlabel("Porosity (%)")
plt.ylabel("Depth (ft)")
plt.title("Porosity vs Depth")
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(df["Permeability (mD)"], df["Depth (ft)"], c="red", s=40)
plt.gca().invert_yaxis()
plt.xscale("log")
plt.xlabel("Permeability (mD)")
plt.ylabel("Depth (ft)")
plt.title("Permeability vs Depth")
plt.grid(True)

plt.tight_layout()
plt.show()

#Task 2.3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, silhouette_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 1. LOAD CLEANED DATA


df = pd.read_csv("task2_1_cleaned_poroperm_FINAL.csv")

# Clean facies text
df["Facies"] = df["Facies"].str.replace("'", "").str.strip()

print(df.head())
print(df.info())


# 2. SUPERVISED LEARNING: LINEAR REGRESSION
# Predict permeability from porosity


X = df[["Porosity (%)"]].values
y = df["Permeability (mD)"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

r2 = r2_score(y_test, y_pred)

print("\nLINEAR REGRESSION RESULTS")
print("R² score:", r2)
print("Slope:", reg.coef_[0])
print("Intercept:", reg.intercept_)

# Regression plot
plt.figure(figsize=(7,5))
plt.scatter(X_test, y_test, label="Actual", alpha=0.7)
plt.scatter(X_test, y_pred, label="Predicted", alpha=0.7)
plt.xlabel("Porosity (%)")
plt.ylabel("Permeability (mD)")
plt.title("Linear Regression: Permeability vs Porosity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. UNSUPERVISED LEARNING: K-MEANS CLUSTERING


X_cluster = df[["Porosity (%)", "Permeability (mD)"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

sil_score = silhouette_score(X_scaled, clusters)
print("\nK-MEANS CLUSTERING")
print("Silhouette Score:", sil_score)

# Plot clusters
plt.figure(figsize=(7,5))
sns.scatterplot(
    x=df["Porosity (%)"],
    y=df["Permeability (mD)"],
    hue=df["Cluster"],
    palette="Set1",
    s=80
)
plt.yscale("log")
plt.xlabel("Porosity (%)")
plt.ylabel("Permeability (mD)")
plt.title("K-Means Clustering of Poro-Perm Data")
plt.grid(True)
plt.tight_layout()
plt.show()


# 4. FACIES CLASSIFICATION (SUPERVISED)
# Features → Porosity & Permeability
# Target → Facies


X_class = df[["Porosity (%)", "Permeability (mD)"]]
y_class = df["Facies"]

# Encode facies labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_class)

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_encoded, test_size=0.2, random_state=42
)


# 4A. RANDOM FOREST CLASSIFIER


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRANDOM FOREST CLASSIFICATION RESULTS")
print(classification_report(y_test, y_pred_rf, target_names=encoder.classes_))

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()


# 4B. SUPPORT VECTOR MACHINE (SVM)


svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("\nSVM CLASSIFICATION RESULTS")
print(classification_report(y_test, y_pred_svm, target_names=encoder.classes_))

cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    cmap="Oranges"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()


# 4C. MULTI-LAYER PERCEPTRON (ANN)


mlp = MLPClassifier(
    hidden_layer_sizes=(50,50),
    max_iter=1000,
    random_state=42
)

mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)

print("\nANN (MLP) CLASSIFICATION RESULTS")
print(classification_report(y_test, y_pred_mlp, target_names=encoder.classes_))

cm_mlp = confusion_matrix(y_test, y_pred_mlp)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm_mlp,
    annot=True,
    fmt="d",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    cmap="Greens"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ANN Confusion Matrix")
plt.tight_layout()
plt.show()

