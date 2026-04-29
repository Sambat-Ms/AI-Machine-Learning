# ===============================
# 1. Paradigm Decision Function
# ===============================
def choose_paradigm(has_labels: bool, goal: str) -> str:
    if has_labels:
        if goal == "predict_category":
            return "Classification"
        elif goal == "predict_number":
            return "Regression"
    else:
        if goal == "discover_groups":
            return "Clustering"
        elif goal == "compress_data":
            return "Dimensionality Reduction"
    return "Invalid input"


print("=== Paradigm Decision ===")
cases = [
    (True, "predict_category"),
    (True, "predict_number"),
    (False, "discover_groups"),
    (False, "compress_data"),
]

for has_labels, goal in cases:
    result = choose_paradigm(has_labels, goal)
    print(f"has_labels={has_labels},  goal='{goal}' → {result}")


# ===============================
# 2. Supervised Learning (KNN)
# ===============================
print("\n=== Supervised: KNN on Digits Dataset ===")

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict
y_pred = knn.predict(X_test_scaled)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f} ({acc*100:.1f}%)\n")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Find lowest precision class
report_dict = classification_report(y_test, y_pred, output_dict=True)
precisions = {int(k): v["precision"] for k, v in report_dict.items() if k.isdigit()}
lowest_digit = min(precisions, key=precisions.get)

print(f"Lowest precision digit: {lowest_digit} ({precisions[lowest_digit]:.2f})")

# Comment
# Digit 8 is often confused with 3 and 9 due to similar closed-loop shapes


# ===============================
# 3. Unsupervised Learning (KMeans)
# ===============================
print("\n=== Unsupervised: KMeans on Digits (No Labels) ===")

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Cluster sizes
cluster_sizes = np.bincount(clusters)
print("Cluster sizes:", cluster_sizes.tolist())

# Largest cluster
largest_cluster = np.argmax(cluster_sizes)
print(f"Largest cluster: Cluster {largest_cluster} ({cluster_sizes[largest_cluster]} samples)")

# ===============================
# 4. Save cluster visualization
# ===============================
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(kmeans.cluster_centers_[i].reshape(8, 8), cmap="gray")
    ax.set_title(f"Cluster {i}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("digit_clusters.png")
print("[Saved digit_clusters.png]")