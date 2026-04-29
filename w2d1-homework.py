import numpy as np

def manual_dot(a: np.ndarray, b: np.ndarray) -> float:
    # Convert to numpy arrays (in case lists are passed)
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Check that shapes match
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape")
    
    # Element-wise multiply, then sum
    return float(np.sum(a * b))


# ===============================
# ✅ Tests
# ===============================
tests = [
    (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
    (np.array([0.5, 1.5, 2.5, 3.5]), np.array([1.0, 2.0, 3.0, 4.0]))
]

print("=== Manual Dot Product (Loop) ===\n")

for a, b in tests:
    manual = manual_dot(a, b)
    builtin = np.dot(a, b)

    print(f"a = {a}, b = {b}")
    print(f"manual_dot result:  {manual}")
    print(f"np.dot result:      {builtin}")
    print(f"Match: {np.isclose(manual, builtin)}\n")

# ==============================================
# 2. Manual Matrix Multiplication (No @ or np.matmul)
# ==============================================
import numpy as np

def manual_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # Validate dimensions
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices")

    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Cannot multiply {A.shape} × {B.shape} — "
            f"inner dimensions {A.shape[1]} ≠ {B.shape[0]}"
        )

    m, n = A.shape
    _, p = B.shape

    result = np.zeros((m, p))

    # Manual multiplication
    for i in range(m):
        for j in range(p):
            result[i, j] = np.sum(A[i, :] * B[:, j])

    return result


# ===============================
# ✅ Test 1 (Valid multiplication)
# ===============================
print("=== Manual Matrix Multiplication ===")

A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

manual = manual_matmul(A, B)
builtin = A @ B

print("A (2×3) @ B (3×2):")
print("manual_matmul result:")
print(manual)
print("np.matmul result:")
print(builtin)
print(f"Match: {np.allclose(manual, builtin)}\n")


# ===============================
# ❌ Test 2 (Shape mismatch)
# ===============================
print("Testing shape mismatch (2×3 @ 2×3):")

A_bad = np.array([[1, 2, 3],
                  [4, 5, 6]])

B_bad = np.array([[1, 2, 3],
                  [4, 5, 6]])

try:
    manual_matmul(A_bad, B_bad)
except ValueError as e:
    print(f"ValueError caught: {e}")


#===============================
import numpy as np

print("=== Batch Predictions ===")

# Dataset
X = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1800, 3, 8],
    [2200, 5, 2],
    [1200, 2, 15]
])

# Weights and bias
w = np.array([150, 20000, -500])
b = 50000

# Predictions
predictions = X @ w + b

print("House features: [size_sqft, bedrooms, age_years]")
print("Weights: size=$150/sqft, bedrooms=$20000, age=-$500")
print("Bias: $50000\n")

# Print each sample with prediction
for i, (features, pred) in enumerate(zip(X, predictions), start=1):
    print(f"Sample {i}: {features} → Predicted price: ${pred:,.0f}")


#==============================================

import numpy as np

print("=== Cosine Similarity: Word Vectors ===")

# Define word embeddings (3D vectors)
king = np.array([0.9, 0.3, 0.1])
queen = np.array([0.8, 0.4, 0.2])
apple = np.array([0.1, 0.1, 0.9])

# Cosine similarity function
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

# Compute similarities
sim_king_queen = cos_sim(king, queen)
sim_king_apple = cos_sim(king, apple)
sim_queen_apple = cos_sim(queen, apple)

# Print results
print(f"cos_sim(king, queen) = {sim_king_queen:.4f}  ← Most similar")
print(f"cos_sim(king, apple) = {sim_king_apple:.4f}")
print(f"cos_sim(queen, apple) = {sim_queen_apple:.4f}")

# Explanation:
# "king" and "queen" are most similar because their vectors point in nearly
# the same direction in this 3D space (very small angle between them).
# "apple" is different (fruit vs royalty), so its vector points in a different
# direction, giving lower cosine similarity scores.