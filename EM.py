import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def generate_object(center, cov, n_points):
    return np.random.multivariate_normal(center, cov, n_points)

points = np.vstack([
    generate_object([2, 2], [[0.1, 0], [0, 0.1]], 150),
    generate_object([6, 3], [[0.2, 0.05], [0.05, 0.1]], 150),
    generate_object([4, 7], [[0.1, -0.03], [-0.03, 0.2]], 150)
])

N, D = points.shape
K = 3  # 物体数（既知と仮定）

# ----------------------------
# 2. GMM 初期化
# ----------------------------
mu = points[np.random.choice(N, K, replace=False)]
Sigma = np.array([np.eye(D) for _ in range(K)])
pi = np.ones(K) / K

# ----------------------------
# 3. ガウス分布（多次元）
# ----------------------------
def gaussian(x, mu, Sigma):
    d = x.shape[1]
    diff = x - mu
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    norm = 1.0 / np.sqrt((2 * np.pi)**d * det)
    return norm * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))

# ----------------------------
# 4. EMアルゴリズム
# ----------------------------
for _ in range(1):

    # === E-step ===
    gamma = np.zeros((N, K))
    for k in range(K):
        gamma[:, k] = pi[k] * gaussian(points, mu[k], Sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)

    # === M-step ===
    Nk = gamma.sum(axis=0)
    for k in range(K):
        mu[k] = np.sum(gamma[:, k, None] * points, axis=0) / Nk[k]
        diff = points - mu[k]
        Sigma[k] = (gamma[:, k, None, None] *
                    np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / Nk[k]
        pi[k] = Nk[k] / N

# ----------------------------
# 5. 結果の可視化
# ----------------------------
labels = np.argmax(gamma, axis=1)

plt.figure(figsize=(6, 6))
for k in range(K):
    cluster = points[labels == k]
    plt.scatter(cluster[:, 0], cluster[:, 1], s=10, label=f"Object {k+1}")
    plt.scatter(mu[k][0], mu[k][1], c='black', marker='x', s=100)

plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Object Detection using EM (GMM)")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()
