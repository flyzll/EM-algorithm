import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed()

def generate_object(center, cov, n_points):
    return np.random.multivariate_normal(center, cov, n_points)

def gaussian(x, mu, Sigma):
    diff = x - mu
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    norm = 1.0 / np.sqrt((2 * np.pi)**dimention * det)* np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))
    return norm 

points = np.vstack([
    generate_object([2, 2, 2],
                    [[0.3, 0.1, 0.0],
                     [0.1, 0.3, 0.0],
                     [0.0, 0.0, 0.3]], 300),

    generate_object([2, 10, 3],
                    [[0.3, 0.0, 0.0],
                     [0.0, 0.3, 0.0],
                     [0.0, 0.0, 0.3]], 300),

    generate_object([10, 14, 15],
                    [[0.3, 0.0, 0.0],
                     [0.0, 0.3, 0.0],
                     [0.0, 0.0, 0.3]], 300)
])

N, dimention = points.shape

K = 100 #初期クラスタ数

mu = points[np.random.choice(N, K, replace=True)] #平均値
Sigma = np.array([np.eye(dimention) for _ in range(K)]) #共分散
pi = np.ones(K) / K  # 混合係数

tol = 1e-3         # クラスタ中心の変化の閾値
stable_limit = 3    #　3回連続で閾値以下なら収束とみなす
stable_count = 0    # 連続カウンタ
mu_prev = None


for iteration in range(1000):
    # E-step
    gamma = np.zeros((N, K))
    for k in range(K):
        gamma[:, k] = pi[k] * gaussian(points, mu[k], Sigma[k])

    gamma_sum = gamma.sum(axis=1, keepdims=True)
    gamma_sum[gamma_sum == 0] = 1e-12
    gamma /= gamma_sum

    # M-step
    Nk = gamma.sum(axis=0)

    for k in range(K):
        if Nk[k] < 1e-6:
            continue

        mu[k] = np.sum(gamma[:, k, None] * points, axis=0) / Nk[k]

        diff = points - mu[k]
        Sigma[k] = (
            gamma[:, k, None, None]
            * np.einsum('ni,nj->nij', diff, diff)
        ).sum(axis=0) / Nk[k]

        pi[k] = Nk[k] / N

    threshold = 0.01   # 全体の1%未満
    valid = pi > threshold #クラスタが有効かどうかの判定

    mu = mu[valid]
    Sigma = Sigma[valid]
    pi = pi[valid]

    pi /= pi.sum()
    K = len(pi)

    # 収束判定 
    if mu_prev is not None and mu.shape == mu_prev.shape:
        diff = np.linalg.norm(mu - mu_prev)

        if diff < tol:
            stable_count += 1
            
        else:
            stable_count = 0

        if stable_count >= stable_limit:
            print(f"iter {iteration}: K = {K}")
            print("Stop EM.")
            break

    mu_prev = mu.copy()


gamma = np.zeros((N, K))
for k in range(K):
    gamma[:, k] = pi[k] * gaussian(points, mu[k], Sigma[k])
gamma /= gamma.sum(axis=1, keepdims=True)

labels = np.argmax(gamma, axis=1)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

for k in range(K):
    cluster = points[labels == k]
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=10)

    ax.scatter(mu[k][0], mu[k][1], mu[k][2],
               c='black', marker='x', s=100)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"3D EM with Over-clustering (Final K = {K})")
plt.show()
