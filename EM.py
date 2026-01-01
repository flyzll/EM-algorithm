import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed()

# ===============================
# データ生成関数
# ===============================
def generate_object(center, cov, n_points):
    return np.random.multivariate_normal(center, cov, n_points)

def gaussian(x, mu, Sigma):
    diff = x - mu
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    norm = (
        1.0
        / np.sqrt((2 * np.pi) ** dimension * det)
        * np.exp(-0.5 * np.sum(diff @ inv * diff, axis=1))
    )
    return norm

true_params = [
    {
        "mu": np.array([2, 2, 2]),
        "Sigma": np.array([[0.3, 0.1, 0.0],
                           [0.1, 0.3, 0.0],
                           [0.0, 0.0, 0.3]])
    },
    {
        "mu": np.array([2, 10, 3]),
        "Sigma": np.array([[0.3, 0.0, 0.0],
                           [0.0, 0.3, 0.0],
                           [0.0, 0.0, 0.3]])
    },
    {
        "mu": np.array([10, 14, 15]),
        "Sigma": np.array([[0.3, 0.0, 0.0],
                           [0.0, 0.3, 0.0],
                           [0.0, 0.0, 0.3]])
    }
]

points = np.vstack([
    generate_object(true_params[0]["mu"], true_params[0]["Sigma"], 100),
    generate_object(true_params[1]["mu"], true_params[1]["Sigma"], 200),
    generate_object(true_params[2]["mu"], true_params[2]["Sigma"], 100)
])

N, dimension = points.shape

K = 3
mins = points.min(axis=0)
maxs = points.max(axis=0)

mu = np.random.uniform(mins, maxs, size=(K, dimension))
Sigma = np.array([np.eye(dimension) for _ in range(K)])
pi = np.ones(K) / K


if __name__ == "__main__":

    for iteration in range(100):
        # E-step
        gamma = np.zeros((N, K))
        for k in range(K):
            gamma[:, k] = pi[k] * gaussian(points, mu[k], Sigma[k])

        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1e-12
        gamma /= gamma_sum

        # M-step 
        Nk = gamma.sum(axis=0)
        print(Nk)
        for k in range(K):
            if Nk[k] < 1e-6:
                continue

            mu[k] = np.sum(gamma[:, k, None] * points, axis=0) / Nk[k]

            diff = points - mu[k]
            Sigma[k] = (
                gamma[:, k, None, None]
                * np.einsum("ni,nj->nij", diff, diff)
            ).sum(axis=0) / Nk[k]

            pi[k] = Nk[k] / N

        pi /= pi.sum()


    print(" TRUE DATA PARAMETERS ")
    for i, p in enumerate(true_params):
        print(f"\n[True Cluster {i}]")
        print("Mean (mu):")
        print(p["mu"])
        print("Covariance (Sigma):")
        print(p["Sigma"])


    print(" ESTIMATED PARAMETERS (EM) ")
    for k in range(K):
        print(f"\n[Estimated Cluster {k}]")
        print("Mean (mu):")
        print(mu[k])
        print("Covariance (Sigma):")
        print(Sigma[k])
  

    gamma = np.zeros((N, K))
    for k in range(K):
        gamma[:, k] = pi[k] * gaussian(points, mu[k], Sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)

    labels = np.argmax(gamma, axis=1)

    # プロット
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for k in range(K):
        cluster = points[labels == k]
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=10)
        ax.scatter(mu[k][0], mu[k][1], mu[k][2],
                   c="black", marker="x", s=100)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D EM algorithm")
    plt.show()
