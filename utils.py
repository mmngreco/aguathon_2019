import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import seaborn as sns


def recurrence_matrix(s, eps=0.10, steps=10):
    d = pdist(s[:, None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z


if __name__ == "__main__":
    N = 100

    plt.figure()
    __import__('pdb').set_trace()
    x = np.arange(N)
    plt.imshow(recurrence_matrix(x))
    plt.show()

    plt.figure()
    x = np.random.randn(N)
    plt.imshow(recurrence_matrix(x))
    plt.show()

    plt.figure()
    sns.heatmap(recurrence_matrix(x))
    plt.show()

    plt.figure()
    x = np.sin(np.arange(N))
    sns.heatmap(recurrence_matrix(x), cmap="viridis")
    plt.show()

