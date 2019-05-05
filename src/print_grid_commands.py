import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from numpy.random import choice


def grid_search():
    seed = 7
    np.random.seed(seed)
    params = {
        "-b": range(200, 8000, 128),
        "-f": range(60, 300, 4),
        "--l1l2": ["'l2(0)'", "'l2(0.001)'", "'l2(0.005)'", "'l1(0.001)'", "'l1(0.005)'", "'l1(0.01)'"],
        "-n1": range(6, 2048, 32),
        "-n2": range(6, 512, 32),
        "-e": range(10, 100, 20),
        "-p": [3, 6, 9, 12],
        "-lb": range(1, 300, 5),
        "-la": [24],
     }

    N = 100
    params_collection = ParameterSampler(params, n_iter=N, random_state=0)

    for params in params_collection:
        args = " ".join([f"{k} {v}"for k, v in params.items()])
        print(f"\"python CNN_feat2d.py {args}\"")


if __name__ == "__main__":
    # constants
    grid_search()

