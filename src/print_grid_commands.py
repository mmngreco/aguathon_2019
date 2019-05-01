import numpy as np
from sklearn.model_selection import ParameterGrid


def grid_search():
    seed = 7
    np.random.seed(seed)
    params = {
        "-b": [*range(200, 8000, 200)],
        "-f": [*range(60, 300, 8)],
        "--l1l2": ["'l2(0)'", "'l2(0.001)'", "'l2(0.005)'"],
        "-n1": [*range(6, 2048, 64)],
        "-n2": [0],
        "-e": [40],
        "-p": [3, 6, 9, 12],
        "-la": [72]
     }

    params_collection = list(ParameterGrid(params))
    np.random.shuffle(params_collection)
    N = 100

    for params in params_collection[:N]:
        args = " ".join([f"{k} {v}"for k, v in params.items()])
        print(f"\"python CNN_feat2d.py {args}\"")


if __name__ == "__main__":
    # constants
    grid_search()

