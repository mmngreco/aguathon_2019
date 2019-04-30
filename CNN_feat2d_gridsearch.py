import numpy as np
from sklearn.model_selection import ParameterGrid


def grid_search():
    seed = 7
    np.random.seed(seed)
    params = {
        "-b": [800, 1600, 2400, 4800],
        "-f": [8, 16, 32],
        "--l1l2": ["'l2(0)'", "'l2(0.01)'", "'l2(0.02)'"],
        "-n1": [8, 16, 32],
        "-n2": [0, 2, 8, 16],
        "-e": [10, 20, 30],
        "-p": [10],
        "-la": [24]
     }

    params_collection = list(ParameterGrid(params))
    np.random.shuffle(params_collection)

    for params in params_collection:
        args = " ".join([f"{k} {v}"for k, v in params.items()])
        print(f"\"python CNN_feat2d.py {args}\"")


if __name__ == "__main__":
    # constants
    grid_search()

