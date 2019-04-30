import numpy as np
from sklearn.model_selection import ParameterGrid


def grid_search():
    seed = 7
    np.random.seed(seed)
    params = {
        "-b": [900],
        "-f": [*range(70, 200, 8)],
        "--l1l2": ["'l2(0)'", "'l2(0.01)'", "'l2(0.02)'"],
        "-n1": [250, 488],
        "-n2": [*range(0, 32, 2)],
        "-e": [40],
        "-p": [6],
        "-la": [72]
     }

    params_collection = list(ParameterGrid(params))
    np.random.shuffle(params_collection)

    for params in params_collection:
        args = " ".join([f"{k} {v}"for k, v in params.items()])
        print(f"\"python CNN_feat2d.py {args}\"")


if __name__ == "__main__":
    # constants
    grid_search()

