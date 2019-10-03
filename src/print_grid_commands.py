"""Utility to make a grid search using command line.

This script builds a combination of parameters given allowed values. The
combination is randomized but limited by a number of realization defined by the
user.

"""
import numpy as np

from sklearn.model_selection import ParameterGrid, ParameterSampler


def main(params, n):
    seed = 7
    np.random.seed(seed)
    params_collection = ParameterSampler(params, n_iter=n, random_state=0)

    for params in params_collection:
        args = " ".join([f"{k} {v}"for k, v in params.items()])
        print(f"\"python train_model.py {args}\"")


if __name__ == "__main__":
    import sys
    try:
        n = sys.argv[1]
    except IndexError:
        n = 100  # Default value

    params = {
        "-b": range(200, 8000, 128),
        "-f": range(60, 300, 4),
        "--l1l2": ["'l2(0)'", "'l2(0.001)'", "'l2(0.005)'", "'l1(0.001)'",
                   "'l1(0.005)'", "'l1(0.01)'"],
        "-n1": range(6, 2048, 32),
        "-n2": range(6, 512, 32),
        "-e": range(10, 100, 20),
        "-p": [3, 6, 9, 12],
        "-lb": range(1, 300, 5),
        "-la": [24],
     }

    main(params=params, n=n)

