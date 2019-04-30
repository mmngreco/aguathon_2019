"""CNN model."""
import logging
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
from uuid import uuid4
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data(fname):
    """load_data to global scope"""
    global data, data_fixed, all_levels, river
    fname = Path("./datos.csv")
    assert fname.exists()
    data = pd.read_csv(fname, index_col=0)
    data.index = data.index.astype("datetime64[ns]")
    data.sort_index(ascending=True, inplace=True)

    data_fixed = data.groupby(lambda x: x.weekofyear)\
                     .transform(lambda x: x.fillna(x.mean()))

    all_levels = data_fixed.iloc[:, :6].values.astype("float32")




def main():
    load_data()

    if PDB:
        __import__('pdb').set_trace()

    train_yhat = model.predict(train_X)
    score_train = mean_squared_error(train_y, train_yhat)

    test_yhat = model.predict(test_X)
    score_test = mean_squared_error(test_y, test_yhat)

    log.info(f"SCORE_TRAIN={score_train}")
    log.info(f"SCORE_TEST={score_test}")

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = "CNN-%s-%s-%s-%.4f-%.4f" % (LOOK_AHEAD, UUID, date, score_train, score_test)
    log.info(f"fname={fname}")

    plot_pred(train_y, train_yhat, f"train-{fname}")
    plot_pred(train_y[-2000:], train_yhat[-2000:], f"train-{fname}-ZOOM_LAST")

    plot_pred(test_y, test_yhat, f"test-{fname}")
    plot_pred(test_y[-2000:], test_yhat[-2000:], f"test-{fname}-ZOOM_LAST")

    model.save("models/%s.h5" % fname)


def load_model(name=None, path="models/"):
    if name is None:
        name = ""
    dir = Path(path)
    files = dir.glob(f"*{name}.h5")
    out = defaultdict(list)
    for f in files:
        fname_list = f.stem.split("-")
        out["loss"].append(fname_list[-2])
        out["val_loss"].append(fname_list[-1])
        out["fname"].append(str(f))
    return out


if __name__ == "__main__":
    # constants
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # parameters
    UUID = "-".join([utils.get_git_revision_short_hash(), str(uuid4())])
    SPLIT = args.split
    BATCH_SIZE = args.batch_size
    # N_STEPS = args.look_back
    EPOCHS = args.epochs
    X_FREQ = 1
    LOOK_AHEAD = args.look_ahead
    FILTERS = args.filters
    KERNEL_SIZE = args.kernel_size
    PATIENCE = args.patience
    L1L2 = eval(args.l1l2)
    CELLS_D1 = args.neurons1
    CELLS_D2 = args.neurons2
    NEURONS_OUT = 1
    PDB = bool(args.pdb)

    fmt = ('%(name)s | %(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    handlers = [sh]

    if args.log_path:
        fh = logging.FileHandler(f"{args.log_path}/{UUID}.log")
        handlers.append(fh)

    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

    try:
        log = logging.getLogger(__file__)
    except:
        log = logging.getLogger()

    log.info("COMMAND=python %s" % " ".join(sys.argv))
    main()

