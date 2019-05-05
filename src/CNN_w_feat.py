"""CNN model."""
import os
import sys
import logging

from uuid import uuid4
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import l1, l2

import utils
import matplotlib.pyplot as plt

from init import *
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if sys.platform == "darwin":   # to make it work on macos
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def build_model(n_steps, n_features, filters, kernel_size, L1L2, D1, D2, DOUT,
                initializer=None):
    """build_model.

    Parameters
    ----------
    n_steps : int
    n_features : int
    filters : int
    kernel_size : int
    L1L2 : function
    D1 : int
    D2 : int
    DOUT : int
    initializer : initializer

    Returns
    -------
    model : tf.keras.model
    """

    model = Sequential()
    model.add(Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        input_shape=(n_steps, n_features),
        kernel_initializer=initializer,
    ))
    model.add(Dense(
        D1,
        activation='relu',
        kernel_regularizer=L1L2,
        kernel_initializer=initializer,
    ))
    if D2 is not None:
        model.add(Dense(
            D2,
            activation='relu',
            kernel_regularizer=L1L2,
            kernel_initializer=initializer,
        ))
    model.add(Flatten())
    model.add(Dense(DOUT))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def plot_pred(y, yhat, name, output_dir):
    """Plot prediction against target.

    Parameters
    ----------
    y : numpy.array
        Real vector.
    yhat : numpy.array
        Predicted vector.
    name : str
        Name of the figure to be stored.
    output_dir : pathlib.Path
        Where the plots will be stored.

    Returns
    -------
    If name and output_dir are provided the plot are saved in
    output_dir/name.png.
    """
    ax = pd.DataFrame(y, columns=["y%s" % LOOK_AHEAD]).plot(figsize=(15, 10))
    pd.DataFrame(yhat, columns=["yhat%s" % LOOK_AHEAD]).plot(ax=ax)
    plt.title("%s" % name)
    plt.tight_layout()
    plt.savefig(f"{output_dir / name}.png")

    pd.DataFrame(y-yhat, columns=[f"yhat {LOOK_AHEAD}"]).plot(figsize=(15, 10))
    plt.title("diff-%s" % name)
    plt.tight_layout()
    plt.savefig(f"{output_dir / name}-diff.png")


def main():
    data = utils.load_data(IN_FILE)
    data["i"] = np.arange(data.shape[0])
    send_log()

    if PDB:
        __import__('pdb').set_trace()

    target_idx = 5
    train, test = train_test_split(data.values, train_size=SPLIT, shuffle=False)
    train_X, train_y = utils.split_sequence_time_steps(
        train, LOOK_BACK, LOOK_AHEAD, target_idx
        )
    test_X, test_y = utils.split_sequence_time_steps(
        test, LOOK_BACK, LOOK_AHEAD, target_idx
        )

    train_y = train_y.squeeze(-1)
    test_y = test_y.squeeze(-1)

    if PDB:
        __import__('pdb').set_trace()

    TIME_STEPS = train_X.shape[1]
    N_FEATURES = train_X.shape[2]

    initializer = glorot_normal(7)
    model = build_model(
        TIME_STEPS,
        N_FEATURES,
        FILTERS,
        KERNEL_SIZE,
        L1L2,
        D1,
        D2,
        DOUT,
        initializer
    )

    history = model.fit(
        x=train_X,
        y=train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        # validation_split=1-SPLIT,
        shuffle=SHUFFLE,
        validation_data=(test_X, test_y),
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=PATIENCE,
                verbose=0,
                mode='auto',
            )]
    )

    if PDB:
        __import__('pdb').set_trace()

    train_yhat = model.predict(train_X)
    score_train = mean_squared_error(train_y, train_yhat)

    test_yhat = model.predict(test_X)
    score_test = mean_squared_error(test_y, test_yhat)

    log.info(f"SCORE_TRAIN={score_train}")
    log.info(f"SCORE_TEST={score_test}")

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"CNN-{LOOK_AHEAD}-{UUID}-{date}-{score_test:.4f}"
    log.info(f"fname={fname}")

    if PLOT:
        plot_pred(train_y, train_yhat, f"train-{fname}")
        plot_pred(train_y[-2000:], train_yhat[-2000:], f"train-{fname}-LAST")

        plot_pred(test_y, test_yhat, f"test-{fname}")
        plot_pred(test_y[-2000:], test_yhat[-2000:], f"test-{fname}-LAST")

    model.save(f"{MODELS_DIR / fname}.h5")


def parse_model_files(name, path):
    files = path.glob(f"{name}.h5")
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
    # -------------------------------------------------------------------------
    doc = "test/train fraction split"
    parser.add_argument("-s", "--split", type=float, default=0.8, help=doc)
    doc = "Batch size."
    parser.add_argument("-b", "--batch-size", type=int, default=512, help=doc)
    doc = "Set patience training param."
    parser.add_argument("-p", "--patience", type=int, default=20, help=doc)
    doc = "Number of epochs."
    parser.add_argument("-e", "--epochs", type=int, default=230, help=doc)
    parser.add_argument("--shuffle", type=bool, default=False)
    # -------------------------------------------------------------------------
    # Structural params
    doc = "Kernel Size. A kernel_size equalt to -1 implies equal to look_back."
    parser.add_argument("-k", "--kernel-size", type=int, default=-1, help=doc)
    doc = "Look back."
    parser.add_argument("-lb", "--look-back", type=int, default=10, help=doc)
    doc = "Look ahead."
    parser.add_argument("-la", "--look-ahead", type=int, default=24, help=doc)
    doc = "Number of filters."
    parser.add_argument("-f", "--filters", type=int, default=8, help=doc)
    doc = "Number of neurons in the first Dense layer."
    parser.add_argument("-n1", "--neurons1", type=int, default=48, help=doc)
    doc = "Number of neurons in the sencond Dense layer."
    parser.add_argument("-n2", "--neurons2", type=int, default=None, help=doc)
    doc = "Define a regularizer like 'l1(0.1)' or 'l2(0.1)'."
    parser.add_argument("-l", "--l1l2", type=str, default="None", help=doc)
    # -------------------------------------------------------------------------
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--pdb", action="count")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # process some variables
    if not args.neurons2:
        args.neurons2 = None

    if not len(args.log_path):
        args.log_path = LOG_DIR
    elif args.log_path == "None":
        args.log_path = False
    else:
        args.log_path = Path(args.log_path)

    if args.kernel_size == -1:
        args.kernel_size = args.look_back
    # -------------------------------------------------------------------------

    def send_log():
        """Useful function to send arguments parameters passed to logging."""
        log.info(f"UUID={UUID}")
        log.info(f"SPLIT={SPLIT}")
        log.info(f"BATCH_SIZE={BATCH_SIZE}")
        log.info(f"EPOCHS={EPOCHS}")
        log.info(f"PATIENCE={PATIENCE}")
        log.info(f"X_FREQ={X_FREQ}")
        log.info(f"LOOK_BACK={LOOK_BACK}")
        log.info(f"LOOK_AHEAD={LOOK_AHEAD}")
        log.info(f"KERNEL_SIZE={KERNEL_SIZE}")
        log.info(f"FILTERS={FILTERS}")
        log.info(f"L1L2={L1L2}")
        log.info(f"D1={D1}")
        log.info(f"D2={D2}")
        log.info(f"DOUT={DOUT}")
        log.info(f"PLOT={PLOT}")
        log.info(f"SHUFFLE={SHUFFLE}")

    # -------------------------------------------------------------------------
    # Parameters
    UUID = f"{utils.get_git_revision_short_hash()}-{uuid4()}"
    SPLIT = args.split
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    X_FREQ = 1
    LOOK_BACK = args.look_back
    LOOK_AHEAD = args.look_ahead
    FILTERS = args.filters
    KERNEL_SIZE = args.kernel_size
    PATIENCE = args.patience
    L1L2 = eval(args.l1l2)
    D1 = args.neurons1
    D2 = args.neurons2
    DOUT = 1
    PDB = bool(args.pdb)
    PLOT = bool(args.plot)
    SHUFFLE = bool(args.shuffle)

    # -------------------------------------------------------------------------
    # logging configuration
    fmt = ('%(name)s | %(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    handlers = [sh]

    if args.log_path:
        fh = logging.FileHandler(f"{args.log_path / UUID}.log")
        handlers.append(fh)

    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

    try:
        log = logging.getLogger(__file__)
    except NameError:
        log = logging.getLogger()

    log.info("COMMAND=python %s" % " ".join(sys.argv))
    # -------------------------------------------------------------------------
    main()

