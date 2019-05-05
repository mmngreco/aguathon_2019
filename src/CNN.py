"""CNN model."""
import os
import sys
import logging

from init import *
from uuid import uuid4
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        TensorBoard)
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import l1, l2

import utils
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if sys.platform == "darwin":   # to make it work on macos
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_test_split(dataset, train_frac):
    train_size = int(len(dataset)*train_frac)
    return dataset[:train_size, :], dataset[train_size:, :]


def split_sequence_time_steps(sequence, look_back, x_freq, y_ahead=0,
                              norm=True, return_scaler=False):
    """Returns X and y variables from a sequence.

    Parameters
    ----------
    sequence : array-like
        1d sequence which it will be used to split in X and y.
    look_back : int
        Number of periods you want to have for each row.
    x_freq : int
        Decimation of x over look_back axis.
    y_ahead : int
        Distance ahead to align with each row of x.
    norm : bool
        If standarize X or not.
    return_scaler : bool
        If norm is True, you may want retrieve the scaler used.

    Returns
    -------
    X, y : array-like
    scaler : Scaler, optional
    """
    X, y = list(), list()
    _y_ahead = y_ahead

    last_idx = len(sequence) - _y_ahead - 1
    sequence_x = sequence

    if norm:
        scaler = StandardScaler()
        sequence_x = scaler.fit_transform(sequence_x)

    for i in range(0, last_idx):
        end_ix = i + look_back  # find the end of this pattern
        if end_ix > last_idx:  # check if we are beyond the sequence
            break

        seq_x = sequence_x[i:end_ix:x_freq]
        X.append(seq_x)

        seq_y = sequence[end_ix + y_ahead, 0]
        y.append(seq_y)

    out_x = np.array(X)
    out_y = np.array(y)[:, None]

    if norm and return_scaler:
        return out_x, out_y, scaler

    return out_x, out_y


def load_data(fname, fill_na=True, y_data=False):
    """load_data.

    Parameters
    ----------
    fname : pathlib.Path
        Path where data will be loaded.
    fill_na : bool, optional
        Pass true if you want fill nans values.
    y_data : bool, optional
        If y data (target) is returned or not.

    Returns
    -------
    x : pandas.DataFrame
    y : pandas.DataFrame, optional
    """
    assert fname.exists(), "File doesn't exist."
    data = pd.read_csv(fname, index_col=0)
    data.index = data.index.astype("datetime64[ns]")
    data = data.sort_index(ascending=True)
    x = data.iloc[:, :6]

    if fill_na:
        x = (x.groupby(lambda x: x.weekofyear)
              .transform(lambda x: x.fillna(x.mean())))
    if y_data:
        y = data.iloc[:, 7:]
        return x, y
    return x


def build_model(n_steps, n_features, filters, kernel_size, L1L2, D1, D2, DOUT, initializer=None):
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

    pd.DataFrame(y-yhat, columns=["yhat%s" % LOOK_AHEAD]).plot(figsize=(15, 10))
    plt.title("diff-%s" % name)
    plt.tight_layout()
    plt.savefig(f"{output_dir / name}-diff.png")


def split_sequence(x, x_target=0, look_ahead=1, norm_x=True, return_scaler=False):
    """Given a sequence returns a X, y data.

    Parameters
    ----------
    x : 2d-array-like
    x_target : int
        If x has more than one column, you need specify which columns will be
        used as y.
    look_ahead : int
        Horizon prediction to pick each value of x as y.
    norm_x : bool
        Pass True If x will be standarized.
    return_scaler : bool
        Pass True if want return scaler used to standarize x.

    Returns
    -------
    x : array-like
    y : array-like
    scaler : StandrScaler, optional

    """
    y = x
    if norm_x:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    out_x = x[:-look_ahead]
    out_y = y[look_ahead:, x_target][:, None].copy()

    if norm_x and return_scaler:
        return out_x, out_y, scaler

    return out_x, out_y


def main():
    data = load_data(IN_FILE).values
    send_log()

    target_idx = 5
    train, test = train_test_split(data, SPLIT)
    train_X, train_y = split_sequence(train, target_idx, LOOK_AHEAD)
    test_X, test_y = split_sequence(test, target_idx, LOOK_AHEAD)

    test_X = test_X[:, :, None]
    train_X = train_X[:, :, None]
    # test_y = test_y[:, None, :]
    # train_y = train_y[:, None, :]

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
    doc = "test/train split"
    parser.add_argument("-s", "--split", type=float, default=0.8, help=doc)
    doc = "Batch size."
    parser.add_argument("-b", "--batch-size", type=int, default=512, help=doc)
    doc = "Kernel Size."
    parser.add_argument("-k", "--kernel-size", type=int, default=6, help=doc)
    doc = "Number of epochs."
    parser.add_argument("-e", "--epochs", type=int, default=230, help=doc)
    doc = "Number of filters."
    parser.add_argument("-f", "--filters", type=int, default=8, help=doc)
    doc = "Look back, no used here."
    parser.add_argument("-lb", "--look-back", type=int, default=0, help=doc)
    doc = "Look ahead."
    parser.add_argument("-la", "--look-ahead", type=int, default=1, help=doc)
    doc = "Set patience training param."
    parser.add_argument("-p", "--patience", type=int, default=20, help=doc)
    doc = "Number of neurons in the first Dense layer."
    parser.add_argument("-n1", "--neurons1", type=int, default=48, help=doc)
    doc = "Number of neurons in the sencond Dense layer."
    parser.add_argument("-n2", "--neurons2", type=int, default=None, help=doc)
    doc = "Define a regularizer like 'l1(0.1)' or 'l2(0.1)'."
    parser.add_argument("-l", "--l1l2", type=str, default="None", help=doc)
    parser.add_argument("--pdb", action="count")
    parser.add_argument("--log-path", type=str, default="")
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--shuffle", type=bool, default=False)
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Path

    # -------------------------------------------------------------------------
    # process some variables
    if not args.neurons2:
        args.neurons2 = None

    if not len(args.log_path):
        args.log_path = LOG_DIR
    else:
        args.log_path = Path(args.log_path)
    # -------------------------------------------------------------------------

    def send_log():
        """Useful function to send arguments parameters passed to logging."""
        log.info(f"UUID={UUID}")
        log.info(f"SPLIT={SPLIT}")
        log.info(f"BATCH_SIZE={BATCH_SIZE}")
        log.info(f"EPOCHS={EPOCHS}")
        log.info(f"PATIENCE={PATIENCE}")
        log.info(f"X_FREQ={X_FREQ}")
        log.info(f"LOOK_AHEAD={LOOK_AHEAD}")
        log.info(f"FILTERS={FILTERS}")
        log.info(f"KERNEL_SIZE={KERNEL_SIZE}")
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

