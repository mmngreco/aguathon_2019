"""CNN model."""
import utils
import logging
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
from uuid import uuid4
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import l1, l2
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    # Embedding,
    Dense,
    Dropout,
    Conv1D,
    Conv2D,
    Flatten,
    Reshape,
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_test_split(dataset, train_frac):
    train_size = int(len(dataset)*train_frac)
    return dataset[:train_size, :], dataset[train_size:, :]


def split_sequence(sequence, look_back, x_freq, y_ahead=0, norm=True, return_scaler=False):
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
    # _y_ahead = max(y_ahead)
    _y_ahead = y_ahead

    last_idx = len(sequence) - _y_ahead - 1
    sequence_x = sequence

    if norm:
        scaler = StandardScaler()
        sequence_x = scaler.fit_transform(sequence_x)

    for i in range(0, last_idx):
        # find the end of this pattern
        end_ix = i + look_back
        # check if we are beyond the sequence
        if end_ix > last_idx:
            break
        # gather input and output parts of the pattern
        seq_x = sequence_x[i:end_ix:x_freq]
        X.append(seq_x)

        # each row
        # seq_y = [sequence[end_ix + y_ahead_i, 0] for y_ahead_i in y_ahead]
        seq_y = sequence[end_ix + y_ahead, 0]
        y.append(seq_y)

    out_x = array(X)
    # out_y = [yi[:, None] for yi in array(y).T]
    out_y = array(y)[:, None]

    if norm and return_scaler:
        return out_x, out_y, scaler

    return out_x, out_y


def load_data():
    """load_data to global scope"""
    global data, data_fixed, all_levels, river
    fname = "ENTRADA/datos.csv"
    data = pd.read_csv(fname, index_col=0)
    data.index = data.index.astype("datetime64[ns]")
    data.sort_index(ascending=True, inplace=True)

    data_fixed = data.groupby(lambda x: x.weekofyear)\
                     .transform(lambda x: x.fillna(x.mean()))

    all_levels = data_fixed.iloc[:, :6].values.astype("float32")
    river = all_levels[:, 5]  # zgz


# def build_model(n_steps, n_features, filters, kernel_size, L1L2, CELLS_D1, NEURONS_OUT):
#     """build_model"""
#     model = Sequential()
#     model.add(Conv1D(
#         filters=filters,
#         kernel_size=kernel_size,
#         activation='relu',
#         input_shape=(n_steps, n_features),
#     ))
#     model.add(Dense(
#         CELLS_D1,
#         activation='relu',
#         kernel_regularizer=L1L2,
#     ))
#     # model.add(Dense(
#     #     n_features,
#     #     activation='relu',
#     #     kernel_regularizer=L1L2,
#     # ))
#     model.add(Flatten())
#     model.add(Dense(NEURONS_OUT))
#     model.compile(optimizer='adam', loss='mse')
#     model.summary()
#     return model

def build_model2(n_steps, n_features, filters, kernel_size, L1L2, CELLS_D1, CELLS_D2, NEURONS_OUT, initializer=None):
    """build_model"""

    model = Sequential()
    model.add(Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        input_shape=(n_steps, n_features),
        kernel_initializer=initializer,
    ))
    model.add(Dense(
        CELLS_D1,
        activation='relu',
        kernel_regularizer=L1L2,
        kernel_initializer=initializer,
    ))
    if CELLS_D2 is not None:
        model.add(Dense(
            CELLS_D2,
            activation='relu',
            kernel_regularizer=L1L2,
            kernel_initializer=initializer,
        ))
    model.add(Flatten())
    model.add(Dense(NEURONS_OUT))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def plot_pred(y, yhat, name):
    ax = pd.DataFrame(y, columns=["y%s" % LOOK_AHEAD]).plot(figsize=(15, 10))
    pd.DataFrame(yhat, columns=["yhat%s" % LOOK_AHEAD]).plot(ax=ax)
    plt.title("%s" % name)
    plt.tight_layout()
    plt.savefig("figures/%s.png" % (name))

    pd.DataFrame(y-yhat, columns=["yhat%s" % LOOK_AHEAD]).plot(figsize=(15, 10))
    plt.title("diff-%s" % name)
    plt.tight_layout()
    plt.savefig("figures/%s-diff.png" % (name))


# def main():
#     global UUID, SPLIT, BATCH_SIZE, N_STEPS, EPOCHS, X_FREQ, LOOK_AHEAD, FILTERS, KERNEL_SIZE, L1L2, CELLS_D1, PDB
#     load_data()

#     log.info(f"UUID={UUID}")
#     log.info(f"SPLIT={SPLIT}")
#     log.info(f"BATCH_SIZE={BATCH_SIZE}")
#     log.info(f"N_STEPS={N_STEPS}")
#     log.info(f"EPOCHS={EPOCHS}")
#     log.info(f"X_FREQ={X_FREQ}")
#     log.info(f"LOOK_AHEAD={LOOK_AHEAD}")
#     log.info(f"FILTERS={FILTERS}")
#     log.info(f"KERNEL_SIZE={KERNEL_SIZE}")
#     log.info(f"L1L2={L1L2}")
#     log.info(f"CELLS_D1={CELLS_D1}")
#     log.info(f"NEURONS_OUT={NEURONS_OUT}")

#     train_X_collection = []
#     train_y_collection = []

#     test_X_collection = []
#     test_y_collection = []

#     for river in all_levels.T:
#         train, test = train_test_split(river[:, None], SPLIT)

#         train_X, train_y = split_sequence(train, N_STEPS, X_FREQ, LOOK_AHEAD)
#         test_X, test_y = split_sequence(test, N_STEPS, X_FREQ, LOOK_AHEAD)

#         train_X_collection.append(train_X)
#         train_y_collection.append(train_y)

#         test_X_collection.append(test_X)
#         test_y_collection.append(test_y)

#     train_X = np.concatenate(train_X_collection, axis=2)
#     test_X = np.concatenate(test_X_collection, axis=2)

#     if PDB:
#         __import__('pdb').set_trace()
#     TIME_STEPS = train_X.shape[1]
#     N_FEATURES = train_X.shape[2]

#     model = build_model(TIME_STEPS, N_FEATURES, FILTERS, KERNEL_SIZE, L1L2, CELLS_D1, NEURONS_OUT)

#     history = model.fit(
#         x=train_X,
#         y=train_y,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         verbose=1,
#         # validation_split=1-SPLIT,
#         # shuffle=False,
#         validation_data=(test_X, test_y),
#         callbacks=[
#             # TensorBoard(
#             #     log_dir='./logs',
#             #     histogram_freq=0,
#             #     write_graph=True,
#             #     write_images=False
#             # ),
#             # ModelCheckpoint(
#             #     filepath='models/cnn-%s.{epoch:02d}-{val_loss:.2f}.h5' % UUID,
#             #     monitor='val_loss',
#             #     save_best_only=True
#             # ),
#             EarlyStopping(
#                 monitor='val_loss',
#                 min_delta=0.0001,
#                 patience=PATIENCE,
#                 verbose=0,
#                 mode='auto',
#             )
#         ]
#     )

#     if PDB:
#         __import__('pdb').set_trace()
#     train_yhat = model.predict(train_X)
#     score_train = mean_squared_error(train_y, train_yhat)

#     test_yhat = model.predict(test_X)
#     score_test = mean_squared_error(test_y, test_yhat)

#     log.info(f"SCORE_TRAIN={score_train}")
#     log.info(f"SCORE_TEST={score_test}")

#     date = datetime.now().strftime("%Y%m%d-%H%M%S")
#     fname = "CNN-%s-%s-%.4f-%.4f" % (UUID, date, score_train, score_test)
#     log.info(f"fname={fname}")

#     plot_pred(train_y, train_yhat, f"train-{fname}")
#     plot_pred(train_y[-2000:], train_yhat[-2000:], f"train-{fname}-ZOOM_LAST")

#     plot_pred(test_y, test_yhat, f"test-{fname}")
#     plot_pred(test_y[-2000:], test_yhat[-2000:], f"test-{fname}-ZOOM_LAST")

#     model.save("models/%s.h5" % fname)
#     with open("models/%s.json" % fname, "w") as f:
#         f.write(model.to_json())

def split_sequence2(x, x_target=5, look_ahead=1, norm=True, return_scaler=False):
    y = x
    if norm:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    out_x = x[:-look_ahead]
    out_y = y[look_ahead:, x_target][:, None].copy()

    if norm and return_scaler:
        return out_x, out_y, scaler

    return out_x, out_y


def main():
    # global UUID, SPLIT, BATCH_SIZE, N_STEPS, EPOCHS, X_FREQ, LOOK_AHEAD, FILTERS, KERNEL_SIZE, L1L2, CELLS_D1, PDB
    global UUID, SPLIT, BATCH_SIZE, EPOCHS, X_FREQ, LOOK_AHEAD, FILTERS, KERNEL_SIZE, L1L2, CELLS_D1, PDB, NEURONS_OUT
    load_data()
    send_log()

    train, test = train_test_split(all_levels, SPLIT)
    train_X, train_y = split_sequence2(train, 5, LOOK_AHEAD)
    test_X, test_y = split_sequence2(test, 5, LOOK_AHEAD)

    test_X = test_X[:, :, None]
    train_X = train_X[:, :, None]
    # test_y = test_y[:, None, :]
    # train_y = train_y[:, None, :]

    if PDB:
        __import__('pdb').set_trace()

    TIME_STEPS = train_X.shape[1]
    N_FEATURES = train_X.shape[2]

    initializer=glorot_normal(7)
    model = build_model2(TIME_STEPS, N_FEATURES, FILTERS, KERNEL_SIZE, L1L2, CELLS_D1, CELLS_D2, NEURONS_OUT, initializer)

    history = model.fit(
        x=train_X,
        y=train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        # validation_split=1-SPLIT,
        # shuffle=False,
        validation_data=(test_X, test_y),
        callbacks=[
            # TensorBoard(
            #     log_dir='./logs',
            #     histogram_freq=0,
            #     write_graph=True,
            #     write_images=False
            # ),
            # ModelCheckpoint(
            #     filepath='models/cnn-%s.{epoch:02d}-{val_loss:.2f}.h5' % UUID,
            #     monitor='val_loss',
            #     save_best_only=True
            # ),
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=PATIENCE,
                verbose=0,
                mode='auto',
            )
        ]
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
    fname = "CNN-%s-%s-%s-%.4f-%.4f" % (LOOK_AHEAD, UUID, date, score_train, score_test)
    log.info(f"fname={fname}")

    plot_pred(train_y, train_yhat, f"train-{fname}")
    plot_pred(train_y[-2000:], train_yhat[-2000:], f"train-{fname}-ZOOM_LAST")

    plot_pred(test_y, test_yhat, f"test-{fname}")
    plot_pred(test_y[-2000:], test_yhat[-2000:], f"test-{fname}-ZOOM_LAST")

    model.save("models/%s.h5" % fname)


def load_model(name=None, path="models/"):
    from pathlib import Path
    from collections import defaultdict
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
    doc = "Look ahead."
    parser.add_argument("-la", "--look-ahead", type=int, default="24", help=doc)
    doc = "Set patience training param."
    parser.add_argument("-p", "--patience", type=int, default=20, help=doc)
    doc = "Number of neurons in the first Dense layer."
    parser.add_argument("-n1", "--neurons1", type=int, default=48, help=doc)
    doc = "Number of neurons in the sencond Dense layer."
    parser.add_argument("-n2", "--neurons2", type=int, default=None, help=doc)
    doc = "Define a regularizer like 'l1(0.1)' or 'l2(0.1)'."
    parser.add_argument("-l", "--l1l2", type=str, default="None", help=doc)
    parser.add_argument("--pdb", action="count")
    parser.add_argument("--log-path", type=str, default="logs")
    args = parser.parse_args()

    if not args.neurons2:
        args.neurons2 = None
    def send_log():
        log.info(f"UUID={UUID}")
        log.info(f"SPLIT={SPLIT}")
        log.info(f"BATCH_SIZE={BATCH_SIZE}")
        # log.info(f"N_STEPS={N_STEPS}")
        log.info(f"EPOCHS={EPOCHS}")
        log.info(f"PATIENCE={PATIENCE}")
        log.info(f"X_FREQ={X_FREQ}")
        log.info(f"LOOK_AHEAD={LOOK_AHEAD}")
        log.info(f"FILTERS={FILTERS}")
        log.info(f"KERNEL_SIZE={KERNEL_SIZE}")
        log.info(f"L1L2={L1L2}")
        log.info(f"CELLS_D1={CELLS_D1}")
        log.info(f"CELLS_D2={CELLS_D2}")
        log.info(f"NEURONS_OUT={NEURONS_OUT}")

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

