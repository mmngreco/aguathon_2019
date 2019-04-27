"""CNN model."""
import logging
import sys
import os
from datetime import datetime
from uuid import uuid4
import pandas as pd
import matplotlib.pyplot as plt
import utils
import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
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

fmt = '%(name)s | %(asctime)s | %(levelname)s | %(message)s'
logging.basicConfig(format=fmt, level=logging.INFO)
log = logging.getLogger(__name__)

if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_test_split(dataset, train_frac):
    train_size = int(len(dataset)*train_frac)
    return dataset[:train_size, :], dataset[train_size:, :]


def create_datasets(dataset, look_back=1, look_ahead=1, jump=1):
    """create_datasets.
    """
    data_x, data_y = [], []

    for i in range(0, len(dataset)-look_back-look_ahead+1, jump):
        window = dataset[i:(i+look_back), 0]
        data_x.append(window)
        data_y.append(dataset[i + look_back:i + look_back + look_ahead, 0])

    return np.array(data_x), np.array(data_y)


def reverse_scale(data, mean, std):
    return data*std + mean


def calculate_error(train_y, test_y, pred_train, pred_test):
    test_score = np.sqrt(mean_squared_error(test_y, pred_test))
    train_score = np.sqrt(mean_squared_error(train_y, pred_train))
    return train_score, test_score


def mean_absolute_percentage(data_y, data_y_pred):
    return np.mean(np.abs((data_y - data_y_pred) / data_y)) * 100


def root_mse(pred_test, test_y):
    t = []
    for i in range(20):
        score = np.sqrt(
            mean_squared_error(pred_test[:, i, :], test_y[:, i, :])
        )
        t.append(score)

    return score


def split_sequence(sequence, look_back, x_freq, y_ahead=[0], norm=True,
                   return_scaler=False):
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
    _y_ahead = max(y_ahead)

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
        seq_y = [sequence[end_ix + y_ahead_i, 0] for y_ahead_i in y_ahead]
        y.append(seq_y)

    out_x = array(X)
    # out_y = [yi[:, None] for yi in array(y).T]
    out_y = array(y)

    if norm and return_scaler:
        return out_x, out_y, scaler

    return out_x, out_y

# CNN
# ----------------------------------------------------------------------------


def load_data():
    """load_data to global scope"""
    global data, data_fixed, all_levels, river
    fname = "./datos.csv"
    data = pd.read_csv(fname, index_col=0)
    data.index = data.index.astype("datetime64[ns]")
    data.sort_index(ascending=True, inplace=True)

    data_fixed = data.groupby(lambda x: x.weekofyear)\
                     .transform(lambda x: x.fillna(x.mean()))

    all_levels = data_fixed.iloc[:, :6].values.astype("float32")
    river = all_levels[:, 5]  # zgz


def build_model(n_steps, n_features, filters, kernel_size, L1L2, NEURONS1,
                NEURONS_OUT):
    """build_model"""
    model = Sequential()
    model.add(Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        input_shape=(n_steps, n_features),
    ))
    model.add(Dense(
        NEURONS1,
        activation='relu',
        kernel_regularizer=L1L2,
    ))
    # model.add(Dense(
    #     n_features,
    #     activation='relu',
    #     kernel_regularizer=L1L2,
    # ))
    model.add(Flatten())
    model.add(Dense(NEURONS_OUT))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def plot_pred(y, yhat, name):
    ahead_list = JUMP
    for i in range(y.shape[1]):
        ahead = ahead_list[i]
        _y = y[:, i][:, None]
        _yhat = yhat[:, i][:, None]
        ax = pd.DataFrame(_y, columns=["y%s" % ahead]).plot(figsize=(15, 10))
        pd.DataFrame(_yhat, columns=["yhat%s" % ahead]).plot(ax=ax)
        plt.title("%s" % name)
        plt.tight_layout()
        plt.savefig("figures/%s-%s.png" % (name, ahead))

        pd.DataFrame(_y-_yhat, columns=["yhat%s" % ahead]).plot(figsize=(15, 10))
        plt.title("diff-%s" % name)
        plt.tight_layout()
        plt.savefig("figures/%s-%s-diff.png" % (name, ahead))


def main():
    global UUID, SPLIT, BATCH_SIZE, N_STEPS, EPOCHS, X_FREQ, JUMP, FILTERS, KERNEL_SIZE, L1L2, NEURONS1
    load_data()

    log.info(f"UUID={UUID}")
    log.info(f"SPLIT={SPLIT}")
    log.info(f"BATCH_SIZE={BATCH_SIZE}")
    log.info(f"N_STEPS={N_STEPS}")
    log.info(f"EPOCHS={EPOCHS}")
    log.info(f"X_FREQ={X_FREQ}")
    log.info(f"JUMP={JUMP}")
    log.info(f"FILTERS={FILTERS}")
    log.info(f"KERNEL_SIZE={KERNEL_SIZE}")
    log.info(f"L1L2={L1L2}")
    log.info(f"NEURONS1={NEURONS1}")
    log.info(f"NEURONS_OUT={NEURONS_OUT}")

    train_X_collection = []
    train_y_collection = []

    test_X_collection = []
    test_y_collection = []

    for river in all_levels.T:
        train, test = train_test_split(river[:, None], SPLIT)

        train_X, train_y = split_sequence(train, N_STEPS, X_FREQ, JUMP)
        test_X, test_y = split_sequence(test, N_STEPS, X_FREQ, JUMP)

        train_X_collection.append(train_X)
        train_y_collection.append(train_y)

        test_X_collection.append(test_X)
        test_y_collection.append(test_y)

    train_X = np.concatenate(train_X_collection, axis=2)
    test_X = np.concatenate(test_X_collection, axis=2)

    # __import__('pdb').set_trace()
    TIME_STEPS = train_X.shape[1]
    N_FEATURES = train_X.shape[2]
    model = build_model(TIME_STEPS, N_FEATURES, FILTERS, KERNEL_SIZE, L1L2,
                        NEURONS1, NEURONS_OUT)

    history = model.fit(
        x=train_X,
        y=train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        # validation_split=1-SPLIT,
        shuffle=False,
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
                min_delta=0.00001,
                patience=PATIENCE,
                verbose=2,
                mode='auto',
            )
        ]
    )
    __import__('pdb').set_trace()
    train_yhat = model.predict(train_X)
    score_train = mean_squared_error(train_y, train_yhat)

    test_yhat = model.predict(test_X)
    score_test = mean_squared_error(test_y, test_yhat)

    log.info(f"SCORE_TRAIN={score_train}")
    log.info(f"SCORE_TEST={score_test}")

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = "CNN-%s-%s-%.4f-%.4f" % (UUID, date, score_train, score_test)
    log.info(f"fname={fname}")

    plot_pred(train_y, train_yhat, f"train-{fname}")
    plot_pred(train_y[-2000:], train_yhat[-2000:], f"train-{fname}-ZOOM_LAST")

    plot_pred(test_y, test_yhat, f"test-{fname}")
    plot_pred(test_y[-2000:], test_yhat[-2000:], f"test-{fname}-ZOOM_LAST")

    model.save("models/%s.h5" % fname)
    with open("models/%s.json" % fname, "w") as f:
        f.write(model.to_json())


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
    import argparse
    parser = argparse.ArgumentParser()

    doc = "test/train split"
    parser.add_argument("-s", "--split", type=float, default=0.8, help=doc)
    doc = "Batch size."
    parser.add_argument("-b", "--batch-size", type=int, default=512, help=doc)
    doc = "Number of time steps to look back."
    parser.add_argument("-lb", "--look-back", type=int, default=64, help=doc)
    doc = "Number of epochs."
    parser.add_argument("-e", "--epochs", type=int, default=230, help=doc)
    doc = "Number of filters."
    parser.add_argument("-f", "--filters", type=int, default=8, help=doc)
    doc = "Look ahead."
    parser.add_argument("-la", "--look-ahead", type=str, default="[24]", help=doc)
    doc = "Set patience training param."
    parser.add_argument("-p", "--patience", type=int, default=20, help=doc)
    doc = "Number of neurons in the first Dense layer."
    parser.add_argument("-n1", "--neurons1", type=int, default=48, help=doc)
    doc = "Define a regularizer like 'l1(0.1)' or 'l2(0.1)'."
    parser.add_argument("-l", "--l1l2", type=str, default="None", help=doc)
    args = parser.parse_args()

    # parameters
    UUID = utils.get_git_revision_short_hash()
    SPLIT = args.split
    BATCH_SIZE = args.batch_size
    N_STEPS = args.look_back
    EPOCHS = args.epochs
    X_FREQ = 1
    JUMP = eval(args.look_ahead)
    FILTERS = args.filters
    KERNEL_SIZE = args.look_back
    PATIENCE = args.patience
    L1L2 = eval(args.l1l2)
    NEURONS1 = args.neurons1
    NEURONS_OUT = len(JUMP)
    __import__('pdb').set_trace()
    main()

