"""CNN model."""
import sys
import os
from uuid import uuid4
import pandas as pd
import matplotlib.pyplot as plt
import utils
import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
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
    # MaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
plt.ion()
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# plt.switch_backend('TkAgg')
# get_ipython().run_line_magic('matplotlib', 'inline')


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


def split_sequence(sequence, look_back, x_freq, y_ahead=0, norm=True,
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
    last_idx = len(sequence)-y_ahead-1
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

        seq_y = sequence[end_ix + y_ahead]
        y.append(seq_y)

    if norm and return_scaler:
        return array(X), array(y), scaler

    return array(X), array(y)

# CNN
# ----------------------------------------------------------------------------


def load_data():
    """load_data"""
    global data, data_fixed, all_levels, river
    fname = "./datos.csv"
    data = pd.read_csv(fname, index_col=0)
    data.index = data.index.astype("datetime64[ns]")
    data.sort_index(ascending=True, inplace=True)

    data_fixed = data.groupby(lambda x: x.weekofyear)\
                     .transform(lambda x: x.fillna(x.mean()))

    all_levels = data_fixed.iloc[:, :6].values.astype("float32")
    # win = 6
    # all_levels = data_fixed.iloc[:, :6].rolling(win).mean().values[win:].astype("float32")

    river = all_levels[:, 5]  # zgz


def build_model(n_steps, n_features, filters, kernel_size):
    """build_model"""
    model = Sequential()
    model.add(Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        input_shape=(n_steps, n_features),
        # kernel_regularizer=l1(0.1),
        # bias_regularizer=l1(0.1),
    ))
    # model.add(Conv1D(
    #     filters=64,
    #     kernel_size=8,
    #     activation='relu',
    #     kernel_regularizer=l1(0.1),
    #     bias_regularizer=l1(0.1),
    # ))
    # model.add(Conv1D(
    #     filters=32,
    #     kernel_size=8,
    #     activation='relu',
    #     input_shape=(n_steps, n_features),
    #     kernel_regularizer=l1(0.2),
    # ))
    # model.add(Dropout(0.4))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Dense(
    #     4,
    #     activation='relu',
    #     kernel_regularizer=l1(0.1),
    #     bias_regularizer=l1(0.1),
    # ))
    model.add(Dense(
        n_features,
        activation='relu',
        kernel_regularizer=l1(0.1),
        bias_regularizer=l1(0.1),
    ))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def plot_pred(y, yhat, uuid, score, name):
    ax = pd.DataFrame(y, columns=["y"]).plot(figsize=(15, 10))
    pd.DataFrame(yhat, columns=["yhat"]).plot(ax=ax)
    plt.title("%s - %s - Score=%.3f" % (name, uuid, score))
    plt.savefig("figures/%s-%s.png" % (name, uuid))
    plt.show()


def main():
    global UUID, SPLIT, BATCH_SIZE, N_STEPS, EPOCHS, X_FREQ, JUMP, FILTERS, KERNEL_SIZE
    load_data()

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
    # train_y = train_y
    # test_y = test_y

    TIME_STEPS = train_X.shape[1]
    N_FEATURES = train_X.shape[2]
    model = build_model(TIME_STEPS, N_FEATURES, FILTERS, KERNEL_SIZE)

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
            TensorBoard(
                log_dir='./logs',
                histogram_freq=0,
                write_graph=True,
                write_images=False
            ),
            # ModelCheckpoint(
            #     filepath='models/cnn-%s.{epoch:02d}-{val_loss:.2f}.h5' % UUID,
            #     monitor='val_loss',
            #     save_best_only=True
            # ),
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.00001,
                patience=5,
                verbose=2,
                mode='auto',
            )
        ]
    )

    train_yhat = model.predict(train_X)
    score_train = mean_squared_error(train_y, train_yhat)
    plot_pred(train_y, train_yhat, UUID, score_train, "train")
    plot_pred(train_y[-2000:], train_yhat[-2000:], UUID, score_train, "train-2000")

    test_yhat = model.predict(test_X)
    score_test = mean_squared_error(test_y, test_yhat)
    plot_pred(test_y, test_yhat, UUID, score_test, "test")
    plot_pred(test_y[-2000:], test_yhat[-2000:], UUID, score_test, "test-2000")


    fname = "models/CNN-%s-%.4f-%.4f" % (UUID, score_train, score_test)
    model.save("%s.h5" % fname)
    with open("%s.json" % fname, "w") as f:
        f.write(model.to_json())


if __name__ == "__main__":
    # constants
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", type=float, default=0.8, help="test/train split")
    parser.add_argument("-b", "--batch-size", type=int, default=512, help="batch size.")
    parser.add_argument("-n", "--n-steps", type=int, default=64, help="Number of steps to look back.")
    parser.add_argument("-e", "--epochs", type=int, default=230, help="Number of epochs.")
    parser.add_argument("-k", "--kernel-size", type=int, default=8, help="Number of epochs.")
    parser.add_argument("-f", "--filters", type=int, default=8, help="Number of filters.")
    parser.add_argument("-a", "--ahead", type=int, default=24, help="Look ahead.")
    args = parser.parse_args()

    # parameters
    UUID = utils.get_git_revision_short_hash()
    SPLIT = args.split
    BATCH_SIZE = args.batch_size
    N_STEPS = args.n_steps
    EPOCHS = args.epochs
    X_FREQ = 1
    JUMP = args.ahead
    FILTERS = args.filters
    KERNEL_SIZE = args.kernel_size

    main()

