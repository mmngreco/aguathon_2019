"""CNN model."""
import pickle
import sys
import os
from uuid import uuid4
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
    # Embedding,
    Dense,
    # Dropout,
    Conv1D,
    Flatten,
    MaxPooling1D,
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


def split_sequence(sequence, n_steps, x_freq, y_ahead=0):
    X, y = list(), list()
    last_idx = len(sequence)-y_ahead-1
    for i in range(0, last_idx):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > last_idx:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix:x_freq], sequence[end_ix+y_ahead]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# ----------------------------------------------------------------------------
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

    all_levels = data_fixed.iloc[:, :6].values.astype("float64")
    river = all_levels[:, 5]  # zgz


def build_model(n_steps, n_features):
    """build_model"""
    model = Sequential()
    model.add(Conv1D(
        filters=16,
        kernel_size=16,
        activation='relu',
        input_shape=(n_steps, n_features),
    ))
    model.add(Conv1D(
        filters=16,
        kernel_size=8,
        activation='relu',
        kernel_regularizer=l1(0.01),
    ))
    model.add(Conv1D(
        filters=16,
        kernel_size=8,
        activation='relu',
        kernel_regularizer=l1(0.01),
    ))
    model.add(Conv1D(
        filters=16,
        kernel_size=8,
        activation='relu',
        kernel_regularizer=l1(0.01),
    ))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model


def plot_pred(y, yhat, uuid, score, name):
    ax = pd.DataFrame(y, columns=["y"]).plot(figsize=(15, 10))
    pd.DataFrame(yhat, columns=["yhat"]).plot(ax=ax)
    plt.title("Score=%.3f" % score)
    plt.savefig("%s-%s.png" % (name, uuid))
    plt.show()


if __name__ == "__main__":

    # constants
    UUID = str(uuid4())[:8]
    SPLIT = 0.8
    BATCH_SIZE = 1200
    N_STEPS = 700
    N_FEATURES = 1
    EPOCHS = 30
    X_FREQ = 8
    JUMP = 72

    # variables
    load_data()
    train, test = train_test_split(river[:, None], SPLIT)
    train_X, train_y = split_sequence(train, N_STEPS, X_FREQ, JUMP)
    model = build_model(train_X.shape[1], N_FEATURES)

    scaler = StandardScaler()
    train_X_norm = scaler.fit_transform(train_X.squeeze())[:, :, None]
    history = model.fit(
        x=train_X_norm,
        y=train_y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        validation_split=1-SPLIT,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.00001,
                patience=5,
                verbose=2,
                mode='auto',
            )
        ]
    )

    train_yhat = model.predict(train_X_norm)
    score_train = mean_squared_error(train_y, train_yhat)
    plot_pred(train_y, train_yhat, UUID, score_train, "train")

    test_X, test_y = split_sequence(test, N_STEPS, X_FREQ, JUMP)
    test_X_norm = scaler.transform(test_X.squeeze())[:, :, None]
    test_yhat = model.predict(test_X_norm)
    score_test = mean_squared_error(test_y, test_yhat)
    plot_pred(test_y, test_yhat, UUID, score_test, "test")
    model.save("CNN-%s-%.4f_%.4f.pkl" % (UUID, score_train, score_test))

