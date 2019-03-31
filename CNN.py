# coding: utf-8
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Activation, RepeatVector, TimeDistributed, Conv1D, Flatten
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# plt.switch_backend('TkAgg')
# get_ipython().run_line_magic('matplotlib', 'inline')


def train_test_split(dataset, train_frac):
    train_size = int(len(dataset)*train_frac)
    return dataset[:train_size, :], dataset[train_size: ,:]


def create_datasets(dataset, look_back=1, look_ahead=1, jump=1):
    data_x, data_y = [], []
    for i in range(0, len(dataset)-look_back-look_ahead+1, jump):
        window = dataset[i:(i+look_back), 0]
        data_x.append(window)
        data_y.append(dataset[i + look_back:i + look_back + look_ahead , 0])
    return np.array(data_x), np.array(data_y)


def reverse_scale(data, mean, std):
    return data*std + mean


def calculate_error(train_y, test_y, pred_train, pred_test):
    test_score = math.sqrt(mean_squared_error(test_y, pred_test))
    train_score = math.sqrt(mean_squared_error(train_y, pred_train))
    return train_score, test_score


def mean_absolute_percentage(y, y_pred):
    return np.mean(np.abs((y - y_pred) / y)) * 100


def root_mse(pred_test, test_y):
    t = []
    for i in range(20):
        score = math.sqrt(mean_squared_error(pred_test[:,i,:], test_y[:,i,:]))
        t.append(score)
        print(i+1, "  ->  ", score)

    return score


def plot_errors(pred_test, test_y, errors):
    plt.figure(figsize=(20,10))
    plt.subplot(311)
    plt.plot(test_y[:,23,:], label="Observed")
    plt.plot(pred_test[:,23,:], color="red", label="Predicted, MAPE: "+ str(round(errors[23], 5))+"%")
    plt.title("24 step ahead prediction")
    plt.ylabel("River Level")
    plt.legend(loc=1, fontsize = 8, framealpha=0.8)

    plt.subplot(312)
    plt.plot(pred_test[:,47,:], color="red", label="Predicted, MAPE: "+ str(round(errors[47], 5))+"%")
    plt.plot(test_y[:,47,:], label="Observed")
    plt.title("48 step ahead prediction")
    plt.legend(loc=1, fontsize = 8, framealpha=0.8)

    plt.subplot(313)
    plt.plot(pred_test[:,71,:], color="red", label="Predicted, MAPE: "+ str(round(errors[71], 5))+"%")
    plt.plot(test_y[:,71,:], label="Observed")
    plt.title("72 step ahead prediction")
    plt.legend(loc=1, fontsize = 8, framealpha=0.8)
    plt.tight_layout()
    plt.show()


def build_seq2seq_model(look_ahead=1):
    m = Sequential()

    # encoder
    m.add(GRU(16, input_shape=(None, 1)))
    # m.add(GRU(16, input_dim = 1))

    # repeat for the number of steps out
    m.add(RepeatVector(look_ahead))

    # decoder
    m.add(GRU(8, return_sequences=True))
    m.add(GRU(8, return_sequences=True))

    # split the output into timesteps
    m.add(TimeDistributed(Dense(1)))

    m.compile(loss='mse', optimizer='rmsprop')

    m.summary()
    return m

# univariate data preparation
# split a univariate sequence into samples
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
# # CNN
# ----------------------------------------------------------------------------

# get data
fname = "./datos.csv"
data = pd.read_csv(fname, index_col=0)
data.index = data.index.astype("datetime64[ns]")
data.sort_index(ascending=True, inplace=True)
data_fixed = data.groupby(lambda x: x.weekofyear).transform(lambda x: x.fillna(x.mean()))
all_levels = data_fixed.iloc[:, :6].values.astype("float64")
names = data_fixed.columns[:6]
split = 0.8
batch_size=1200
river = all_levels[:N, 5]  # zgz
# split data into train and test subsets
train, test = train_test_split(river, split)
scaler = StandardScaler()

# ----------------------------------------------------------------------------
# MODEL

# define model
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=200, activation='relu', input_shape=(n_steps, n_features)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=20, kernel_size=200, activation='relu'))
# model.add(Conv1D(filters=20, kernel_size=100, activation='relu'))
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# model.summary()
# define model
model = Sequential()
model.add(Conv1D(filters=20, kernel_size=100, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()


# ----------------------------------------------------------------------------

n_steps = 200
X, y = split_sequence(train, n_steps, jump=72)
n_features = 1
X = scaler.fit_transform(X)

history = model.fit(
    x=X,
    y=y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=2,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=2, mode='auto')]
)

# ax = pd.DataFrame(y[:1000]).plot(label="y")
# pd.DataFrame(X[:1000,-1,0]).plot(ax=ax)

# ----------------------------------------------------------------------------

yhat=model.predict(X)

# ----------------------------------------------------------------------------

ax = pd.DataFrame(y[:5000], columns=["y"]).plot()
pd.DataFrame(yhat[:5000], columns=["yhat"]).plot(ax=ax)
ax.savefig("prediction_cnn.png")

scaler.fit_transform(X.flatten()[:, None])
test_ = scaler.transform(test)

# choose a number of time steps
# split into samples
X, y = split_sequence(test_, n_steps, x_freq=24, y_ahead=72)
model.predict(X)
errors = [mean_absolute_percentage(test_ytrue[:,i,:], test_yhat[:,i,:]) for i in range(test_ytrue.shape[1])]

# ----------------------------------------------------------------------------

with open("cnn.pkl", "wb") as f:
    pickle.dump(f, model)

