import subprocess

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from numpy.lib.stride_tricks import as_strided

from sklearn.preprocessing import StandardScaler


def get_git_revision_hash():
    cmd = ['git', 'rev-parse', 'HEAD']
    return subprocess.check_output(cmd).decode("ascii").strip()


def get_git_revision_short_hash():
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    return subprocess.check_output(cmd).decode("ascii").strip()


def crosscorr(x, y, nlags):
    xdm = x - np.mean(x)
    ydm = y - np.mean(y)
    lags = np.arange(-nlags+1, nlags)
    allag = np.arange(1 - x.shape[0], x.shape[0])
    crosscorr = np.correlate(xdm, ydm, mode='full')
    crosscorr /= x.shape[0] * np.std(x) * np.std(y)
    out = pd.Series(crosscorr, index=allag)
    return out.loc[lags]


def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    if detrend:
        import matplotlib.mlab as mlab

        x = mlab.detrend_mean(np.asarray(x))  # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))

    c = np.correlate(x, y, mode='full')

    if normed:
        # this is the transformation function
        n = np.sqrt(np.dot(x, x) * np.dot(y, y))
        c = np.true_divide(c, n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c


def recurrence_matrix(s, eps=0.10, steps=10):
    d = pdist(s[:, None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z


def split_sequence_time_steps(sequence, look_back, look_ahead, target_idx,
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
    sequence_x = sequence[:-look_ahead]
    sequence_y = sequence[look_back + look_ahead - 1:, [target_idx]]
    if norm:
        scaler = StandardScaler()
        sequence_x = scaler.fit_transform(sequence_x)
    out_x = ndseq2strided(sequence_x, look_back)
    out_y = ndseq2strided(sequence_y, 1)
    if norm and return_scaler:
        return out_x, out_y, scaler
    return out_x, out_y


def seq2strided(seq, window):
    # http://arogozhnikov.github.io/2015/09/30/NumpyTipsAndTricks2.html
    stride = seq.strides[0]
    ts = as_strided(
            seq,
            shape=[len(seq) - window + 1, window],
            strides=[stride, stride])
    return ts


def ndseq2strided(ndseq, win_rows, win_cols=None):
    rows, cols = ndseq.shape

    if win_cols is None:
        _win_cols = cols

    shape = [rows - win_rows + 1, cols - _win_cols + 1, win_rows, _win_cols]
    strides = ndseq.strides + ndseq.strides
    strided_seq = as_strided(ndseq, shape=shape, strides=strides)

    if win_cols is None:
        axis2squeeze = 1  # this is due to cols == win_cols
        strided_seq = np.squeeze(strided_seq, axis2squeeze)

    return strided_seq


def train_test_split(dataset, train_frac):
    train_size = int(len(dataset)*train_frac)
    return dataset[:train_size, :], dataset[train_size:, :]


def sequece_nd2time_steps(x, col_target, look_back, y_ahead, x_freq,
                          norm=True, return_scaler=False, split=0.8):
    n_vars = x.shape[1]
    train_X_collection = []
    train_y_collection = []

    test_X_collection = []
    test_y_collection = []

    train, test = train_test_split(x[:, None], train_size=split, shuffle=False)

    for i in range(n_vars):
        train_X, train_y = split_sequence_time_steps(
            train, look_back, x_freq, y_ahead=y_ahead, norm=norm,
            return_scaler=return_scaler)
        test_X, test_y = split_sequence_time_steps(
            train, look_back, x_freq, y_ahead=y_ahead, norm=norm,
            return_scaler=return_scaler)

        train_X_collection.append(train_X)
        test_X_collection.append(test_X)

        train_y_collection.append(train_y)
        test_y_collection.append(test_y)

    train_X = np.concatenate(train_X_collection, axis=2)
    test_X = np.concatenate(test_X_collection, axis=2)


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


def split_sequence(x, x_target=0, look_ahead=1, norm_x=True,
                   return_scaler=False):
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

