import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("ascii").strip()


def get_git_revision_short_hash():
    try: 
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode("ascii").strip()
    except:
        return ""


def crosscorr(x, y, nlags):
   xdm = x - np.mean(x)
   ydm = y - np.mean(y)
   lags = np.arange(-nlags+1, nlags)
   allag =  np.arange(-x.shape[0] + 1, x.shape[0])
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
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

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


def example():
    N = 100

    plt.figure()
    __import__('pdb').set_trace()
    x = np.arange(N)
    plt.imshow(recurrence_matrix(x))
    plt.show()

    plt.figure()
    x = np.random.randn(N)
    plt.imshow(recurrence_matrix(x))
    plt.show()

    plt.figure()
    sns.heatmap(recurrence_matrix(x))
    plt.show()

    plt.figure()
    x = np.sin(np.arange(N))
    sns.heatmap(recurrence_matrix(x), cmap="viridis")
    plt.show()

