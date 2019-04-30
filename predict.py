"""CNN model."""
import logging
import sys
import os
import pandas as pd
import utils
from uuid import uuid4
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data(fname, fill_na=True):
    """load_data to global scope"""

    log.info(f"Loading file names {fname}")
    fname = Path("./datos.csv")
    global data, data_fixed, all_levels, river
    assert fname.exists()
    data = pd.read_csv(fname, index_col=0)
    data.index = data.index.astype("datetime64[ns]")
    data.sort_index(ascending=True, inplace=True)

    if fill_na:
        data = data.groupby(lambda x: x.weekofyear)\
                   .transform(lambda x: x.fillna(x.mean()))

    out = data.iloc[:, :6].values.astype("float32")
    return out


def export_data(data, fname):
    log.info(f"Exporting file to {fname}.")
    columns = ["zgz_nr", "pred24", "pred48", "pre72"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(fname)
    log.info(f"File named {fname} exported.")



def main():
    load_data()
    Path("best_models.txt")
    m24 = load_models()
    m48 = load_models()
    m72 = load_models()

    train_yhat = model.predict(train_X)
    score_train = mean_squared_error(train_y, train_yhat)

    test_yhat = model.predict(test_X)
    score_test = mean_squared_error(test_y, test_yhat)

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    log.info(f"fname={fname}")


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

    fmt = ('%(name)s | %(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    handlers = [sh]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)

    try:
        log = logging.getLogger(__file__)
    except:
        log = logging.getLogger()

    main()

