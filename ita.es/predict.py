"""CNN model."""
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_data(fname, fill_na=True, y_data=False):
    """Loading data."""
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


def main():
    IN_DIR = Path("ENTRADA/datos.csv")
    OUT_DIR = Path("SALIDA/resultados.csv")
    data = load_data(IN_DIR)
    m24 = load_model("CNN_24.h5")
    m48 = load_model("CNN_48.h5")
    m72 = load_model("CNN_72.h5")
    model_list = [m24, m48, m72]
    yhat_list = []
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)[:, :, None]

    for model in model_list:
        yhat_list.append(model.predict(X))

    values = np.concatenate(yhat_list, axis=1)
    df = pd.DataFrame(
        values,
        index=data.index,
        columns=["H24", "H48", "H72"]
    )
    df.index.name = "time"
    df.to_csv(OUT_DIR, index=True)


if __name__ == "__main__":
    main()

