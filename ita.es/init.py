"""Initialize."""
from pathlib import Path

FILE_DIR = Path(__file__).parents[0].absolute().expanduser()
ASSETS_DIR = FILE_DIR
MODELS_DIR = ASSETS_DIR
FIG_DIR = ASSETS_DIR
DATA_DIR = ASSETS_DIR
IN_FILE = DATA_DIR / "ENTRADA" / "datos.csv"
OUT_FILE = DATA_DIR / "SALIDA" / "resultados.csv"
LOG_DIR = ASSETS_DIR
