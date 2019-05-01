"""Initialize."""
from pathlib import Path

FILE_DIR = Path(__file__).parents[1].absolute().expanduser()
ASSETS_DIR = FILE_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
FIG_DIR = ASSETS_DIR / "figures"
DATA_DIR = ASSETS_DIR / "data"
IN_FILE = DATA_DIR / "raw" / "datos.csv"
OUT_FILE = DATA_DIR / "out" / "resultados.csv"
LOG_DIR = ASSETS_DIR / "logs"
