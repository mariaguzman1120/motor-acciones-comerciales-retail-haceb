import pandas as pd

from python.metadata.pipeline_config import INPUT_FILES


def read_csv_file(path: str) -> pd.DataFrame:
    """Lee un archivo CSV manteniendo texto como tipo string."""
    data = pd.read_csv(path, dtype='string', keep_default_na=True)
    return data


def read_input_data(input_files: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """Lee las fuentes de entrada configuradas para el pipeline."""
    files = input_files or INPUT_FILES
    input_data = {name: read_csv_file(path) for name, path in files.items()}
    return input_data
