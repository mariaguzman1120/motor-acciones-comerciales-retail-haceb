"""Carga, limpieza y persistencia de datos crudos para el pipeline NBA."""

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def load_raw_data():
    """Carga los tres archivos CSV crudos.

    Returns:
        Tupla (clientes, transacciones, interacciones) como DataFrames.
    """
    clientes = pd.read_csv(os.path.join(RAW_DIR, 'clientes.csv'))
    transacciones = pd.read_csv(
        os.path.join(RAW_DIR, 'transacciones.csv'),
        parse_dates=['fecha_compra'],
    )
    interacciones = pd.read_csv(
        os.path.join(RAW_DIR, 'interacciones_tienda.csv'),
        parse_dates=['fecha_visita'],
    )
    logger.info(
        'Datos cargados: clientes=%d, transacciones=%d, interacciones=%d',
        len(clientes), len(transacciones), len(interacciones),
    )
    return clientes, transacciones, interacciones


def clean_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset de clientes.

    Reemplaza celular y email invalidos por flags binarios de validez.
    Imputa el score de credito faltante con la mediana por zona.

    Args:
        df: DataFrame crudo de clientes.

    Returns:
        DataFrame limpio con columnas de flag en lugar de celular y email.
    """
    df['tiene_celular_valido'] = df['celular_contacto'].apply(
        lambda x: 1 if re.match(r'^\d{10}$', str(x)) else 0
    )
    df['tiene_email_valido'] = df['email_contacto'].apply(
        lambda x: 1 if re.match(r'^[\w.-]+@[\w.-]+\.\w+$', str(x)) else 0
    )

    mediana_zona = df.groupby('zona_geografica')['score_credito_interno']
    mediana_zona = mediana_zona.transform('median')
    df['score_credito_interno'] = df['score_credito_interno'].fillna(mediana_zona)
    df['score_credito_interno'] = df['score_credito_interno'].fillna(
        df['score_credito_interno'].median()
    )

    df = df.drop(columns=['celular_contacto', 'email_contacto'])
    logger.info('Clientes limpiados: %d filas, %d columnas', *df.shape)
    return df


def save_processed(
    clientes_clean: pd.DataFrame,
    transacciones: pd.DataFrame,
    interacciones: pd.DataFrame,
) -> None:
    """Persiste los datasets limpios como archivos Parquet.

    Args:
        clientes_clean: DataFrame de clientes limpio.
        transacciones: DataFrame de transacciones.
        interacciones: DataFrame de interacciones de tienda.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    clientes_clean.to_parquet(
        os.path.join(PROCESSED_DIR, 'clientes_clean.parquet'), index=False
    )
    transacciones.to_parquet(
        os.path.join(PROCESSED_DIR, 'transacciones_clean.parquet'), index=False
    )
    interacciones.to_parquet(
        os.path.join(PROCESSED_DIR, 'interacciones_clean.parquet'), index=False
    )
    logger.info('Datos procesados guardados en %s', PROCESSED_DIR)
