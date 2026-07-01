"""Segmentacion de clientes con K-Means: entrenamiento y seleccion de k."""

import logging
import os

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

SEGMENT_FEATURES = [
    'recency_dias', 'frequency', 'monetary_total', 'monetary_avg',
    'total_visitas', 'tasa_conversion_hist', 'usa_credito_propio',
    'diversidad_categorias', 'score_credito_interno', 'antiguedad_meses',
]

SEGMENT_NAMES = {
    0: 'VIP Comprador',
    1: 'Cliente Financiado',
    2: 'Visitante Servicios',
    3: 'Cliente Dormido',
    4: 'Nuevo Potencial',
}


def find_optimal_k(
    X_scaled: pd.DataFrame,
    k_range: range = range(2, 8),
) -> pd.DataFrame:
    """Evalua K-Means para un rango de k usando inercia y silhouette score.

    Args:
        X_scaled: Matriz de features escalada.
        k_range: Rango de valores de k a evaluar.

    Returns:
        DataFrame con columnas k, inertia y silhouette.
    """
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
        logger.info('k=%d silhouette=%.4f inertia=%.0f', k, sil, km.inertia_)
    return pd.DataFrame(results)


def train_segmentation(c360: pd.DataFrame, n_clusters: int = 5):
    """Entrena la segmentacion K-Means y persiste los artefactos del modelo.

    Args:
        c360: DataFrame Customer 360.
        n_clusters: Numero de clusters.

    Returns:
        Tupla (c360_con_segmentos, KMeans ajustado, StandardScaler ajustado).
    """
    X = c360[SEGMENT_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    c360 = c360.loc[:, :]
    c360['segmento_id'] = labels
    c360['segmento_nombre'] = c360['segmento_id'].map(
        lambda x: SEGMENT_NAMES.get(x, f'Segmento {x}')
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(km, os.path.join(MODELS_DIR, 'kmeans_segmentation.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler_segmentation.joblib'))

    logger.info(
        'Segmentacion entrenada: k=%d, silhouette=%.4f', n_clusters, sil
    )
    return c360, km, scaler
