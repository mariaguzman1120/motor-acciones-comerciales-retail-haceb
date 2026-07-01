"""Flujo de entrenamiento local con Prefect para el motor NBA."""

import logging
import os
from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from prefect import flow, task
except ImportError:
    logger.warning('Prefect no disponible, ejecutando sin orquestacion')

    def task(**kwargs):
        def decorator(fn):
            return fn
        return decorator

    def flow(**kwargs):
        def decorator(fn):
            return fn
        return decorator

from python.data.preprocessing import clean_clientes, load_raw_data, save_processed
from python.features.build_features import build_customer_360
from python.models.train_propensity import (
    optimize_hyperparams,
    prepare_model_data,
    train_final_model,
)
from python.models.train_segmentation import (
    SEGMENT_FEATURES,
    find_optimal_k,
    train_segmentation,
)
from python.monitoring.drift_report import generate_drift_report


@task(name='load_raw_data')
def load_raw_data_task() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los datasets crudos desde disco."""
    return load_raw_data()


@task(name='clean_clientes')
def clean_clientes_task(clientes: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset de clientes."""
    return clean_clientes(clientes)


@task(name='save_processed')
def save_processed_task(
    clientes_clean: pd.DataFrame,
    transacciones: pd.DataFrame,
    interacciones: pd.DataFrame,
) -> None:
    """Guarda los datasets procesados requeridos por el flujo."""
    save_processed(clientes_clean, transacciones, interacciones)


@task(name='build_customer_360')
def build_customer_360_task(
    clientes_clean: pd.DataFrame,
    transacciones: pd.DataFrame,
    interacciones: pd.DataFrame,
) -> pd.DataFrame:
    """Construye la tabla Customer 360."""
    return build_customer_360(clientes_clean, transacciones, interacciones)


@task(name='find_best_k')
def find_best_k_task(c360: pd.DataFrame) -> int:
    """Selecciona el k optimo para K-Means usando silhouette score."""
    X_seg = c360[SEGMENT_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_seg_scaled = scaler.fit_transform(X_seg)
    k_results = find_optimal_k(X_seg_scaled)
    best_k = int(k_results.loc[k_results['silhouette'].idxmax(), 'k'])
    return best_k


@task(name='train_segmentation')
def train_segmentation_task(
    c360: pd.DataFrame,
    n_clusters: int,
) -> pd.DataFrame:
    """Entrena la segmentacion y devuelve el Customer 360 enriquecido."""
    c360_seg, _, _ = train_segmentation(c360, n_clusters=n_clusters)
    return c360_seg


@task(name='prepare_model_data')
def prepare_model_data_task(
    c360_seg: pd.DataFrame,
    interacciones: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepara la matriz de entrenamiento del modelo de propension."""
    X, y, _ = prepare_model_data(c360_seg, interacciones)
    return X, y


@task(name='optimize_hyperparams')
def optimize_hyperparams_task(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
) -> dict[str, Any]:
    """Optimiza hiperparametros y devuelve el mejor conjunto encontrado."""
    study = optimize_hyperparams(X, y, n_trials=n_trials)
    return study.best_params


@task(name='train_final_model')
def train_final_model_task(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict[str, Any],
    experiment_name: str,
) -> tuple[float, float]:
    """Entrena el modelo final y devuelve metricas resumidas."""
    _, cv_scores = train_final_model(
        X,
        y,
        best_params,
        experiment_name=experiment_name,
    )
    return float(cv_scores.mean()), float(cv_scores.std())


@task(name='generate_drift_report')
def generate_drift_report_task(
    X: pd.DataFrame,
    y: pd.Series,
    output_name: str,
) -> str:
    """Genera un reporte de drift simple sobre dos cortes del dataset."""
    mid = len(X) // 2
    reference = X.iloc[:mid, :].loc[:, :]
    current = X.iloc[mid:, :].loc[:, :]
    reference['target'] = y.iloc[:mid].values
    current['target'] = y.iloc[mid:].values
    output_path = generate_drift_report(reference, current, output_name)
    return os.path.abspath(output_path)


@flow(name='training_pipeline_prefect', log_prints=True)
def training_flow(
    n_trials: int = 50,
    experiment_name: str = 'propension_compra_visita',
    create_drift_report: bool = True,
    drift_output_name: str = 'drift_report_prefect.html',
) -> dict[str, Any]:
    """Ejecuta el entrenamiento completo con observabilidad de Prefect.

    Args:
        n_trials: Numero de iteraciones para Optuna.
        experiment_name: Nombre del experimento para MLflow.
        create_drift_report: Indica si debe generar reporte de drift.
        drift_output_name: Nombre del archivo HTML de drift.

    Returns:
        Resumen de la ejecucion con metricas y rutas relevantes.
    """
    clientes, transacciones, interacciones = load_raw_data_task()
    clientes_clean = clean_clientes_task(clientes)
    save_processed_task(clientes_clean, transacciones, interacciones)

    c360 = build_customer_360_task(clientes_clean, transacciones, interacciones)
    best_k = find_best_k_task(c360)
    c360_seg = train_segmentation_task(c360, best_k)
    X, y = prepare_model_data_task(c360_seg, interacciones)
    best_params = optimize_hyperparams_task(X, y, n_trials)
    auc_mean, auc_std = train_final_model_task(
        X,
        y,
        best_params,
        experiment_name,
    )

    report_path = None
    if create_drift_report:
        report_path = generate_drift_report_task(X, y, drift_output_name)

    return {
        'experiment_name': experiment_name,
        'best_k': best_k,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'drift_report_path': report_path,
        'model_path': os.path.abspath(os.path.join('models', 'lgbm_propension.joblib')),
        'customer_360_path': os.path.abspath(
            os.path.join('data', 'processed', 'customer_360.parquet')
        ),
    }


if __name__ == '__main__':
    training_flow()
