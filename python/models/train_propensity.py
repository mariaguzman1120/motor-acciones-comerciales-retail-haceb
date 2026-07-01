"""Modelo de propension LightGBM: preparacion de datos, HPO y tracking."""

import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

FEATURE_COLS = [
    'zona_geografica', 'antiguedad_meses', 'score_credito_interno',
    'tiene_celular_valido', 'tiene_email_valido',
    'recency_dias', 'frequency', 'monetary_total', 'monetary_avg',
    'diversidad_categorias', 'total_visitas', 'tasa_conversion_hist',
    'visitas_servicio_tecnico', 'visitas_cotizacion', 'visitas_reclamo',
    'motivo_visita', 'id_tienda', 'dia_semana', 'mes', 'es_quincena',
]

CAT_COLS = ['zona_geografica', 'motivo_visita', 'id_tienda']


def prepare_model_data(c360: pd.DataFrame, interacciones: pd.DataFrame):
    """Une el Customer 360 con las visitas y codifica variables categoricas.

    Cada fila representa una visita a tienda. El target es compro_en_visita.
    Persiste el OrdinalEncoder ajustado en models/.

    Args:
        c360: DataFrame Customer 360 (incluye columnas de segmentacion).
        interacciones: DataFrame de interacciones de tienda limpio.

    Returns:
        Tupla (X, y, OrdinalEncoder ajustado).
    """
    df = interacciones[
        [
            'id_cliente', 'id_tienda', 'fecha_visita',
            'motivo_visita', 'compro_en_visita',
        ]
    ].loc[:, :]

    merge_cols = [
        'id_cliente', 'antiguedad_meses', 'score_credito_interno',
        'tiene_celular_valido', 'tiene_email_valido', 'recency_dias',
        'frequency', 'monetary_total', 'monetary_avg', 'diversidad_categorias',
        'total_visitas', 'tasa_conversion_hist', 'visitas_servicio_tecnico',
        'visitas_cotizacion', 'visitas_reclamo', 'zona_geografica',
    ]
    df = df.merge(c360[merge_cols], on='id_cliente', how='left')

    df['dia_semana'] = df['fecha_visita'].dt.dayofweek
    df['mes'] = df['fecha_visita'].dt.month
    quincena_dias = [14, 15, 16, 28, 29, 30, 31]
    df['es_quincena'] = df['fecha_visita'].dt.day.isin(quincena_dias)
    df['es_quincena'] = df['es_quincena'].astype(int)

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[CAT_COLS] = enc.fit_transform(df[CAT_COLS])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(enc, os.path.join(MODELS_DIR, 'ordinal_encoder.joblib'))

    X = df[FEATURE_COLS].fillna(0)
    y = df['compro_en_visita']

    logger.info(
        'Datos preparados: %d obs, %d features, target=%.1f%%',
        len(X), X.shape[1], y.mean() * 100,
    )
    return X, y, enc


def optimize_hyperparams(X: pd.DataFrame, y: pd.Series, n_trials: int = 50):
    """Ejecuta busqueda bayesiana de hiperparametros con Optuna.

    Args:
        X: Matriz de features.
        y: Serie binaria de target.
        n_trials: Numero de trials de Optuna.

    Returns:
        Objeto Study de Optuna completado.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.01, 0.3, log=True
            ),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'min_child_samples': trial.suggest_int(
                'min_child_samples', 10, 60
            ),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.5, 1.0
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha', 1e-3, 10.0, log=True
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda', 1e-3, 10.0, log=True
            ),
            'verbose': -1,
            'random_state': 42,
        }
        model = LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    logger.info(
        'Optuna best AUC: %.4f params: %s', study.best_value, study.best_params
    )
    return study


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    experiment_name: str = 'propension_compra_visita',
):
    """Entrena el modelo final con los mejores hiperparametros y lo registra.

    Persiste el modelo ajustado en models/lgbm_propension.joblib.

    Args:
        X: Matriz de features.
        y: Serie binaria de target.
        best_params: Hiperparametros del Study de Optuna.
        experiment_name: Nombre del experimento en MLflow.

    Returns:
        Tupla (LGBMClassifier ajustado, array de AUC por fold CV).
    """
    mlflow.set_experiment(experiment_name)

    best_params['verbose'] = -1
    best_params['random_state'] = 42
    model = LGBMClassifier(**best_params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    model.fit(X, y)

    with mlflow.start_run(run_name='lgbm_optuna_best'):
        mlflow.log_params(best_params)
        mlflow.log_metric('auc_cv_mean', cv_scores.mean())
        mlflow.log_metric('auc_cv_std', cv_scores.std())
        mlflow.sklearn.log_model(
            model,
            'modelo_propension',
            skops_trusted_types=[
                'collections.OrderedDict',
                'lightgbm.basic.Booster',
                'lightgbm.sklearn.LGBMClassifier',
            ],
        )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'lgbm_propension.joblib'))

    logger.info(
        'Modelo final: AUC CV=%.4f (+/- %.4f)',
        cv_scores.mean(), cv_scores.std(),
    )
    return model, cv_scores
