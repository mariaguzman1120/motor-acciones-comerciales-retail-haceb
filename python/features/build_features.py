"""Ingenieria de caracteristicas: construye la tabla Customer 360."""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def build_tx_features(transacciones: pd.DataFrame) -> pd.DataFrame:
    """Agrega transacciones en features RFM por cliente.

    Args:
        transacciones: DataFrame de transacciones limpias.

    Returns:
        Un registro por cliente con features RFM y afinidad de producto.
    """
    fecha_ref = transacciones['fecha_compra'].max()

    agg = transacciones.groupby('id_cliente').agg(
        recency_dias=('fecha_compra', lambda x: (fecha_ref - x.max()).days),
        frequency=('id_transaccion', 'count'),
        monetary_total=('monto_pago', 'sum'),
        monetary_avg=('monto_pago', 'mean'),
        ticket_max=('monto_pago', 'max'),
        diversidad_categorias=('categoria_producto', 'nunique'),
    )
    agg = agg.reset_index()

    cat_favorita = transacciones.groupby('id_cliente')['categoria_producto']
    cat_favorita = cat_favorita.agg(lambda x: x.value_counts().index[0])
    cat_favorita = cat_favorita.rename('categoria_favorita')

    medio_habitual = transacciones.groupby('id_cliente')['medio_pago']
    medio_habitual = medio_habitual.agg(lambda x: x.value_counts().index[0])
    medio_habitual = medio_habitual.rename('medio_pago_habitual')

    usa_credito = transacciones.groupby('id_cliente')['medio_pago']
    usa_credito = usa_credito.agg(
        lambda x: int((x == 'credito_propio_retail').any())
    )
    usa_credito = usa_credito.rename('usa_credito_propio')

    agg = agg.merge(cat_favorita, on='id_cliente')
    agg = agg.merge(medio_habitual, on='id_cliente')
    agg = agg.merge(usa_credito, on='id_cliente')

    logger.info(
        'Features de transacciones: %d clientes, %d features', *agg.shape
    )
    return agg


def build_inter_features(interacciones: pd.DataFrame) -> pd.DataFrame:
    """Agrega el historial de visitas en features conductuales por cliente.

    Args:
        interacciones: DataFrame de interacciones de tienda limpias.

    Returns:
        Un registro por cliente con frecuencia de visita y tasa de conversion.
    """
    fecha_ref = interacciones['fecha_visita'].max()

    agg = interacciones.groupby('id_cliente').agg(
        total_visitas=('id_interaccion', 'count'),
        tasa_conversion_hist=('compro_en_visita', 'mean'),
        visitas_servicio_tecnico=(
            'motivo_visita',
            lambda x: (x == 'servicio_tecnico').sum(),
        ),
        visitas_cotizacion=(
            'motivo_visita',
            lambda x: (x == 'cotizacion').sum(),
        ),
        visitas_reclamo=(
            'motivo_visita',
            lambda x: (x == 'reclamo').sum(),
        ),
        visitas_pago_cuota=(
            'motivo_visita',
            lambda x: (x == 'pago_cuota_credito').sum(),
        ),
        dias_desde_ultima_visita=(
            'fecha_visita',
            lambda x: (fecha_ref - x.max()).days,
        ),
    )
    agg = agg.reset_index()

    logger.info(
        'Features de interacciones: %d clientes, %d features', *agg.shape
    )
    return agg


def build_customer_360(
    clientes_clean: pd.DataFrame,
    transacciones: pd.DataFrame,
    interacciones: pd.DataFrame,
) -> pd.DataFrame:
    """Une todos los conjuntos de features en una tabla Customer 360.

    Combina perfil del cliente, features RFM y features conductuales.
    Imputa con cero los nulos numericos de clientes sin historial.
    Guarda el resultado en data/processed/customer_360.parquet.

    Args:
        clientes_clean: DataFrame de clientes limpio.
        transacciones: DataFrame de transacciones limpio.
        interacciones: DataFrame de interacciones de tienda limpio.

    Returns:
        DataFrame Customer 360 con un registro por cliente.
    """
    tx_feat = build_tx_features(transacciones)
    inter_feat = build_inter_features(interacciones)

    c360 = clientes_clean.merge(tx_feat, on='id_cliente', how='left')
    c360 = c360.merge(inter_feat, on='id_cliente', how='left')

    numeric_cols = c360.select_dtypes(include=[np.number]).columns
    c360[numeric_cols] = c360[numeric_cols].fillna(0)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    c360.to_parquet(os.path.join(PROCESSED_DIR, 'customer_360.parquet'), index=False)
    logger.info('Customer 360: %d clientes, %d features', *c360.shape)
    return c360
