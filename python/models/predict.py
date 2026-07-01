"""Inferencia NBA en tiempo real para asesores en punto de venta."""

import logging
import os

import joblib
import pandas as pd

from python.models.train_segmentation import SEGMENT_FEATURES, SEGMENT_NAMES
from python.models.train_propensity import FEATURE_COLS

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

CATEGORY_TRANSITIONS = {
    'refrigeracion': 'lavado',
    'lavado': 'coccion',
    'coccion': 'pequenos_electrodomesticos',
    'pequenos_electrodomesticos': 'refrigeracion',
}


class NBAPredictor:
    """Carga artefactos del modelo y genera recomendaciones en tienda."""

    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS_DIR, 'lgbm_propension.joblib'))
        self.km = joblib.load(os.path.join(MODELS_DIR, 'kmeans_segmentation.joblib'))
        self.scaler_seg = joblib.load(
            os.path.join(MODELS_DIR, 'scaler_segmentation.joblib')
        )
        self.encoder = joblib.load(os.path.join(MODELS_DIR, 'ordinal_encoder.joblib'))
        self.c360 = pd.read_parquet(os.path.join(PROCESSED_DIR, 'customer_360.parquet'))
        logger.info('NBAPredictor inicializado')

    def predict(self, id_cliente: str, motivo_visita: str) -> dict:
        """Genera una recomendacion NBA para la visita de un cliente.

        Args:
            id_cliente: Identificador del cliente (ej. 'cli_043').
            motivo_visita: Motivo de la visita actual.

        Returns:
            Dict con segmento, score_compra_hoy, nivel, next_best_category
            y accion_recomendada. Retorna {'error': ...} si no existe.
        """
        cliente = self.c360[self.c360['id_cliente'] == id_cliente]
        if cliente.empty:
            return {'error': f'Cliente {id_cliente} no encontrado'}

        cliente = cliente.iloc[0]

        seg_features = pd.DataFrame([cliente[SEGMENT_FEATURES]])
        seg_features = seg_features.fillna(0)
        seg_features = seg_features.infer_objects(copy=False)
        seg_scaled = self.scaler_seg.transform(seg_features)
        segmento_id = int(self.km.predict(seg_scaled)[0])
        segmento_nombre = SEGMENT_NAMES.get(
            segmento_id, f'Segmento {segmento_id}'
        )

        features = self._build_realtime_features(cliente, motivo_visita)
        score = float(self.model.predict_proba(features)[0][1])

        cat_favorita = cliente.get('categoria_favorita', 'refrigeracion')
        next_cat = CATEGORY_TRANSITIONS.get(cat_favorita, 'refrigeracion')
        nivel = 'ALTO' if score > 0.6 else 'MEDIO' if score > 0.35 else 'BAJO'

        return {
            'id_cliente': id_cliente,
            'segmento': segmento_nombre,
            'score_compra_hoy': round(score, 3),
            'nivel': nivel,
            'next_best_category': next_cat,
            'categoria_favorita': cat_favorita,
            'accion_recomendada': self._build_action(
                segmento_nombre, next_cat, nivel
            ),
        }

    def _build_realtime_features(
        self, cliente: pd.Series, motivo_visita: str
    ) -> pd.DataFrame:
        """Construye el vector de features para una visita en tiempo real.

        Args:
            cliente: Fila unica del Customer 360.
            motivo_visita: Motivo de visita capturado en la entrada.

        Returns:
            DataFrame de una fila alineado a FEATURE_COLS.
        """
        row = {col: 0 for col in FEATURE_COLS}
        for col in FEATURE_COLS:
            if col in cliente.index:
                row[col] = cliente[col]

        row['motivo_visita'] = motivo_visita
        row['dia_semana'] = pd.Timestamp.now().dayofweek
        row['mes'] = pd.Timestamp.now().month
        quincena = [14, 15, 16, 28, 29, 30, 31]
        row['es_quincena'] = int(pd.Timestamp.now().day in quincena)

        df = pd.DataFrame([row])
        cat_cols = ['zona_geografica', 'motivo_visita', 'id_tienda']
        for col in cat_cols:
            df[col] = df[col].astype(str)
        df[cat_cols] = self.encoder.transform(df[cat_cols])

        return df[FEATURE_COLS]

    def _build_action(self, segmento: str, next_cat: str, nivel: str) -> str:
        """Mapea segmento y categoria a un argumento de venta para el asesor.

        Args:
            segmento: Nombre del segmento del cliente.
            next_cat: Categoria de producto recomendada.
            nivel: Nivel de propension (ALTO / MEDIO / BAJO).

        Returns:
            Texto de accion legible para el asesor en tienda.
        """
        vip = f'Ofrecer {next_cat} premium con garantia extendida'
        financiado = f'Mostrar {next_cat} con financiacion a 24 cuotas'
        servicios = f'Ofrecer descuento en {next_cat} por fidelidad'
        dormido = f'Promocion especial de reactivacion en {next_cat}'
        potencial = f'Presentar catalogo completo de {next_cat}'
        acciones = {
            'VIP Comprador': vip,
            'Cliente Financiado': financiado,
            'Visitante Servicios': servicios,
            'Cliente Dormido': dormido,
            'Nuevo Potencial': potencial,
        }
        base = acciones.get(segmento, f'Recomendar {next_cat}')
        if nivel == 'BAJO':
            base += ' - prioridad baja, no presionar'
        return base
