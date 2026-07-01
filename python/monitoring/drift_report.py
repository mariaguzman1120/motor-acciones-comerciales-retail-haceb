"""Generacion de reportes de drift con Evidently AI para monitoreo."""

import logging
import os

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_name: str = 'drift_report.html',
) -> str:
    """Genera un reporte HTML de drift entre datos de referencia y actuales.

    Args:
        reference: Dataset de referencia (ej. datos de entrenamiento).
        current: Datos recientes a comparar contra la referencia.
        output_name: Nombre del archivo HTML guardado en reports/.

    Returns:
        Ruta al reporte HTML generado.
    """
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_path = os.path.join(REPORTS_DIR, output_name)
    snapshot.save_html(output_path)
    logger.info('Reporte de drift guardado en %s', output_path)
    return output_path
