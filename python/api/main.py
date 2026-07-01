"""Endpoint FastAPI de inferencia para el motor NBA retail."""

import logging
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from python.models.predict import NBAPredictor  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = None


class PredictRequest(BaseModel):
    id_cliente: str
    motivo_visita: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los artefactos del modelo al arrancar la API."""
    global predictor
    predictor = NBAPredictor()
    logger.info('API lista')
    yield


app = FastAPI(title='NBA Retail API', version='1.0.0', lifespan=lifespan)


@app.post('/predict')
def predict(req: PredictRequest) -> dict:
    """Retorna la recomendacion NBA para una visita de cliente.

    Args:
        req: Cuerpo de la solicitud con id_cliente y motivo_visita.

    Returns:
        Dict con la prediccion NBA.

    Raises:
        HTTPException: 404 si el cliente no existe en el Customer 360.
    """
    result = predictor.predict(req.id_cliente, req.motivo_visita)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result


@app.get('/health')
def health() -> dict:
    """Verificacion de disponibilidad de la API."""
    return {'status': 'ok'}
