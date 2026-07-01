# Prueba tecnica Haceb - Ciencia de Datos Monetizacion

Este proyecto resuelve una prueba tecnica para estructurar una estrategia analitica de monetizacion en tiendas fisicas a partir de tres fuentes:

- `data/clientes.csv`
- `data/transacciones.csv`
- `data/interacciones_tienda.csv`

El punto de entrada operativo es `main.py`. El pipeline valida los insumos, limpia y tipifica datos, construye variables historicas, entrena candidatos iniciales y genera outputs accionables para asesores comerciales en punto de venta.

La propuesta objetivo va mas alla de un modelo de propension: plantea un sistema de Next Best Action para retail fisico, con calidad de datos, modelos champion/challenger, recomendacion de categoria/servicio, experimentacion, despliegue gratuito y monitoreo de decaimiento.

## Ejecucion

```powershell
python main.py
```

## Preparacion del entorno

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Estructura del repositorio

- `data/`: insumos base entregados para la prueba.
- `docs/`: explicacion del problema, estrategia y entregables ejecutivos.
- `notebooks/`: exploracion y analisis complementario.
- `python/`: modulos del pipeline de validacion, limpieza, features, modelado y salida comercial.
- `output/`: artefactos generados por la ejecucion del pipeline.
- `main.py`: punto de entrada para ejecutar la solucion.

## Salidas principales

- `output/processed/training_dataset.csv`: base modelable de visitas con variables historicas.
- `output/model/model_metrics.json`: metricas del modelo.
- `output/model/model_coefficients.csv`: coeficientes interpretables.
- `output/business/store_activation_opportunities.csv`: oportunidades priorizadas por cliente y tienda.
- `output/audit/audit_log.json`: trazabilidad de la corrida.

## Documentacion

- `docs/contexto_problema.md`: contexto completo del caso y contrato minimo de datos.
- `docs/estrategia_avanzada.md`: estrategia moderna de ciencia de datos, MLOps, despliegue y monitoreo.
- `docs/solucion_analitica.md`: solucion propuesta, arquitectura y reglas de despliegue.
- `docs/slide_ejecutivo.md`: one page ejecutivo para explicar la activacion comercial.
