# Motor de Acciones Comerciales Retail - Haceb

Sistema de Next Best Action (NBA) para retail fisico que combina dos modelos complementarios:

- **Modelo de Clustering (K-Means)**: responde el COMO atender al cliente. Define el perfil de comportamiento, la psicologia de compra (VIP, cazador de ofertas, en riesgo) y el tono que el asesor debe usar.
- **Modelo Predictivo (LightGBM)**: responde el QUE ofrecerle. Define la categoria exacta de electrodomestico o servicio con mayor probabilidad de compra en esa visita.

## Preparacion del entorno

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecucion del pipeline de entrenamiento

```powershell
python main.py
```

Ejecuta el flujo completo orquestado con Prefect:

1. Carga y limpieza de datos crudos
2. Construccion del Customer 360 (features RFM + conductuales)
3. Segmentacion K-Means con seleccion automatica de k
4. Optimizacion de hiperparametros con Optuna (50 trials)
5. Entrenamiento LightGBM con tracking en MLflow
6. Reporte de drift con Evidently

## Levantar los servicios

```powershell
# Terminal 1: MLflow UI
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Terminal 2: API de inferencia
python -m uvicorn python.api.main:app --reload --port 8000

# Terminal 3: Dashboard para asesores
python -m streamlit run python/front/app.py --server.port 8501
```

## Estructura del repositorio

```
data/
  raw/                          <- datos originales (clientes, transacciones, interacciones)
  processed/                    <- outputs limpios en parquet (generados por el pipeline)

python/
  data/preprocessing.py         <- carga, limpieza, flags de calidad, guardado en parquet
  features/build_features.py    <- Customer 360: RFM extendido + behavioral + contexto
  models/
    train_segmentation.py       <- K-Means, seleccion de k, persistencia de artefactos
    train_propensity.py         <- LightGBM + Optuna + MLflow tracking
    predict.py                  <- NBAPredictor: inferencia en tiempo real
  api/main.py                   <- FastAPI endpoint /predict
  front/app.py                  <- Streamlit dashboard para el asesor
  monitoring/drift_report.py    <- reporte de drift con Evidently
  training_flow_prefect.py      <- orquestacion Prefect del pipeline completo

models/                         <- artefactos serializados (.joblib)
reports/                        <- reportes HTML de drift
notebooks/                      <- exploracion y analisis complementario
docs/                           <- documentacion funcional y tecnica

main.py                         <- punto de entrada para ejecutar la solucion
requirements.txt                <- dependencias del proyecto
```

## Salidas principales

- `data/processed/customer_360.parquet`: tabla unificada por cliente con features RFM y conductuales
- `models/kmeans_segmentation.joblib`: modelo de segmentacion entrenado
- `models/lgbm_propension.joblib`: modelo de propension a compra
- `reports/drift_report_prefect.html`: reporte de drift Evidently
- `mlflow.db`: registro de experimentos y metricas

## Resultados obtenidos

- AUC-ROC: ~0.89 (LightGBM + Optuna)
- Lift decil top: 1.8x vs baseline
- Pipeline end-to-end validado: datos crudos -> prediccion accionable
