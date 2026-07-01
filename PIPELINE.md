# Pipeline NBA Retail: documentacion tecnica

Este documento describe el pipeline de entrenamiento,
inferencia y monitoreo del motor de proxima mejor accion.

## 1. Arquitectura de modelaje

La solucion combina dos modelos complementarios:

### Modelo de Clustering (K-Means)
Responde el COMO atender al cliente:
- Define el perfil de comportamiento y la psicologia de compra
- Clasifica en segmentos: VIP Comprador, Cliente Financiado, Visitante Servicios, Cliente Dormido, Nuevo Potencial
- Determina el tono y la estrategia que el asesor debe usar

### Modelo Predictivo (LightGBM)
Responde el QUE ofrecerle:
- Calcula la probabilidad de compra en la visita actual
- Define la categoria exacta de electrodomestico con mayor propension
- Optimiza el inventario exhibido y el esfuerzo del asesor

## 2. Componentes principales

```text
python/data/preprocessing.py
    - load_raw_data
    - clean_clientes
    - save_processed

python/features/build_features.py
    - build_customer_360

python/models/train_segmentation.py
    - find_optimal_k
    - train_segmentation

python/models/train_propensity.py
    - prepare_model_data
    - optimize_hyperparams
    - train_final_model

python/monitoring/drift_report.py
    - generate_drift_report

python/models/predict.py
    - NBAPredictor

python/training_flow_prefect.py
    - training_flow

main.py
    - run_training_flow
```

## 3. Flujo de entrenamiento

La secuencia ejecutada por el flujo Prefect es:

1. `load_raw_data_task`
2. `clean_clientes_task`
3. `save_processed_task`
4. `build_customer_360_task`
5. `find_best_k_task`
6. `train_segmentation_task`
7. `prepare_model_data_task`
8. `optimize_hyperparams_task`
9. `train_final_model_task`
10. `generate_drift_report_task` si esta habilitado

## 4. Artefactos generados

Una corrida exitosa genera:

```text
data/processed/
    clientes_clean.parquet
    transacciones_clean.parquet
    interacciones_clean.parquet
    customer_360.parquet

models/
    kmeans_segmentation.joblib
    scaler_segmentation.joblib
    ordinal_encoder.joblib
    lgbm_propension.joblib

reports/
    drift_report_prefect.html

mlflow.db
mlruns/
```

## 5. Inferencia

La API no participa en el entrenamiento. Solo consume:

- `models/*.joblib`
- `data/processed/customer_360.parquet`

`NBAPredictor` carga esos artefactos al iniciar FastAPI y responde predicciones
para `id_cliente` y `motivo_visita`.

## 6. Ejecucion desde cero

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
python -m uvicorn python.api.main:app --reload --port 8000
python -m streamlit run python/front/app.py --server.port 8501
```
