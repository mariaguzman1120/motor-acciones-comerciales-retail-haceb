# Pipeline NBA Retail: documentacion tecnica

Descripcion detallada del pipeline de entrenamiento, los modelos, sus variables, la inferencia y el monitoreo del motor de Proxima Mejor Accion.

---

## 1. Vision de la solucion

El motor combina dos preguntas de negocio en dos modelos complementarios:

| Pregunta | Modelo | Tipo | Rol |
| --- | --- | --- | --- |
| Como atender al cliente que llega? | K-Means | No supervisado | Define el perfil y el tono comercial |
| Que ofrecerle y con que urgencia? | LightGBM | Supervisado | Estima la probabilidad de compra en la visita |

Ambos alimentan la misma respuesta al asesor: un guion accionable con nivel de prioridad y categoria a proponer.

---

## 2. Flujo end-to-end

```text
data/raw/
  clientes.csv, transacciones.csv, interacciones_tienda.csv
      |
      v
python/data/preprocessing.py
  load_raw_data + clean_clientes + save_processed
      |
      v
python/features/build_features.py
  build_customer_360 (RFM + conductual + credito)
      |
      v
python/models/train_segmentation.py
  find_optimal_k + train_segmentation (K-Means)
      |
      v
python/models/train_propensity.py
  prepare_model_data + optimize_hyperparams + train_final_model (LightGBM)
      |
      v
python/monitoring/drift_report.py
  generate_drift_report (Evidently)
      |
      v
Artefactos productivos:
  models/*.joblib, data/processed/customer_360.parquet
      |
      v
python/api/main.py (FastAPI)   ->   python/front/app.py (Streamlit)
```

La orquestacion la realiza [python/training_flow_prefect.py](python/training_flow_prefect.py) con Prefect. Cada etapa es una `@task` con logging y observabilidad. El flujo completo se dispara con `python main.py`.

---

## 3. Datos de entrada

Tres CSV crudos en `data/raw/`:

### clientes.csv
Perfil comercial y demografico del cliente.

| Campo | Tipo | Descripcion |
| --- | --- | --- |
| id_cliente | string | Identificador unico |
| zona_geografica | string | Zona de residencia (categorica) |
| antiguedad_meses | int | Meses desde alta como cliente |
| score_credito_interno | float | Score de credito propio, con nulos |
| celular_contacto | string | Celular con calidad baja |
| email_contacto | string | Email con calidad baja |

### transacciones.csv
Historia de compras del cliente.

| Campo | Tipo | Descripcion |
| --- | --- | --- |
| id_transaccion | string | Identificador unico |
| id_cliente | string | Cliente que compra |
| fecha_compra | datetime | Fecha de la transaccion |
| categoria_producto | string | Refrigeracion, lavado, coccion, pequenos_electrodomesticos |
| monto_pago | float | Monto en pesos |
| medio_pago | string | Efectivo, tarjeta, credito propio del retail |

### interacciones_tienda.csv
Cada visita a la sala de exhibicion.

| Campo | Tipo | Descripcion |
| --- | --- | --- |
| id_interaccion | string | Identificador unico |
| id_cliente | string | Cliente que visita |
| id_tienda | string | Punto de venta |
| fecha_visita | datetime | Fecha de la visita |
| motivo_visita | string | Cotizacion, pago cuota, servicio tecnico, reclamo |
| compro_en_visita | int | Target (1 = compro, 0 = no compro) |

---

## 4. Limpieza de datos

Modulo: [python/data/preprocessing.py](python/data/preprocessing.py)

### load_raw_data
Carga los tres CSV con parsing de fechas en `fecha_compra` y `fecha_visita`. No hace transformacion.

### clean_clientes
Tres decisiones concretas para el dataset de clientes:

1. **Contacto digital de baja calidad** (indicado en el enunciado como limitacion real del negocio):
   - `celular_contacto` -> `tiene_celular_valido` con regex `^\d{10}$`
   - `email_contacto` -> `tiene_email_valido` con regex de email basico
   - Los campos originales se eliminan (no aportan como texto, solo como flag).
2. **Imputacion de `score_credito_interno`**:
   - Mediana por `zona_geografica` (vecinos geograficos son mejor referencia que la media global).
   - Fallback: mediana global si toda la zona esta vacia.
3. **Persistencia**: guarda en `data/processed/*_clean.parquet` con `save_processed`.

Motivacion: el pipeline no depende de contacto digital pero conserva su calidad como feature del modelo (los clientes con datos digitales completos suelen ser mas activos).

---

## 5. Ingenieria de caracteristicas: Customer 360

Modulo: [python/features/build_features.py](python/features/build_features.py)

Construye una tabla con un registro por cliente que combina todas las senales disponibles.

### 5.1 Features de transacciones (build_tx_features)

Agregacion RFM extendida por `id_cliente`:

| Feature | Definicion | Uso comercial |
| --- | --- | --- |
| recency_dias | Dias desde la ultima compra | Recencia RFM |
| frequency | Numero total de transacciones | Frecuencia RFM |
| monetary_total | Suma de todos los tickets | Valor total del cliente |
| monetary_avg | Ticket promedio | Poder adquisitivo |
| ticket_max | Ticket maximo historico | Potencial premium |
| diversidad_categorias | Numero de categorias distintas compradas | Cross-sell propensity |
| categoria_favorita | Moda de categorias compradas | Ancla para NBA |
| medio_pago_habitual | Moda de medios de pago | Perfil de pago |
| usa_credito_propio | Flag: alguna vez pago con credito Haceb | Segmentacion financiera |

### 5.2 Features de interacciones (build_inter_features)

Agregacion conductual por `id_cliente`:

| Feature | Definicion | Uso comercial |
| --- | --- | --- |
| total_visitas | Numero de visitas historicas | Engagement |
| tasa_conversion_hist | Compras / visitas | Efectividad del asesor con este cliente |
| visitas_servicio_tecnico | Visitas por servicio tecnico | Cliente con equipos en uso |
| visitas_cotizacion | Visitas por cotizacion | Cliente en fase de decision |
| visitas_reclamo | Visitas por reclamo | Cliente con friccion |
| visitas_pago_cuota | Visitas por pago de credito | Cliente financiado |
| dias_desde_ultima_visita | Ventana desde la ultima visita | Recencia de engagement |

### 5.3 Union en Customer 360

`build_customer_360` hace `left join` de clientes + transacciones + interacciones. Los clientes sin historial de transacciones o visitas quedan con nulos numericos que se imputan con `0` (interpretable como "sin historia registrada"). El resultado se persiste en `data/processed/customer_360.parquet`.

Total: **30 features** por cliente disponibles para ambos modelos.

---

## 6. Modelo 1: K-Means (segmentacion no supervisada)

Modulo: [python/models/train_segmentation.py](python/models/train_segmentation.py)

### 6.1 Que hace
Agrupa a los clientes en **5 segmentos comerciales** con base en su comportamiento historico. Cada segmento define el tono y estilo de la conversacion en tienda.

### 6.2 Como funciona
- Algoritmo: `sklearn.cluster.KMeans` con `n_init=10`, `random_state=42`.
- Preprocesamiento: `StandardScaler` para llevar todas las features a media 0 y varianza 1 (K-Means es sensible a escalas).
- Seleccion de k: `find_optimal_k` evalua `k in range(2, 8)` con dos metricas:
  - **Inertia** (suma de distancias intra-cluster)
  - **Silhouette Score** (cohesion vs. separacion, [-1, 1])
- El flujo elige el `k` con mayor Silhouette. En la corrida productiva **k=5**.

### 6.3 Variables de entrada (10 features)

Constante `SEGMENT_FEATURES` en el modulo:

```python
SEGMENT_FEATURES = [
    'recency_dias', 'frequency', 'monetary_total', 'monetary_avg',
    'total_visitas', 'tasa_conversion_hist', 'usa_credito_propio',
    'diversidad_categorias', 'score_credito_interno', 'antiguedad_meses',
]
```

Todas provienen del Customer 360. Se seleccionaron por representar los ejes comerciales relevantes: valor, recencia, engagement, credito y antiguedad.

### 6.4 Salida
- `segmento_id` (int, 0-4) para cada cliente.
- `segmento_nombre` (string) mapeado por `SEGMENT_NAMES`.

### 6.5 Nombres de segmentos

| ID | Nombre | Descripcion |
| --- | --- | --- |
| 0 | VIP Comprador | Alto ticket historico, baja sensibilidad a precio. Ofrecer gama premium. |
| 1 | Cliente Financiado | Usa credito propio. Responde a planes en cuotas. |
| 2 | Visitante Servicios | Cliente activo con equipos en uso. Alta probabilidad de recompra. |
| 3 | Cliente Dormido | Sin transacciones recientes. Requiere reactivacion. |
| 4 | Nuevo Potencial | Base incipiente. Construir relacion antes de cerrar venta. |

### 6.6 Artefactos persistidos
- `models/kmeans_segmentation.joblib`: modelo `KMeans` ajustado.
- `models/scaler_segmentation.joblib`: `StandardScaler` para transformar features nuevas en inferencia.

### 6.7 Metricas de evaluacion
- **Silhouette Score**: metrica primaria de seleccion.
- **Inertia**: como validacion secundaria (curva del codo).
- Registro por consola durante el entrenamiento.

---

## 7. Modelo 2: LightGBM (propension supervisada)

Modulo: [python/models/train_propensity.py](python/models/train_propensity.py)

### 7.1 Que hace
Predice la probabilidad de que el cliente **compre en la visita actual** dado su perfil y el motivo de visita. La salida se traduce en un nivel de prioridad (ALTO / MEDIO / BAJO) que orienta el esfuerzo del asesor.

### 7.2 Como funciona
- Algoritmo: `LGBMClassifier` (Gradient Boosting sobre arboles).
- HPO: `optuna.create_study(direction='maximize')` con TPE bayesiano.
- Validacion: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Metrica de optimizacion: `roc_auc` (`cross_val_score`).
- Encoder de categoricas: `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` para robustez ante categorias nuevas en inferencia.

### 7.3 Preparacion de datos (prepare_model_data)

Une el Customer 360 con la tabla de interacciones. **Una fila por visita**, no por cliente. Enriquece con features temporales derivadas de `fecha_visita`:
- `dia_semana` (0-6)
- `mes` (1-12)
- `es_quincena` (1 si el dia esta en [14,15,16,28,29,30,31])

Target: `compro_en_visita` (binario).

### 7.4 Variables de entrada (20 features)

Constante `FEATURE_COLS`:

```python
FEATURE_COLS = [
    # Perfil del cliente
    'zona_geografica', 'antiguedad_meses', 'score_credito_interno',
    'tiene_celular_valido', 'tiene_email_valido',
    # RFM
    'recency_dias', 'frequency', 'monetary_total', 'monetary_avg',
    'diversidad_categorias',
    # Conductual
    'total_visitas', 'tasa_conversion_hist',
    'visitas_servicio_tecnico', 'visitas_cotizacion', 'visitas_reclamo',
    # Contexto de la visita
    'motivo_visita', 'id_tienda', 'dia_semana', 'mes', 'es_quincena',
]
```

Variables categoricas codificadas con OrdinalEncoder: `zona_geografica`, `motivo_visita`, `id_tienda`.

### 7.5 Espacio de busqueda de hiperparametros

En `optimize_hyperparams`:

| Hiperparametro | Rango | Distribucion |
| --- | --- | --- |
| n_estimators | 50 - 300 | Uniforme entero |
| learning_rate | 0.01 - 0.3 | Log-uniforme |
| max_depth | 2 - 6 | Uniforme entero |
| min_child_samples | 10 - 60 | Uniforme entero |
| subsample | 0.5 - 1.0 | Uniforme |
| colsample_bytree | 0.5 - 1.0 | Uniforme |
| reg_alpha | 1e-3 - 10.0 | Log-uniforme |
| reg_lambda | 1e-3 - 10.0 | Log-uniforme |

Ejecuta **50 trials** por defecto (configurable en `training_flow`).

### 7.6 Salida
- `predict_proba(X)[:, 1]`: probabilidad de compra en la visita [0, 1].
- Nivel derivado por umbrales:
  - `> 0.60` -> ALTO
  - `> 0.35` -> MEDIO
  - `<= 0.35` -> BAJO

### 7.7 Artefactos persistidos
- `models/lgbm_propension.joblib`: modelo `LGBMClassifier` ajustado.
- `models/ordinal_encoder.joblib`: encoder de categoricas ajustado.

### 7.8 Metricas de evaluacion
- **AUC-ROC** con 5-fold CV (metrica primaria).
- **Media y desviacion estandar** del AUC entre folds (para estabilidad).
- **Lift decil top** vs. baseline (calculado offline).

En la corrida productiva: **AUC = 0.889 (+/- 0.020)**.

---

## 8. Inferencia en tiempo real

Modulo: [python/models/predict.py](python/models/predict.py)

### 8.1 NBAPredictor

Clase que carga todos los artefactos al iniciarse y expone un metodo `predict(id_cliente, motivo_visita)`.

Artefactos cargados en memoria:
- `models/lgbm_propension.joblib`
- `models/kmeans_segmentation.joblib`
- `models/scaler_segmentation.joblib`
- `models/ordinal_encoder.joblib`
- `data/processed/customer_360.parquet`

### 8.2 Flujo del metodo predict

1. Busca al cliente en el Customer 360 (retorna error si no existe).
2. Aplica `StandardScaler` -> `KMeans` -> segmento.
3. Construye features en tiempo real (`_build_realtime_features`): combina perfil + fecha actual + motivo de visita.
4. Aplica `OrdinalEncoder` -> `LGBMClassifier` -> score.
5. Convierte score en nivel ALTO / MEDIO / BAJO.
6. Determina `next_best_category` con la tabla `CATEGORY_TRANSITIONS` (regla de negocio: rotacion entre las 4 categorias del catalogo).
7. Genera `accion_recomendada` con `_build_action` combinando segmento + categoria + nivel.

### 8.3 Contrato de salida

```json
{
  "id_cliente": "cli_001",
  "segmento": "VIP Comprador",
  "score_compra_hoy": 0.72,
  "nivel": "ALTO",
  "next_best_category": "lavado",
  "categoria_favorita": "refrigeracion",
  "accion_recomendada": "Ofrecer lavado premium con garantia extendida"
}
```

### 8.4 API

Modulo: [python/api/main.py](python/api/main.py)

- Framework: FastAPI + Uvicorn.
- Los artefactos se cargan en el `lifespan` async al arrancar (no en cada request).
- Contratos con `pydantic.BaseModel`.
- Endpoints:
  - `POST /predict` con body `{id_cliente, motivo_visita}`.
  - `GET /health` para verificacion de disponibilidad.
- Errores: `HTTPException 404` si el cliente no existe en el Customer 360.

### 8.5 Latencia
- Inferencia en memoria (sin llamadas a BD).
- **p95 < 100 ms** en pruebas locales.

---

## 9. Monitoreo y seguimiento

### 9.1 Tracking de experimentos: MLflow

Modulo: `train_propensity.py::train_final_model`

Cada corrida del pipeline registra automaticamente en `mlflow.db` un run bajo el experimento `propension_compra_visita`:

| Elemento | Que se loguea |
| --- | --- |
| `log_params(best_params)` | Los mejores hiperparametros encontrados por Optuna |
| `log_metric('auc_cv_mean')` | AUC promedio de los 5 folds |
| `log_metric('auc_cv_std')` | Desviacion estandar del AUC entre folds |
| `log_model(model, 'modelo_propension')` | Modelo LightGBM completo con signature y trusted types |

Consulta la UI con:
```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Permite comparar experimentos, ver evolucion de metricas y descargar modelos previos.

### 9.2 Data drift: Evidently

Modulo: [python/monitoring/drift_report.py](python/monitoring/drift_report.py)

Detecta si la distribucion de las features cambia entre dos ventanas temporales (referencia vs. actual).

- Preset: `DataDriftPreset` (Evidently detecta drift con test estadistico apropiado por tipo de columna: KS para numericas, chi2 para categoricas).
- El flujo Prefect divide el dataset en dos mitades cronologicas (primera mitad como referencia, segunda como actual) y genera el reporte automaticamente.
- Salida: `reports/drift_report_prefect.html` con visualizaciones interactivas por feature.

En produccion real, la referencia seria el snapshot del dataset de entrenamiento y `current` seria la ventana movil de la ultima semana / mes.

### 9.3 Observabilidad del pipeline: Prefect

Modulo: [python/training_flow_prefect.py](python/training_flow_prefect.py)

Cada etapa esta decorada con `@task`. Al ejecutar el flujo con Prefect corriendo:
- Se visualiza el DAG completo en la UI.
- Se ve el estado de cada tarea (Success / Failed / Running).
- Logs correlacionados por task y por flow run.
- Reintentos y timeouts configurables.

Fallback: si Prefect no esta instalado, el modulo define `task` y `flow` como decoradores no-op y el pipeline se ejecuta secuencialmente sin observabilidad avanzada.

### 9.4 Logging aplicativo

Cada modulo Python instancia `logging.getLogger(__name__)` con nivel INFO. Los eventos clave (carga de datos, seleccion de k, silhouette, AUC por fold, latencia de tareas) quedan en la salida estandar y son capturados por Prefect cuando esta activo.

---

## 10. Artefactos generados por una corrida completa

```text
data/processed/
    clientes_clean.parquet             clientes limpios
    transacciones_clean.parquet        transacciones limpias
    interacciones_clean.parquet        interacciones limpias
    customer_360.parquet               Customer 360 con segmento

models/
    kmeans_segmentation.joblib         K-Means ajustado
    scaler_segmentation.joblib         StandardScaler asociado
    ordinal_encoder.joblib             OrdinalEncoder de categoricas
    lgbm_propension.joblib             LightGBM ajustado

reports/
    drift_report_prefect.html          reporte de drift Evidently

mlflow.db                              tracking de experimentos
mlruns/                                artefactos de MLflow
```

---

## 11. Ejecucion desde cero

Requisitos previos: entorno con `requirements.txt` instalado y los CSV crudos en `data/raw/`.

```powershell
# 1. Activa el entorno
.\.venv\Scripts\Activate.ps1

# 2. Ejecuta el pipeline completo (carga -> entrenamiento -> drift)
python main.py

# 3. Levanta los tres servicios en paralelo
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
python -m uvicorn python.api.main:app --reload --port 8000
python -m streamlit run python/front/app.py --server.port 8501
```

Luego abre:
- MLflow UI: http://localhost:5000
- API docs (Swagger): http://localhost:8000/docs
- Dashboard del asesor: http://localhost:8501

## 12. Referencias cruzadas

- Slide ejecutivo one-page: [docs/index.html](docs/index.html)
- README: [README.md](README.md)
- Enunciado original: [docs/Prueba tecnica proceso de seleccion.pdf](docs/Prueba%20t%C3%A9cnica%20proceso%20de%20selecci%C3%B3n.pdf)
