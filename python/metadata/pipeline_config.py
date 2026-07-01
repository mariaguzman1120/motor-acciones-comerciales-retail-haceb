from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
PROCESSED_DIR = OUTPUT_DIR / "processed"
MODEL_DIR = OUTPUT_DIR / "model"
BUSINESS_DIR = OUTPUT_DIR / "business"
AUDIT_DIR = OUTPUT_DIR / "audit"

INPUT_FILES = {
    "clients": DATA_DIR / "clientes.csv",
    "transactions": DATA_DIR / "transacciones.csv",
    "interactions": DATA_DIR / "interacciones_tienda.csv",
}

REQUIRED_COLUMNS = {
    "clients": [
        "id_cliente",
        "zona_geografica",
        "antiguedad_meses",
        "celular_contacto",
        "email_contacto",
        "score_credito_interno",
    ],
    "transactions": [
        "id_transaccion",
        "id_cliente",
        "fecha_compra",
        "sku_producto",
        "categoria_producto",
        "monto_pago",
        "id_tienda",
        "medio_pago",
    ],
    "interactions": [
        "id_interaccion",
        "id_cliente",
        "id_tienda",
        "fecha_visita",
        "motivo_visita",
        "compro_en_visita",
    ],
}

PRIMARY_KEYS = {
    "clients": "id_cliente",
    "transactions": "id_transaccion",
    "interactions": "id_interaccion",
}

NUMERIC_FEATURES = [
    "antiguedad_meses",
    "score_credito_interno",
    "has_cellphone",
    "has_email",
    "total_prior_transactions",
    "total_prior_spend",
    "avg_prior_ticket",
    "days_since_last_purchase",
    "prior_distinct_categories",
    "prior_credit_retail_transactions",
    "prior_same_store_transactions",
]

CATEGORICAL_FEATURES = [
    "zona_geografica",
    "id_tienda",
    "motivo_visita",
]

TARGET_COLUMN = "compro_en_visita"

MODEL_PARAMETERS = {
    "learning_rate": 0.08,
    "iterations": 5000,
    "l2_penalty": 0.02,
    "test_fraction": 0.25,
    "decision_threshold": 0.5,
}

MIN_TRAINING_ROWS_WARNING = 100
HIGH_PRIORITY_QUANTILE = 0.75
MEDIUM_PRIORITY_QUANTILE = 0.45
