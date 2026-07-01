import pandas as pd


TEXT_COLUMNS = {
    "clients": ["id_cliente", "zona_geografica", "celular_contacto", "email_contacto"],
    "transactions": [
        "id_transaccion",
        "id_cliente",
        "sku_producto",
        "categoria_producto",
        "id_tienda",
        "medio_pago",
    ],
    "interactions": ["id_interaccion", "id_cliente", "id_tienda", "motivo_visita"],
}


def normalize_text_columns(df, columns):
    clean_df = df.copy()
    for column in columns:
        if column in clean_df.columns:
            clean_df[column] = clean_df[column].astype("string").str.strip().str.lower()
    return clean_df


def clean_clients(df):
    clean_df = normalize_text_columns(df, TEXT_COLUMNS["clients"])
    clean_df["antiguedad_meses"] = pd.to_numeric(clean_df["antiguedad_meses"], errors="coerce")
    clean_df["score_credito_interno"] = pd.to_numeric(
        clean_df["score_credito_interno"], errors="coerce"
    )
    clean_df["has_cellphone"] = clean_df["celular_contacto"].notna().astype(int)
    clean_df["has_email"] = clean_df["email_contacto"].notna().astype(int)
    return clean_df


def clean_transactions(df):
    clean_df = normalize_text_columns(df, TEXT_COLUMNS["transactions"])
    clean_df["fecha_compra"] = pd.to_datetime(clean_df["fecha_compra"], errors="coerce")
    clean_df["monto_pago"] = pd.to_numeric(clean_df["monto_pago"], errors="coerce")
    return clean_df


def clean_interactions(df):
    clean_df = normalize_text_columns(df, TEXT_COLUMNS["interactions"])
    clean_df["fecha_visita"] = pd.to_datetime(clean_df["fecha_visita"], errors="coerce")
    clean_df["compro_en_visita"] = pd.to_numeric(clean_df["compro_en_visita"], errors="coerce")
    return clean_df


def clean_input_data(raw_data):
    clean_data = {
        "clients": clean_clients(raw_data["clients"]),
        "transactions": clean_transactions(raw_data["transactions"]),
        "interactions": clean_interactions(raw_data["interactions"]),
    }
    return clean_data
