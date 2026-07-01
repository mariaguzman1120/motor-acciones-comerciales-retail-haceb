import json
from datetime import datetime

import pandas as pd

from python.metadata.pipeline_config import (
    INPUT_FILES,
    MIN_TRAINING_ROWS_WARNING,
    PRIMARY_KEYS,
    REQUIRED_COLUMNS,
)


def build_message(level, rule, dataset, details):
    message = {
        "level": level,
        "rule": rule,
        "dataset": dataset,
        "details": details,
    }
    return message


def validate_input_files(input_files=None):
    files = input_files or INPUT_FILES
    messages = []
    for name, path in files.items():
        if not path.exists():
            messages.append(
                build_message("error", "missing_input_file", name, f"No existe {path}")
            )
    return messages


def validate_required_columns(data):
    messages = []
    for name, df in data.items():
        missing_columns = sorted(set(REQUIRED_COLUMNS[name]) - set(df.columns))
        if missing_columns:
            messages.append(
                build_message(
                    "error",
                    "missing_required_columns",
                    name,
                    f"Faltan columnas: {missing_columns}",
                )
            )
    return messages


def validate_primary_keys(data):
    messages = []
    for name, key in PRIMARY_KEYS.items():
        df = data[name]
        if key not in df.columns:
            continue
        null_count = int(df[key].isna().sum())
        duplicate_count = int(df[key].duplicated().sum())
        if null_count:
            messages.append(
                build_message("error", "null_primary_key", name, f"{key}: {null_count}")
            )
        if duplicate_count:
            messages.append(
                build_message(
                    "error",
                    "duplicated_primary_key",
                    name,
                    f"{key}: {duplicate_count}",
                )
            )
    return messages


def validate_business_rules(data):
    messages = []
    clients = data["clients"]
    transactions = data["transactions"]
    interactions = data["interactions"]

    client_ids = set(clients["id_cliente"].dropna())
    orphan_transactions = sorted(set(transactions["id_cliente"].dropna()) - client_ids)
    orphan_interactions = sorted(set(interactions["id_cliente"].dropna()) - client_ids)

    if orphan_transactions:
        messages.append(
            build_message(
                "error",
                "orphan_transaction_clients",
                "transactions",
                f"Clientes no encontrados: {orphan_transactions[:10]}",
            )
        )
    if orphan_interactions:
        messages.append(
            build_message(
                "error",
                "orphan_interaction_clients",
                "interactions",
                f"Clientes no encontrados: {orphan_interactions[:10]}",
            )
        )

    invalid_purchase_dates = int(transactions["fecha_compra"].isna().sum())
    invalid_visit_dates = int(interactions["fecha_visita"].isna().sum())
    invalid_amounts = int(
        transactions["monto_pago"].isna().sum() + (transactions["monto_pago"] < 0).sum()
    )
    invalid_target = int((~interactions["compro_en_visita"].isin([0, 1])).sum())

    if invalid_purchase_dates:
        messages.append(
            build_message(
                "error",
                "invalid_purchase_dates",
                "transactions",
                f"Fechas invalidas: {invalid_purchase_dates}",
            )
        )
    if invalid_visit_dates:
        messages.append(
            build_message(
                "error",
                "invalid_visit_dates",
                "interactions",
                f"Fechas invalidas: {invalid_visit_dates}",
            )
        )
    if invalid_amounts:
        messages.append(
            build_message(
                "error",
                "invalid_amounts",
                "transactions",
                f"Montos nulos o negativos: {invalid_amounts}",
            )
        )
    if invalid_target:
        messages.append(
            build_message(
                "error",
                "invalid_target",
                "interactions",
                f"Valores distintos de 0/1: {invalid_target}",
            )
        )

    missing_cellphones = int(clients["celular_contacto"].isna().sum())
    missing_emails = int(clients["email_contacto"].isna().sum())
    missing_scores = int(clients["score_credito_interno"].isna().sum())
    clients_without_transactions = len(client_ids - set(transactions["id_cliente"].dropna()))
    clients_without_interactions = len(client_ids - set(interactions["id_cliente"].dropna()))

    if missing_cellphones:
        messages.append(
            build_message(
                "warning",
                "missing_cellphones",
                "clients",
                f"Clientes sin celular: {missing_cellphones}",
            )
        )
    if missing_emails:
        messages.append(
            build_message(
                "warning",
                "missing_emails",
                "clients",
                f"Clientes sin correo: {missing_emails}",
            )
        )
    if missing_scores:
        messages.append(
            build_message(
                "warning",
                "missing_credit_scores",
                "clients",
                f"Clientes sin score: {missing_scores}",
            )
        )
    if clients_without_transactions:
        messages.append(
            build_message(
                "warning",
                "clients_without_transactions",
                "clients",
                f"Clientes sin transacciones: {clients_without_transactions}",
            )
        )
    if clients_without_interactions:
        messages.append(
            build_message(
                "warning",
                "clients_without_interactions",
                "clients",
                f"Clientes sin interacciones: {clients_without_interactions}",
            )
        )

    return messages


def validate_training_dataset(training_df):
    if len(training_df) < MIN_TRAINING_ROWS_WARNING:
        messages = [
            build_message(
                "warning",
                "small_training_dataset",
                "training_dataset",
                f"Filas disponibles: {len(training_df)}",
            )
        ]
        return messages
    messages = []
    return messages


def raise_for_errors(messages):
    errors = [message for message in messages if message["level"] == "error"]
    if errors:
        details = "\n".join(f"- {item['dataset']}::{item['rule']}: {item['details']}" for item in errors)
        raise ValueError(f"La validacion encontro reglas bloqueantes:\n{details}")


def write_audit_log(path, run_summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **run_summary,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_empty_value_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica valores nulos o textos vacíos en un DataFrame."""
    empty_values = df.isna()
    text_columns = df.select_dtypes(include=['object', 'string']).columns

    if len(text_columns) > 0:
        text_values = df.loc[:, text_columns].astype('string')
        text_values = text_values.replace(r'^\s*$', pd.NA, regex=True)
        empty_values.loc[:, text_columns] = empty_values.loc[:, text_columns] | text_values.isna()

    return empty_values


def summarize_table_information(df: pd.DataFrame) -> pd.DataFrame:
    """Resume cuánta información disponible tiene una fuente."""
    empty_values = build_empty_value_mask(df)
    total_cells = int(df.shape[0] * df.shape[1])
    empty_cells = int(empty_values.to_numpy().sum())
    available_cells = total_cells - empty_cells

    summary_data = {
        'filas': [int(df.shape[0])],
        'columnas': [int(df.shape[1])],
        'celdas_totales': [total_cells],
        'celdas_con_informacion': [available_cells],
        'pct_informacion_disponible': [available_cells / total_cells if total_cells else pd.NA],
        'celdas_nulas_o_vacias': [empty_cells],
        'pct_nulos_o_vacios': [empty_cells / total_cells if total_cells else pd.NA],
        'filas_duplicadas': [int(df.duplicated().sum())],
    }
    summary = pd.DataFrame(summary_data)
    return summary


def summarize_variable_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Resume completitud y cardinalidad por variable."""
    empty_values = build_empty_value_mask(df)
    missing_values = empty_values.sum(axis=0)
    available_values = len(df) - missing_values

    summary_data = {
        'variable': df.columns,
        'tipo': df.dtypes.astype(str).to_numpy(),
        'registros': int(len(df)),
        'con_informacion': available_values.to_numpy(dtype='int64'),
        'pct_con_informacion': available_values.to_numpy(dtype='float64') / len(df),
        'nulos_o_vacios': missing_values.to_numpy(dtype='int64'),
        'pct_nulos_o_vacios': missing_values.to_numpy(dtype='float64') / len(df),
        'valores_unicos': df.nunique(dropna=True).to_numpy(dtype='int64'),
    }
    quality = pd.DataFrame(summary_data)
    quality = quality.sort_values('pct_nulos_o_vacios', ascending=False)
    quality = quality.reset_index(drop=True)
    return quality


def summarize_affected_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo variables con nulos o valores vacíos."""
    quality = summarize_variable_quality(df)
    affected_variables = quality.query('nulos_o_vacios > 0')
    affected_variables = affected_variables.reset_index(drop=True)
    return affected_variables


def summarize_primary_key_quality(df: pd.DataFrame, primary_key: str) -> pd.DataFrame:
    """Resume nulos, vacíos y duplicados de una llave primaria."""
    empty_values = build_empty_value_mask(df)
    key_empty_values = empty_values.loc[:, primary_key]

    summary_data = {
        'llave_primaria': [primary_key],
        'llaves_nulas_o_vacias': [int(key_empty_values.sum())],
        'llaves_duplicadas': [int(df[primary_key].duplicated().sum())],
        'llaves_unicas': [int(df[primary_key].nunique(dropna=True))],
    }
    summary = pd.DataFrame(summary_data)
    return summary


def summarize_contactability(clients: pd.DataFrame) -> pd.DataFrame:
    """Resume disponibilidad de canales digitales de contacto."""
    empty_values = build_empty_value_mask(clients)
    has_cellphone = ~empty_values.loc[:, 'celular_contacto']
    has_email = ~empty_values.loc[:, 'email_contacto']
    has_both_channels = has_cellphone & has_email
    has_any_channel = has_cellphone | has_email
    has_no_digital_contact = ~has_any_channel

    summary_data = {
        'indicador': [
            'clientes_con_celular',
            'clientes_con_correo',
            'clientes_con_ambos_canales',
            'clientes_con_al_menos_un_canal',
            'clientes_sin_contacto_digital',
        ],
        'clientes': [
            int(has_cellphone.sum()),
            int(has_email.sum()),
            int(has_both_channels.sum()),
            int(has_any_channel.sum()),
            int(has_no_digital_contact.sum()),
        ],
        'pct_clientes': [
            has_cellphone.mean(),
            has_email.mean(),
            has_both_channels.mean(),
            has_any_channel.mean(),
            has_no_digital_contact.mean(),
        ],
    }
    summary = pd.DataFrame(summary_data)
    return summary


def summarize_transaction_business_quality(
    transactions: pd.DataFrame,
    clients: pd.DataFrame,
) -> pd.DataFrame:
    """Resume validaciones de negocio para transacciones."""
    invalid_dates = transactions['fecha_compra'].isna()
    null_amounts = transactions['monto_pago'].isna()
    negative_amounts = transactions['monto_pago'] < 0

    summary_data = {
        'validacion': [
            'fechas_compra_invalidas',
            'montos_nulos',
            'montos_negativos',
            'clientes_unicos_en_transacciones',
            'tiendas_unicas_en_transacciones',
            'categorias_unicas',
        ],
        'registros': [
            int(invalid_dates.sum()),
            int(null_amounts.sum()),
            int(negative_amounts.sum()),
            int(transactions['id_cliente'].nunique()),
            int(transactions['id_tienda'].nunique()),
            int(transactions['categoria_producto'].nunique()),
        ],
        'porcentaje': [
            invalid_dates.mean(),
            null_amounts.mean(),
            negative_amounts.mean(),
            transactions['id_cliente'].nunique() / len(clients),
            pd.NA,
            pd.NA,
        ],
    }
    summary = pd.DataFrame(summary_data)
    return summary


def summarize_interaction_business_quality(
    interactions: pd.DataFrame,
    clients: pd.DataFrame,
) -> pd.DataFrame:
    """Resume validaciones de negocio para interacciones."""
    invalid_dates = interactions['fecha_visita'].isna()
    null_target = interactions['compro_en_visita'].isna()
    invalid_target = ~interactions['compro_en_visita'].isin([0, 1])

    summary_data = {
        'validacion': [
            'fechas_visita_invalidas',
            'objetivo_nulo',
            'objetivo_fuera_de_0_1',
            'clientes_unicos_en_interacciones',
            'tiendas_unicas_en_interacciones',
            'motivos_visita_unicos',
            'tasa_conversion_global',
        ],
        'registros': [
            int(invalid_dates.sum()),
            int(null_target.sum()),
            int(invalid_target.sum()),
            int(interactions['id_cliente'].nunique()),
            int(interactions['id_tienda'].nunique()),
            int(interactions['motivo_visita'].nunique()),
            int(interactions['compro_en_visita'].sum()),
        ],
        'porcentaje': [
            invalid_dates.mean(),
            null_target.mean(),
            invalid_target.mean(),
            interactions['id_cliente'].nunique() / len(clients),
            pd.NA,
            pd.NA,
            interactions['compro_en_visita'].mean(),
        ],
    }
    summary = pd.DataFrame(summary_data)
    return summary


def summarize_source_integrity(
    clients: pd.DataFrame,
    transactions: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Resume integridad referencial entre las tres fuentes."""
    client_ids = set(clients['id_cliente'].dropna())
    transaction_client_ids = set(transactions['id_cliente'].dropna())
    interaction_client_ids = set(interactions['id_cliente'].dropna())

    summary_data = {
        'validacion': [
            'clientes_en_transacciones_no_en_clientes',
            'clientes_en_interacciones_no_en_clientes',
            'clientes_sin_transacciones',
            'clientes_sin_interacciones',
            'clientes_con_transacciones_e_interacciones',
        ],
        'clientes': [
            len(transaction_client_ids - client_ids),
            len(interaction_client_ids - client_ids),
            len(client_ids - transaction_client_ids),
            len(client_ids - interaction_client_ids),
            len(transaction_client_ids & interaction_client_ids),
        ],
    }
    integrity = pd.DataFrame(summary_data)
    integrity['pct_sobre_clientes'] = integrity['clientes'] / len(client_ids)
    return integrity
