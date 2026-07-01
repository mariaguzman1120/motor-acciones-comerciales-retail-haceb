import numpy as np
import pandas as pd


def _empty_prior_features(interactions):
    prior_features = pd.DataFrame(
        {
            "id_interaccion": interactions["id_interaccion"],
            "total_prior_transactions": 0,
            "total_prior_spend": 0.0,
            "avg_prior_ticket": 0.0,
            "last_purchase_date": pd.NaT,
            "prior_distinct_categories": 0,
            "prior_credit_retail_transactions": 0,
            "prior_same_store_transactions": 0,
        }
    )
    return prior_features


def build_visit_training_dataset(clients, transactions, interactions):
    base = interactions.merge(clients, on="id_cliente", how="left")

    prior_raw = interactions[["id_interaccion", "id_cliente", "id_tienda", "fecha_visita"]].merge(
        transactions,
        on="id_cliente",
        how="left",
        suffixes=("_visit", "_transaction"),
    )
    prior_raw = prior_raw[prior_raw["fecha_compra"] < prior_raw["fecha_visita"]].copy()

    if prior_raw.empty:
        prior_features = _empty_prior_features(interactions)
    else:
        prior_raw["is_credit_retail"] = (
            prior_raw["medio_pago"].eq("credito_propio_retail").astype(int)
        )
        prior_raw["is_same_store"] = (
            prior_raw["id_tienda_transaction"].eq(prior_raw["id_tienda_visit"]).astype(int)
        )
        prior_features = (
            prior_raw.groupby("id_interaccion")
            .agg(
                total_prior_transactions=("id_transaccion", "count"),
                total_prior_spend=("monto_pago", "sum"),
                avg_prior_ticket=("monto_pago", "mean"),
                last_purchase_date=("fecha_compra", "max"),
                prior_distinct_categories=("categoria_producto", "nunique"),
                prior_credit_retail_transactions=("is_credit_retail", "sum"),
                prior_same_store_transactions=("is_same_store", "sum"),
            )
            .reset_index()
        )

    dataset = base.merge(prior_features, on="id_interaccion", how="left")
    dataset["days_since_last_purchase"] = (
        dataset["fecha_visita"] - dataset["last_purchase_date"]
    ).dt.days

    fill_values = {
        "total_prior_transactions": 0,
        "total_prior_spend": 0.0,
        "avg_prior_ticket": 0.0,
        "prior_distinct_categories": 0,
        "prior_credit_retail_transactions": 0,
        "prior_same_store_transactions": 0,
        "days_since_last_purchase": 9999,
    }
    dataset = dataset.fillna(value=fill_values)
    return dataset


def _mode_or_missing(series):
    mode = series.dropna().mode()
    if mode.empty:
        value = "sin_historial"
        return value
    value = mode.iloc[0]
    return value


def build_customer_snapshot(clients, transactions, interactions):
    max_transaction_date = transactions["fecha_compra"].max()
    max_visit_date = interactions["fecha_visita"].max()
    reference_date = max(max_transaction_date, max_visit_date) + pd.Timedelta(days=1)

    transaction_features = (
        transactions.groupby("id_cliente")
        .agg(
            total_prior_transactions=("id_transaccion", "count"),
            total_prior_spend=("monto_pago", "sum"),
            avg_prior_ticket=("monto_pago", "mean"),
            last_purchase_date=("fecha_compra", "max"),
            prior_distinct_categories=("categoria_producto", "nunique"),
            favorite_category=("categoria_producto", _mode_or_missing),
            favorite_store_from_purchase=("id_tienda", _mode_or_missing),
            prior_credit_retail_transactions=(
                "medio_pago",
                lambda values: int(values.eq("credito_propio_retail").sum()),
            ),
        )
        .reset_index()
    )

    visit_features = (
        interactions.groupby("id_cliente")
        .agg(
            total_visits=("id_interaccion", "count"),
            converted_visits=("compro_en_visita", "sum"),
            last_visit_date=("fecha_visita", "max"),
            favorite_store_from_visit=("id_tienda", _mode_or_missing),
        )
        .reset_index()
    )

    snapshot = clients.merge(transaction_features, on="id_cliente", how="left").merge(
        visit_features, on="id_cliente", how="left"
    )
    snapshot["days_since_last_purchase"] = (
        reference_date - snapshot["last_purchase_date"]
    ).dt.days

    fill_values = {
        "total_prior_transactions": 0,
        "total_prior_spend": 0.0,
        "avg_prior_ticket": 0.0,
        "prior_distinct_categories": 0,
        "prior_credit_retail_transactions": 0,
        "days_since_last_purchase": 9999,
        "total_visits": 0,
        "converted_visits": 0,
    }
    snapshot = snapshot.fillna(value=fill_values)
    snapshot["conversion_rate_history"] = np.where(
        snapshot["total_visits"] > 0,
        snapshot["converted_visits"] / snapshot["total_visits"],
        np.nan,
    )
    snapshot_result = (snapshot, reference_date)
    return snapshot_result


def get_global_defaults(transactions, interactions):
    top_store = interactions["id_tienda"].mode()
    if top_store.empty:
        top_store = transactions["id_tienda"].mode()
    top_category = transactions["categoria_producto"].mode()
    avg_ticket = transactions["monto_pago"].mean()
    defaults = {
        "store": top_store.iloc[0] if not top_store.empty else "tienda_sin_definir",
        "category": top_category.iloc[0] if not top_category.empty else "categoria_sin_definir",
        "avg_ticket": float(avg_ticket) if pd.notna(avg_ticket) else 0.0,
    }
    return defaults
