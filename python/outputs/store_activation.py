import numpy as np
import pandas as pd

from python.features.feature_engineering import build_customer_snapshot, get_global_defaults
from python.metadata.pipeline_config import (
    HIGH_PRIORITY_QUANTILE,
    MEDIUM_PRIORITY_QUANTILE,
)
from python.models.conversion_model import predict_conversion_probability


def _choose_target_store(snapshot, defaults):
    target_store = (
        snapshot["favorite_store_from_visit"]
        .fillna(snapshot["favorite_store_from_purchase"])
        .fillna(defaults["store"])
    )
    return target_store


def _choose_category(snapshot, transactions, defaults):
    purchased = transactions.groupby("id_cliente")["categoria_producto"].apply(set).to_dict()
    global_categories = transactions["categoria_producto"].value_counts().index.tolist()

    recommendations = []
    for customer_id, favorite_category in zip(
        snapshot["id_cliente"], snapshot["favorite_category"].fillna(defaults["category"])
    ):
        customer_categories = purchased.get(customer_id, set())
        cross_sell = next(
            (category for category in global_categories if category not in customer_categories),
            None,
        )
        recommendations.append(cross_sell or favorite_category or defaults["category"])
    return recommendations


def _build_reason(row):
    if row["total_prior_transactions"] == 0:
        reason = "Cliente sin compras previas: activar exploracion guiada y captura de necesidad."
        return reason
    if row["days_since_last_purchase"] <= 120:
        reason = "Compra reciente y afinidad activa: proponer complemento o categoria cruzada."
        return reason
    if row["prior_same_store_transactions"] > 0:
        reason = "Historial en la tienda objetivo: retomar relacion comercial en piso."
        return reason
    if row["has_cellphone"] or row["has_email"]:
        reason = "Tiene canal de contacto util: invitar visita asistida y reservar asesoria."
        return reason
    reason = "Contacto digital limitado: priorizar reconocimiento y oferta durante visita fisica."
    return reason


def _build_action(row):
    if row["has_cellphone"] or row["has_email"]:
        action = "Agendar visita asistida y preparar oferta de categoria recomendada."
        return action
    action = "Activar alerta en POS/CRM de tienda cuando el cliente sea identificado."
    return action


def build_store_activation_opportunities(clients, transactions, interactions, model_artifact):
    snapshot, reference_date = build_customer_snapshot(clients, transactions, interactions)
    defaults = get_global_defaults(transactions, interactions)

    scoring_df = snapshot.copy()
    scoring_df["id_tienda"] = _choose_target_store(scoring_df, defaults)
    scoring_df["motivo_visita"] = "cotizacion"
    same_store = scoring_df["favorite_store_from_purchase"].eq(scoring_df["id_tienda"]).fillna(False)
    scoring_df["prior_same_store_transactions"] = np.where(
        same_store,
        scoring_df["total_prior_transactions"],
        0,
    )
    scoring_df["conversion_probability"] = predict_conversion_probability(scoring_df, model_artifact)
    scoring_df["recommended_category"] = _choose_category(scoring_df, transactions, defaults)

    expected_ticket = scoring_df["avg_prior_ticket"].replace(0, np.nan).fillna(defaults["avg_ticket"])
    scoring_df["expected_value"] = scoring_df["conversion_probability"] * expected_ticket

    high_cut = scoring_df["expected_value"].quantile(HIGH_PRIORITY_QUANTILE)
    medium_cut = scoring_df["expected_value"].quantile(MEDIUM_PRIORITY_QUANTILE)
    scoring_df["priority"] = np.select(
        [
            scoring_df["expected_value"] >= high_cut,
            scoring_df["expected_value"] >= medium_cut,
        ],
        ["alta", "media"],
        default="baja",
    )
    scoring_df["business_reason"] = scoring_df.apply(_build_reason, axis=1)
    scoring_df["recommended_action"] = scoring_df.apply(_build_action, axis=1)
    scoring_df["reference_date"] = reference_date.date().isoformat()

    columns = [
        "reference_date",
        "id_cliente",
        "zona_geografica",
        "id_tienda",
        "recommended_category",
        "conversion_probability",
        "expected_value",
        "priority",
        "business_reason",
        "recommended_action",
        "total_prior_transactions",
        "total_prior_spend",
        "days_since_last_purchase",
        "has_cellphone",
        "has_email",
    ]
    priority_order = {"alta": 0, "media": 1, "baja": 2}
    output = scoring_df[columns].copy()
    output["_priority_order"] = output["priority"].map(priority_order)
    output = (
        output.sort_values(["_priority_order", "expected_value"], ascending=[True, False])
        .drop(columns="_priority_order")
        .reset_index(drop=True)
    )
    return output
