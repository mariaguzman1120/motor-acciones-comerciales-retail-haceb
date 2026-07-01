import json

import numpy as np
import pandas as pd

from python.metadata.pipeline_config import (
    CATEGORICAL_FEATURES,
    MODEL_PARAMETERS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def _sigmoid(values):
    probabilities = 1 / (1 + np.exp(-np.clip(values, -35, 35)))
    return probabilities


def _auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positive_count = int(y_true.sum())
    negative_count = int(len(y_true) - positive_count)
    if positive_count == 0 or negative_count == 0:
        auc = None
        return auc

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    positive_rank_sum = ranks[y_true == 1].sum()
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )
    auc = float(auc)
    return auc


def _log_loss(y_true, y_score):
    clipped = np.clip(y_score, 1e-9, 1 - 1e-9)
    loss = float(-(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)).mean())
    return loss


def fit_preprocessor(df):
    numeric_medians = df[NUMERIC_FEATURES].median(numeric_only=True).fillna(0.0)
    numeric_means = df[NUMERIC_FEATURES].fillna(numeric_medians).mean()
    numeric_stds = df[NUMERIC_FEATURES].fillna(numeric_medians).std().replace(0, 1).fillna(1)
    categorical_levels = {
        column: sorted(df[column].fillna("missing").astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }
    feature_names = ["intercept"] + NUMERIC_FEATURES[:]
    for column, levels in categorical_levels.items():
        feature_names.extend([f"{column}={level}" for level in levels])
    preprocessor = {
        "numeric_medians": numeric_medians.to_dict(),
        "numeric_means": numeric_means.to_dict(),
        "numeric_stds": numeric_stds.to_dict(),
        "categorical_levels": categorical_levels,
        "feature_names": feature_names,
    }
    return preprocessor


def transform_features(df, preprocessor):
    parts = [np.ones((len(df), 1))]

    numeric = df[NUMERIC_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        median = preprocessor["numeric_medians"][column]
        mean = preprocessor["numeric_means"][column]
        std = preprocessor["numeric_stds"][column] or 1
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce").fillna(median)
        numeric[column] = (numeric[column] - mean) / std
    parts.append(numeric.to_numpy(dtype=float))

    for column in CATEGORICAL_FEATURES:
        values = df[column].fillna("missing").astype(str)
        for level in preprocessor["categorical_levels"][column]:
            parts.append(values.eq(level).astype(float).to_numpy().reshape(-1, 1))

    features = np.hstack(parts)
    return features


def temporal_train_test_split(df):
    ordered = df.sort_values("fecha_visita").reset_index(drop=True)
    test_size = max(1, int(len(ordered) * MODEL_PARAMETERS["test_fraction"]))
    train = ordered.iloc[:-test_size].copy()
    test = ordered.iloc[-test_size:].copy()
    split = (train, test)
    return split


def train_logistic_regression(x_train, y_train):
    weights = np.zeros(x_train.shape[1])
    learning_rate = MODEL_PARAMETERS["learning_rate"]
    l2_penalty = MODEL_PARAMETERS["l2_penalty"]

    for _ in range(MODEL_PARAMETERS["iterations"]):
        probabilities = _sigmoid(x_train @ weights)
        gradient = (x_train.T @ (probabilities - y_train)) / len(y_train)
        regularization = l2_penalty * weights
        regularization[0] = 0
        weights -= learning_rate * (gradient + regularization)

    return weights


def evaluate_model(y_true, probabilities):
    threshold = MODEL_PARAMETERS["decision_threshold"]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(predictions)),
        "accuracy": float(np.mean(predictions == y_true)),
        "log_loss": _log_loss(y_true, probabilities),
        "auc": _auc_score(y_true, probabilities),
    }
    return metrics


def train_conversion_model(training_df):
    train_df, test_df = temporal_train_test_split(training_df)
    candidates = [
        _train_logistic_candidate(train_df, test_df),
        _train_rate_candidate(train_df, test_df, ["id_tienda"]),
        _train_rate_candidate(train_df, test_df, ["id_tienda", "motivo_visita"]),
        _train_rate_candidate(train_df, test_df, ["motivo_visita"]),
    ]
    champion = sorted(candidates, key=_candidate_sort_key, reverse=True)[0]
    champion["metrics"]["selected_model"] = champion["model_name"]
    champion["metrics"]["candidates"] = {
        candidate["model_name"]: candidate["metrics"]["test"] for candidate in candidates
    }
    return champion


def _candidate_sort_key(candidate):
    test_metrics = candidate["metrics"]["test"]
    auc = test_metrics["auc"] if test_metrics["auc"] is not None else -1
    sort_key = (auc, -test_metrics["log_loss"], test_metrics["accuracy"])
    return sort_key


def _train_logistic_candidate(train_df, test_df):
    preprocessor = fit_preprocessor(train_df)
    x_train = transform_features(train_df, preprocessor)
    x_test = transform_features(test_df, preprocessor)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    weights = train_logistic_regression(x_train, y_train)
    train_probabilities = _sigmoid(x_train @ weights)
    test_probabilities = _sigmoid(x_test @ weights)

    artifact = {
        "model_type": "logistic_regression",
        "model_name": "logistic_regression_full_features",
        "preprocessor": preprocessor,
        "weights": weights,
        "feature_names": preprocessor["feature_names"],
        "metrics": {
            "train": evaluate_model(y_train, train_probabilities),
            "test": evaluate_model(y_test, test_probabilities),
        },
    }
    return artifact


def _train_rate_candidate(train_df, test_df, group_columns, alpha=20):
    global_rate = float(train_df[TARGET_COLUMN].mean())
    rate_table = (
        train_df.groupby(group_columns)[TARGET_COLUMN]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "positives", "count": "observations"})
    )
    rate_table["conversion_probability"] = (
        rate_table["positives"] + alpha * global_rate
    ) / (rate_table["observations"] + alpha)

    artifact = {
        "model_type": "bayesian_rate",
        "model_name": f"bayesian_rate_by_{'_'.join(group_columns)}",
        "group_columns": group_columns,
        "global_rate": global_rate,
        "alpha": alpha,
        "rate_table": rate_table,
        "metrics": {},
    }
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    artifact["metrics"] = {
        "train": evaluate_model(y_train, _predict_rate_model(train_df, artifact)),
        "test": evaluate_model(y_test, _predict_rate_model(test_df, artifact)),
    }
    return artifact


def _predict_rate_model(df, artifact):
    rate_table = artifact["rate_table"]
    group_columns = artifact["group_columns"]
    probabilities = df[group_columns].merge(
        rate_table[group_columns + ["conversion_probability"]],
        on=group_columns,
        how="left",
    )["conversion_probability"]
    predictions = probabilities.fillna(artifact["global_rate"]).to_numpy(dtype=float)
    return predictions


def predict_conversion_probability(df, artifact):
    if artifact["model_type"] == "bayesian_rate":
        probabilities = _predict_rate_model(df, artifact)
        return probabilities
    x_values = transform_features(df, artifact["preprocessor"])
    probabilities = _sigmoid(x_values @ artifact["weights"])
    return probabilities


def write_model_outputs(model_dir, artifact):
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_dir / "model_metrics.json"
    coefficients_path = model_dir / "model_coefficients.csv"

    metrics_path.write_text(
        json.dumps(artifact["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if artifact["model_type"] == "bayesian_rate":
        coefficients = artifact["rate_table"].copy()
        coefficients.insert(0, "model_name", artifact["model_name"])
        coefficients = coefficients.sort_values("conversion_probability", ascending=False)
    else:
        coefficients = pd.DataFrame(
            {
                "feature": artifact["feature_names"],
                "coefficient": artifact["weights"],
            }
        ).sort_values("coefficient", ascending=False)
    coefficients.to_csv(coefficients_path, index=False, encoding="utf-8")
