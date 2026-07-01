from python.features.feature_engineering import build_visit_training_dataset
from python.metadata.pipeline_config import (
    AUDIT_DIR,
    BUSINESS_DIR,
    INPUT_FILES,
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
)
from python.models.conversion_model import train_conversion_model, write_model_outputs
from python.outputs.store_activation import build_store_activation_opportunities
from python.utils.cleaners import clean_input_data
from python.utils.pipeline_validation import (
    raise_for_errors,
    validate_business_rules,
    validate_input_files,
    validate_primary_keys,
    validate_required_columns,
    validate_training_dataset,
    write_audit_log,
)
from python.utils.readers import read_input_data


def ensure_output_dirs():
    for path in [OUTPUT_DIR, PROCESSED_DIR, MODEL_DIR, BUSINESS_DIR, AUDIT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def run_pipeline():
    ensure_output_dirs()

    messages = validate_input_files(INPUT_FILES)
    raise_for_errors(messages)

    raw_data = read_input_data(INPUT_FILES)
    messages.extend(validate_required_columns(raw_data))
    raise_for_errors(messages)

    clean_data = clean_input_data(raw_data)
    messages.extend(validate_primary_keys(clean_data))
    messages.extend(validate_business_rules(clean_data))
    raise_for_errors(messages)

    training_df = build_visit_training_dataset(
        clean_data["clients"],
        clean_data["transactions"],
        clean_data["interactions"],
    )
    messages.extend(validate_training_dataset(training_df))

    training_path = PROCESSED_DIR / "training_dataset.csv"
    training_df.to_csv(training_path, index=False, encoding="utf-8")

    model_artifact = train_conversion_model(training_df)
    write_model_outputs(MODEL_DIR, model_artifact)

    opportunities = build_store_activation_opportunities(
        clean_data["clients"],
        clean_data["transactions"],
        clean_data["interactions"],
        model_artifact,
    )
    opportunities_path = BUSINESS_DIR / "store_activation_opportunities.csv"
    opportunities.to_csv(opportunities_path, index=False, encoding="utf-8")

    audit_path = AUDIT_DIR / "audit_log.json"
    write_audit_log(
        audit_path,
        {
            "input_files": {name: str(path) for name, path in INPUT_FILES.items()},
            "rows": {name: int(df.shape[0]) for name, df in clean_data.items()},
            "messages": messages,
            "outputs": {
                "training_dataset": str(training_path),
                "model_metrics": str(MODEL_DIR / "model_metrics.json"),
                "model_coefficients": str(MODEL_DIR / "model_coefficients.csv"),
                "store_activation_opportunities": str(opportunities_path),
            },
        },
    )

    pipeline_result = {
        "messages": messages,
        "metrics": model_artifact["metrics"],
        "opportunities_path": opportunities_path,
        "audit_path": audit_path,
    }
    return pipeline_result


if __name__ == "__main__":
    result = run_pipeline()
    warning_count = sum(1 for message in result["messages"] if message["level"] == "warning")
    print("Pipeline ejecutado correctamente.")
    print(f"Advertencias registradas: {warning_count}")
    print(f"Salida comercial: {result['opportunities_path']}")
    print(f"Auditoria: {result['audit_path']}")
