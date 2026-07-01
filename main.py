"""Punto de entrada simple para ejecutar el flujo de entrenamiento."""

from typing import Any

from python.training_flow_prefect import training_flow


def run_training_flow() -> dict[str, Any]:
    """Ejecuta el flujo Prefect de entrenamiento.

    Returns:
        Resumen de la ejecucion del flujo.
    """
    return training_flow()


if __name__ == '__main__':
    run_training_flow()
