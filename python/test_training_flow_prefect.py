"""Pruebas unitarias para el flujo de entrenamiento con Prefect."""

import importlib
import unittest


class TrainingFlowPrefectTests(unittest.TestCase):
    """Valida la superficie minima del flujo Prefect."""

    def test_module_exposes_training_flow(self) -> None:
        """El modulo debe exponer un flujo principal ejecutable."""
        module = importlib.import_module('python.training_flow_prefect')
        self.assertTrue(hasattr(module, 'training_flow'))


if __name__ == '__main__':
    unittest.main()
