"""Pruebas para el punto de entrada principal del proyecto."""

import importlib
import unittest
from unittest.mock import patch


class MainEntrypointTests(unittest.TestCase):
    """Valida la ejecucion simple desde la raiz del proyecto."""

    def test_run_training_flow_calls_prefect_flow(self) -> None:
        """El punto de entrada debe delegar en el flujo Prefect."""
        module = importlib.import_module('main')
        with patch.object(module, 'training_flow', return_value={'ok': True}) as mock_flow:
            result = module.run_training_flow()
        self.assertEqual(result, {'ok': True})
        mock_flow.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
