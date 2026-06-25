"""
test_pipeline.py
----------------
Prueba mínima que verifica que el pipeline corre y alcanza un desempeño
razonable. Ejecutar desde la carpeta 'codigo/':

    python -m pytest -q          (si tienes pytest)
    python tests/test_pipeline.py  (ejecución directa)
"""

import os
import sys

# Permite importar 'src' al ejecutar el test directamente.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import Pipeline
from src.model_factory import ModelFactory


def test_mlp_desempeno_minimo():
    """El MLP debe superar el 90 % de exactitud en prueba."""
    metricas = Pipeline(model_name="mlp").run(cross_validate=False, make_plots=False)
    assert metricas["exactitud"] > 0.90


def test_factory_modelo_invalido():
    """La fábrica debe rechazar un nombre de modelo desconocido."""
    try:
        ModelFactory.create("inexistente")
        assert False, "Se esperaba un ValueError"
    except ValueError:
        assert True


if __name__ == "__main__":
    test_mlp_desempeno_minimo()
    test_factory_modelo_invalido()
    print("Todas las pruebas pasaron correctamente.")
