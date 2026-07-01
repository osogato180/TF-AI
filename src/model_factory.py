"""
model_factory.py
----------------
Patrón de diseño FACTORY.

Centraliza la creación de las estrategias de clasificación. El resto del sistema
pide un modelo por su nombre ("mlp", "logistic", "tree", "forest") sin conocer
los detalles de su construcción, lo que desacopla la creación del uso.
"""

from src.strategies import (
    MLPStrategy, LogisticStrategy, DecisionTreeStrategy, RandomForestStrategy
)


class ModelFactory:
    """Crea estrategias de clasificación a partir de un nombre."""

    _MODELS = {
        "mlp": MLPStrategy,
        "logistic": LogisticStrategy,
        "tree": DecisionTreeStrategy,
        "forest": RandomForestStrategy,
    }

    @staticmethod
    def create(name):
        """Devuelve una instancia de la estrategia solicitada.

        Lanza ValueError si el nombre no corresponde a un modelo registrado.
        """
        key = name.lower().strip()
        if key not in ModelFactory._MODELS:
            disponibles = ", ".join(ModelFactory._MODELS.keys())
            raise ValueError(f"Modelo '{name}' no reconocido. Opciones: {disponibles}.")
        return ModelFactory._MODELS[key]()
