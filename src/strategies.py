"""
strategies.py
-------------
Patrón de diseño STRATEGY.

Define una interfaz común (`ClassifierStrategy`) para todos los algoritmos de
clasificación. Cada algoritmo concreto (MLP, regresión logística, árbol de
decisión, random forest) la implementa, de modo que son INTERCAMBIABLES sin
modificar el resto del sistema. Añadir un nuevo modelo solo requiere crear una
nueva subclase y registrarla en la fábrica.
"""

from abc import ABC, abstractmethod

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class ClassifierStrategy(ABC):
    """Interfaz común de todos los clasificadores."""

    def __init__(self):
        self.model = None          # lo define cada estrategia concreta
        self.name = "base"

    @abstractmethod
    def train(self, X, y):
        """Entrena el modelo con los datos proporcionados."""
        raise NotImplementedError

    def predict(self, X):
        """Devuelve la clase predicha para cada muestra."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Devuelve la probabilidad de la clase positiva (benigno)."""
        return self.model.predict_proba(X)[:, 1]


class MLPStrategy(ClassifierStrategy):
    """Clasificador basado en un Perceptrón Multicapa (red neuronal)."""

    def __init__(self):
        super().__init__()
        self.name = "Perceptrón Multicapa (MLP)"
        self.model = MLPClassifier(
            hidden_layer_sizes=(16, 8),   # dos capas ocultas
            activation="relu",
            solver="adam",
            alpha=1e-4,                   # regularización L2
            max_iter=600,
            random_state=42,
        )

    def train(self, X, y):
        self.model.fit(X, y)
        return self


class DecisionTreeStrategy(ClassifierStrategy):
    """Clasificador basado en un Árbol de Decisión."""

    def __init__(self):
        super().__init__()
        self.name = "Árbol de Decisión"
        self.model = DecisionTreeClassifier(max_depth=5, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)
        return self


class RandomForestStrategy(ClassifierStrategy):
    """Clasificador basado en un bosque aleatorio (Random Forest)."""

    def __init__(self):
        super().__init__()
        self.name = "Random Forest"
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X, y):
        self.model.fit(X, y)
        return self


class LogisticStrategy(ClassifierStrategy):
    """Modelo base de comparación: regresión logística."""

    def __init__(self):
        super().__init__()
        self.name = "Regresión Logística"
        self.model = LogisticRegression(max_iter=5000)

    def train(self, X, y):
        self.model.fit(X, y)
        return self
