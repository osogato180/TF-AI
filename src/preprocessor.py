"""
preprocessor.py
---------------
Responsabilidad única: preparar los datos para el modelo.

Realiza dos tareas:
    1. Partición estratificada en entrenamiento (80 %) y prueba (20 %),
       preservando la proporción de clases.
    2. Estandarización z-score (media 0, varianza 1). El escalador se ajusta
       SOLO con el conjunto de entrenamiento para evitar fuga de información
       (data leakage) y luego se aplica al de prueba.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessor:
    """Estandariza y divide los datos de forma estratificada."""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()

    def split_and_scale(self, X, y):
        """
        Divide y estandariza los datos.

        Returns
        -------
        X_train, X_test : ndarray
            Características estandarizadas.
        y_train, y_test : Series
            Etiquetas correspondientes.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,                 # mantiene la proporción de clases
            random_state=self.random_state,
        )

        # El escalador se AJUSTA solo con entrenamiento y se APLICA a ambos.
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def transform(self, X):
        """Aplica la estandarización ya ajustada a datos nuevos."""
        return self.scaler.transform(X)
