"""
data_loader.py
--------------
Responsabilidad única: cargar el conjunto de datos Breast Cancer Wisconsin
(Diagnostic) y entregarlo en una estructura tabular lista para procesar.

El conjunto está incluido en scikit-learn, por lo que la carga es offline y
totalmente reproducible.
"""

from sklearn.datasets import load_breast_cancer


class DataLoader:
    """Carga el dataset Breast Cancer Wisconsin (Diagnostic)."""

    def __init__(self):
        # Nombres legibles de las clases. En sklearn: 0 = maligno, 1 = benigno.
        self.target_names = ["Maligno", "Benigno"]
        self.feature_names = None

    def load(self):
        """
        Carga el conjunto de datos.

        Returns
        -------
        X : pandas.DataFrame
            Matriz de características (569 filas x 30 columnas).
        y : pandas.Series
            Variable objetivo binaria (0 = maligno, 1 = benigno).
        """
        data = load_breast_cancer(as_frame=True)
        X = data.data           # DataFrame con las 30 características
        y = data.target         # Series con la etiqueta
        self.feature_names = list(data.feature_names)
        return X, y
