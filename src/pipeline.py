"""
pipeline.py
-----------
Patrón de diseño FACADE.

`Pipeline` ofrece una interfaz simple (`run()`) que esconde la coordinación de
todos los componentes: carga de datos, preprocesamiento, creación del modelo,
entrenamiento, predicción y evaluación. Quien usa el sistema no necesita conocer
los detalles internos.
"""

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_factory import ModelFactory
from src.evaluator import Evaluator


class Pipeline:
    """Orquesta el flujo completo del experimento."""

    def __init__(self, model_name="mlp", out_dir="figuras", random_state=42):
        self.model_name = model_name
        self.random_state = random_state
        self.loader = DataLoader()
        self.preprocessor = Preprocessor(random_state=random_state)
        self.evaluator = Evaluator(
            class_names=self.loader.target_names, out_dir=out_dir
        )
        self.strategy = ModelFactory.create(model_name)

    def run(self, cross_validate=True, make_plots=True):
        """
        Ejecuta el experimento de principio a fin.

        Returns
        -------
        dict
            Métricas del modelo sobre el conjunto de prueba (y CV si aplica).
        """
        # 1) Carga
        X, y = self.loader.load()

        # 2) Preprocesamiento (split estratificado + estandarización)
        X_train, X_test, y_train, y_test = self.preprocessor.split_and_scale(X, y)

        # 3) Entrenamiento
        self.strategy.train(X_train, y_train)

        # 4) Predicción
        y_pred = self.strategy.predict(X_test)
        y_proba = self.strategy.predict_proba(X_test)

        # 5) Evaluación
        metricas = self.evaluator.evaluate(y_test, y_pred, y_proba)

        # 6) Validación cruzada estratificada (sobre todos los datos escalados)
        if cross_validate:
            X_scaled = StandardScaler().fit_transform(X)
            skf = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )
            fresh_model = ModelFactory.create(self.model_name).model
            cv = cross_val_score(fresh_model, X_scaled, y, cv=skf, scoring="accuracy")
            metricas["cv_media"] = float(np.mean(cv))
            metricas["cv_desv"] = float(np.std(cv))

        # 7) Figuras
        if make_plots:
            self.evaluator.plot_confusion(y_test, y_pred)
            self.evaluator.plot_roc(y_test, y_proba)
            self.evaluator.plot_loss(self.strategy)

        # Guardamos el reporte por clase para impresión externa
        self._last_report = self.evaluator.classification_report(y_test, y_pred)
        return metricas

    @property
    def last_report(self):
        return getattr(self, "_last_report", "")
