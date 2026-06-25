"""
evaluator.py
------------
Responsabilidad única: medir el desempeño del modelo y generar las figuras.

Calcula las métricas derivadas de la matriz de confusión (exactitud, precisión,
sensibilidad/recall, F1 y AUC) y produce las visualizaciones del informe:
matriz de confusión, curva ROC y, si está disponible, la curva de pérdida.
"""

import os

import matplotlib
matplotlib.use("Agg")            # backend sin ventana, apto para servidores
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)

AZUL = "#2563eb"
ROJO = "#dc2626"


class Evaluator:
    """Calcula métricas y genera figuras de evaluación."""

    def __init__(self, class_names=("Maligno", "Benigno"), out_dir="figuras"):
        self.class_names = list(class_names)
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def evaluate(self, y_true, y_pred, y_proba=None):
        """Devuelve un diccionario con las métricas principales."""
        metricas = {
            "exactitud": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        }
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            metricas["auc"] = auc(fpr, tpr)
        return metricas

    def classification_report(self, y_true, y_pred):
        """Reporte de texto por clase."""
        return classification_report(
            y_true, y_pred, target_names=self.class_names
        )

    def plot_confusion(self, y_true, y_pred, filename="matriz_confusion.png"):
        """Genera y guarda la matriz de confusión."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(self.class_names)))
        ax.set_xticklabels(self.class_names)
        ax.set_yticks(range(len(self.class_names)))
        ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title("Matriz de confusión")
        path = os.path.join(self.out_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def plot_roc(self, y_true, y_proba, filename="roc.png"):
        """Genera y guarda la curva ROC."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(4.8, 4))
        ax.plot(fpr, tpr, color=AZUL, lw=2, label=f"Modelo (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Curva ROC")
        ax.legend(loc="lower right")
        path = os.path.join(self.out_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def plot_loss(self, strategy, filename="perdida.png"):
        """
        Genera la curva de pérdida si el modelo la expone (solo el MLP).
        Devuelve None si el modelo no tiene `loss_curve_`.
        """
        model = getattr(strategy, "model", None)
        loss = getattr(model, "loss_curve_", None)
        if loss is None:
            return None
        fig, ax = plt.subplots(figsize=(5, 3.3))
        ax.plot(loss, color=AZUL, lw=1.8)
        ax.set_xlabel("Época")
        ax.set_ylabel("Pérdida (log-loss)")
        ax.set_title("Convergencia del entrenamiento")
        path = os.path.join(self.out_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
