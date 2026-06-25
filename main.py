"""
main.py
-------
Punto de entrada del proyecto de clasificación de tumores mamarios.

Ejecuta el Perceptrón Multicapa y, como referencia, el modelo base de regresión
logística. Imprime las métricas y genera las figuras en la carpeta 'figuras/'.

Uso:
    python main.py
"""

from src.pipeline import Pipeline


def imprimir_resultados(nombre, metricas):
    print(f"\n=== {nombre} ===")
    print(f"  Exactitud (prueba) : {metricas['exactitud'] * 100:.2f} %")
    print(f"  Precisión          : {metricas['precision'] * 100:.2f} %")
    print(f"  Sensibilidad (rec.): {metricas['recall'] * 100:.2f} %")
    print(f"  Valor F1           : {metricas['f1'] * 100:.2f} %")
    if "auc" in metricas:
        print(f"  AUC (ROC)          : {metricas['auc']:.3f}")
    if "cv_media" in metricas:
        print(f"  Validación cruzada : {metricas['cv_media'] * 100:.2f} % "
              f"(± {metricas['cv_desv'] * 100:.2f} %)")


def main():
    print("Clasificación de tumores mamarios — Breast Cancer Wisconsin (Diagnostic)")
    print("Equipo «Monos» — USIL")

    # Modelo principal: Perceptrón Multicapa
    pipe_mlp = Pipeline(model_name="mlp")
    metricas_mlp = pipe_mlp.run()
    imprimir_resultados("Perceptrón Multicapa (MLP)", metricas_mlp)
    print("\nReporte por clase (MLP):")
    print(pipe_mlp.last_report)

    # Modelo base de comparación: Regresión Logística
    pipe_lr = Pipeline(model_name="logistic")
    metricas_lr = pipe_lr.run(make_plots=False)
    imprimir_resultados("Regresión Logística (base)", metricas_lr)

    print("\nFiguras generadas en la carpeta 'figuras/'.")


if __name__ == "__main__":
    main()
