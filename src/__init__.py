"""
Paquete src del proyecto de clasificación de tumores mamarios (Equipo Monos - USIL).

Contiene los componentes de la solución, cada uno con una responsabilidad única:
    - data_loader   : carga del conjunto de datos.
    - preprocessor  : estandarización y partición estratificada.
    - strategies    : interfaz y algoritmos de clasificación (patrón Strategy).
    - model_factory : creación de modelos (patrón Factory).
    - evaluator     : cálculo de métricas y generación de figuras.
    - pipeline      : orquestador del flujo completo (patrón Facade).
"""
