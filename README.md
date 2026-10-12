# Wine Quality - Experimentación con MLflow

Este proyecto implementa un flujo reproducible de entrenamiento y evaluación de un modelo Random Forest para predecir la calidad del vino a partir del conjunto de datos Wine Quality Red.

El script `train.py` permite configurar los principales hiperparámetros del modelo desde la línea de comandos y utiliza MLflow para registrar parámetros, métricas, artefactos y el modelo entrenado, facilitando la comparación sistemática entre diferentes experimentos.

## Requisitos

- Python 3
- pip

## Instalación

Se recomienda crear y activar un entorno virtual antes de instalar las dependencias del proyecto.

En Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

## Ejecución del entrenamiento

El entrenamiento puede ejecutarse con los valores predeterminados mediante:

```powershell
python train.py
```

Los principales hiperparámetros pueden configurarse desde la línea de comandos:

```powershell
python train.py --n_estimators 100 --max_depth 10 --min_samples_split 2 --seed 42
```

Para consultar todos los argumentos disponibles:

```powershell
python train.py --help
```

### Mejor configuración evaluada

Entre las configuraciones experimentadas, la corrida `n_estimators_50` obtuvo el mejor desempeño. Puede reproducirse mediante:

```powershell
python train.py --run_name n_estimators_50 --n_estimators 50 --max_depth 10 --min_samples_split 2 --seed 42 --test_size 0.2
```

Esta configuración obtuvo:

- MAE: 0.4597
- RMSE: 0.6066
- R²: 0.4806

## Visualización de experimentos con MLflow

Los experimentos realizados durante el entrenamiento se registran en MLflow bajo el experimento `Wine_Quality_RandomForest`.

Para iniciar la interfaz de MLflow, ejecutar desde el entorno virtual:

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000
```

Posteriormente, abrir en el navegador:

```text
http://127.0.0.1:5000
```

En MLflow es posible consultar y comparar las diferentes corridas realizadas, incluyendo:

- Hiperparámetros utilizados en cada corrida.
- Métricas de evaluación: MAE, RMSE y R².
- Artefacto gráfico `actual_vs_predicted.png`.
- Modelo entrenado almacenado en cada corrida.

Durante la experimentación se realizaron siete corridas controladas modificando `n_estimators`, `max_depth` y `min_samples_split`, manteniendo la semilla y la proporción de prueba constantes para facilitar la comparación de resultados.
