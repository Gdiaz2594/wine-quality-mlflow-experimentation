import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    """Define y procesa los argumentos recibidos desde la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de Random Forest para predecir la "
        "calidad del vino."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="winequality-red.csv",
        help="Ruta al archivo CSV."
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Número de árboles del Random Forest."
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Profundidad máxima de los árboles."
    )

    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Número mínimo de muestras necesarias para dividir un nodo."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad."
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Proporción de datos destinada al conjunto de prueba."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="default_run",
        help="Nombre descriptivo de la corrida en MLflow."
    )

    return parser.parse_args()


def load_data(filepath):
    """Carga el conjunto de datos desde un archivo CSV."""
    data = pd.read_csv(filepath, sep=";")
    return data


def preprocess_data(data):
    """Elimina duplicados y separa las variables predictoras de la
     variable objetivo."""
    data = data.drop_duplicates()

    X = data.drop(columns=["quality"])
    y = data["quality"]

    return X, y


def split_data(X, y, test_size, seed):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators, max_depth,
                min_samples_split, seed):
    """Entrena un modelo Random Forest con los hiperparámetros
      especificados."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y devuelve sus métricas y predicciones."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    return metrics, predictions


def save_prediction_plot(y_test, predictions, output_path):
    """Genera y guarda una gráfica de valores reales vs. predichos."""
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, predictions, alpha=0.6)

    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title("Valores reales vs. predichos")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Ejecuta el flujo completo de entrenamiento y evaluación."""
    args = parse_args()

    mlflow.set_experiment("Wine_Quality_RandomForest")

    data = load_data(args.data_path)
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=args.test_size,
        seed=args.seed
    )

    with mlflow.start_run(run_name=args.run_name):

        model = train_model(
            X_train,
            y_train,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            seed=args.seed
        )

        metrics, predictions = evaluate_model(
            model,
            X_test,
            y_test
        )

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("test_size", args.test_size)

        mlflow.log_metric("MAE", metrics["MAE"])
        mlflow.log_metric("RMSE", metrics["RMSE"])
        mlflow.log_metric("R2", metrics["R2"])

        save_prediction_plot(
            y_test,
            predictions,
            "actual_vs_predicted.png"
        )

        mlflow.log_artifact("actual_vs_predicted.png")

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

    print("Entrenamiento completado.")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2:   {metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
