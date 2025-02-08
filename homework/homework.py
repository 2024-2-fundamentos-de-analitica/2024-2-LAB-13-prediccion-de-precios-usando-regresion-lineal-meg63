#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
import os

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
def crear_pipeline( x_train):
  

    categorical_features=['Fuel_Type','Selling_type','Transmission']
    numerical_features= [col for col in x_train.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_features),
                ('scaler',MinMaxScaler(),numerical_features),
            ],
        )

    pipeline=Pipeline(
            [
                ("preprocessor",preprocessor),
                ('feature_selection',SelectKBest(f_regression)),
                ('classifier', LinearRegression())
            ]
        )

    return pipeline
def optimizar_hiperparametros(pipeline, x_train, y_train):
    param_grid = {
    'feature_selection__k':range(1,25),
    'classifier__fit_intercept':[True,False],
    'classifier__positive':[True,False]

}
    grid_search=GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        )

    
    return grid_search
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
def calcular_metricas(modelo, x_train, y_train, x_test, y_test):
    # Realizar predicciones
    y_train_pred = modelo.predict(x_train)
    y_test_pred = modelo.predict(x_test)

    # Calcular métricas para entrenamiento
    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': float(r2_score(y_train, y_train_pred)),
        'mse': float(mean_squared_error(y_train, y_train_pred)),
        'mad': float(median_absolute_error(y_train, y_train_pred))
    }

    # Calcular métricas para prueba
    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': float(r2_score(y_test, y_test_pred)),
        'mse': float(mean_squared_error(y_test, y_test_pred)),
        'mad': float(median_absolute_error(y_test, y_test_pred)),
    }

    # Crear directorio si no existe
    output_path = "files/output"
    os.makedirs(output_path, exist_ok=True)

    # Guardar métricas en archivo JSON (una métrica por línea)
    metrics_file = os.path.join(output_path, "metrics.json")
    with open(metrics_file, "w") as f:
        for metrics in [metrics_train, metrics_test]:
            json.dump(metrics, f)
            f.write("\n")  # Nueva línea para cada diccionario


def save_model(path: str, estimator: GridSearchCV):
    with gzip.open(path, 'wb') as f:
        pickle.dump(estimator, f)
def main():
    # Cargar los datasets
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")

    # Preprocesar los datos
    train_data["Age"] = 2021 - train_data["Year"]
    train_data = train_data.drop(["Year", "Car_Name"], axis=1)

    test_data["Age"] = 2021 - test_data["Year"]
    test_data = test_data.drop(["Year", "Car_Name"], axis=1)

    # Dividir los datasets en x_train, y_train, x_test, y_test
    x_train = train_data.drop("Present_Price", axis=1)
    y_train = train_data["Present_Price"]
    x_test = test_data.drop("Present_Price", axis=1)
    y_test = test_data["Present_Price"]

    # Crear el pipeline
    # variables categoricas
    # Identificar variables categóricas
    categorical_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
     # Crear y entrenar el pipeline
    pipeline = crear_pipeline(x_train)

    # Optimización de hiperparámetros usando validación cruzada
    grid_search = optimizar_hiperparametros(pipeline, x_train, y_train)
    # Ajustar el modelo con la mejor combinación de hiperparámetros
    grid_search.fit(x_train, y_train)

    # Paso 5.
    # Guardar el modelo comprimido con gzip
    path2 = "./files/models/"
    save_model(
        os.path.join(path2, 'model.pkl.gz'),
        grid_search,
    )
   

    calcular_metricas(grid_search, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()

    