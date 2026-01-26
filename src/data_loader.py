import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def get_prepared_data():
    """
    Carga el dataset Iris y lo normaliza. 
    Retorna los datos X (características) y y (clases reales).
    """
    # Cargamos el dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # Escalado de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, target_names

def stream_data(X, y):
    """
    Generador que simula la llegada de datos uno por uno (Clustering Online).
    """
    for i in range(len(X)):
        yield X[i].reshape(1, -1), y[i]