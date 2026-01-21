from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(X_streamed, assignments, y_real):
    """
    Calcula métricas para validar el rendimiento del clustering.
    """
    # Métrica Externa: Compara tus clusters con las etiquetas reales (0 a 1)
    ari = adjusted_rand_score(y_real, assignments)
    
    # Métrica Interna: Mide qué tan cerca está cada punto de su cluster vs otros
    sil = silhouette_score(X_streamed, assignments)
    
    return ari, sil

def plot_results(X, assignments, centroids):
    """
    Genera un gráfico de dispersión para visualizar los grupos formados.
    """
    plt.figure(figsize=(10, 6))
    # Graficamos los puntos coloreados por su asignación de cluster
    plt.scatter(X[:, 0], X[:, 1], c=assignments, cmap='viridis', marker='o', alpha=0.6)
    # Graficamos los centroides resultantes
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroides')
    
    plt.title("Clustering Online con Restricciones de Tamaño (Dataset Iris)")
    plt.xlabel("Característica 1 (Escalada)")
    plt.ylabel("Característica 2 (Escalada)")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/final_clustering.png')
    plt.show()