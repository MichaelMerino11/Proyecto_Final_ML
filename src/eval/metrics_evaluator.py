from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

def evaluate_model(X, y_pred, y_true):
    """
    Retorna métricas internas y externas.
    """
    metrics = {}
    metrics["ARI"] = adjusted_rand_score(y_true, y_pred)
    metrics["NMI"] = normalized_mutual_info_score(y_true, y_pred)

    # internas
    metrics["silhouette"] = silhouette_score(X, y_pred)
    metrics["davies_bouldin"] = davies_bouldin_score(X, y_pred)
    metrics["calinski_harabasz"] = calinski_harabasz_score(X, y_pred)

    return metrics
