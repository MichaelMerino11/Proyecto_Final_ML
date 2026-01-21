import numpy as np

def stratified_seed_indices(y: np.ndarray, per_class: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Elige 'per_class' índices por clase (estratificado).
    """
    rng = np.random.default_rng(random_state)
    idx_all = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        chosen = rng.choice(idx, size=per_class, replace=False)
        idx_all.append(chosen)
    return np.concatenate(idx_all)

def centroids_from_labeled(X: np.ndarray, y: np.ndarray, labeled_idx: np.ndarray, n_clusters: int):
    """
    Calcula centroides iniciales como promedio por clase, usando SOLO labeled_idx.
    """
    centroids = []
    for c in range(n_clusters):
        idx = labeled_idx[y[labeled_idx] == c]
        if len(idx) == 0:
            raise ValueError(f"No hay semillas para clase {c}.")
        centroids.append(X[idx].mean(axis=0))
    return np.vstack(centroids)
