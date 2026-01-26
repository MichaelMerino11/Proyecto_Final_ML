import numpy as np

class ControlledOnlineKMeans:
    """
    Online K-Means controlado:
    - Elige cluster por distancia a centroides
    - Puede aplicar restricción de tamaño
    - Actualiza el centroide DEL cluster elegido (no del que el modelo "quiera")
    """

    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centroids = None
        self.counts = None

    def init_random(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        self.counts = np.zeros(self.n_clusters, dtype=int)

    def init_with_centroids(self, centroids: np.ndarray):
        self.centroids = centroids.copy()
        self.counts = np.zeros(self.n_clusters, dtype=int)

    def _dist2_all(self, x: np.ndarray) -> np.ndarray:
        diff = self.centroids - x
        return np.sum(diff * diff, axis=1)

    def choose_cluster(self, x: np.ndarray, max_size: int | None = None) -> int:
        """
        Devuelve el cluster más cercano disponible. Si max_size=None, sin restricción.
        """
        d2 = self._dist2_all(x)
        order = np.argsort(d2)

        if max_size is None:
            return int(order[0])

        for j in order:
            if self.counts[j] < max_size:
                return int(j)

        # Si todos están llenos (solo pasa si k*max_size < n)
        return int(order[0])

    def update_centroid(self, j: int, x: np.ndarray):
        """
        Update incremental: c <- c + (1/n_j)(x - c)
        """
        self.counts[j] += 1
        eta = 1.0 / self.counts[j]
        self.centroids[j] = self.centroids[j] + eta * (x - self.centroids[j])

    def partial_fit_predict(self, x: np.ndarray, max_size: int | None = None) -> int:
        if self.centroids is None:
            raise ValueError("Centroides no inicializados.")
        j = self.choose_cluster(x, max_size=max_size)
        self.update_centroid(j, x)
        return j

    def fit_predict_stream(self, X: np.ndarray, max_size: int | None = None) -> np.ndarray:
        y_pred = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            y_pred[i] = self.partial_fit_predict(X[i], max_size=max_size)
        return y_pred
