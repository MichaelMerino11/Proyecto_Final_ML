import numpy as np

class SizeConstrainedManager:
    def __init__(self, n_clusters, max_capacity):
        """
        Controla que ningún cluster exceda la capacidad permitida.
        """
        self.n_clusters = n_clusters
        self.max_capacity = max_capacity
        self.counts = {i: 0 for i in range(n_clusters)}

    def get_best_available_cluster(self, point, model):
        """
        Lógica de reasignación: Si el más cercano está lleno, busca el siguiente.
        """
        # Obtenemos las distancias del punto a todos los centroides
        distances = model.transform(point)[0]
        # Ordenamos los índices de los clusters de más cercano a más lejano
        preferred_clusters = np.argsort(distances)

        for cluster_id in preferred_clusters:
            if self.counts[cluster_id] < self.max_capacity:
                self.counts[cluster_id] += 1
                return cluster_id
        
        # Si todos están llenos (no debería pasar con Iris y capacidad 50)
        return preferred_clusters[0]