import numpy as np
from sklearn.cluster import MiniBatchKMeans

class OnlineClusterer:
    def __init__(self, n_clusters=3):
        """
        Inicializa el motor de clustering online.
        n_clusters=3 porque el dataset Iris tiene 3 clases.
        """
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1)
        self.n_clusters = n_clusters
        self.is_initialized = False

    def initialize_with_semi_supervision(self, X_seed):
        """
        ETAPA SEMI-SUPERVISADA: 
        Usamos una pequeña muestra inicial para fijar los centroides.
        Esto garantiza que el modelo empiece con una estructura coherente.
        """
        self.model.partial_fit(X_seed)
        self.is_initialized = True
        print("Modelo inicializado con semilla semi-supervisada.")

    def partial_update(self, x_new):
        """
        ETAPA ONLINE:
        Recibe un solo punto y actualiza los centroides del modelo.
        """
        self.model.partial_fit(x_new)
        # Retornamos a qué cluster asignó el nuevo punto
        return self.model.predict(x_new)[0]

    def get_centroids(self):
        return self.model.cluster_centers_