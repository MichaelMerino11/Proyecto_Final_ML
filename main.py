from src.data_loader import get_prepared_data, stream_data
from src.online.online_clusterer import OnlineClusterer
from src.constraints.size_manager import SizeConstrainedManager
from src.eval.metrics_evaluator import evaluate_model, plot_results

def main():
    print("--- Iniciando Proyecto de Clustering Online (Iris) ---")
    
    # 1. Preparar datos
    X, y, classes = get_prepared_data()
    print(f"Dataset cargado: {len(X)} instancias, {len(classes)} clases.")

    # 2. Simular flujo online (Estructura base para el algoritmo)
    print("Simulando llegada de datos...")
    data_stream = stream_data(X, y)
    
    for i, (point, label) in enumerate(data_stream):
        # Aquí es donde en la Etapa 2 conectaremos el modelo
        if i % 50 == 0:
            print(f"Procesando instancia {i}...")

    print("--- Etapa 1 completada con éxito ---")

    print("--- Etapa 2: Clustering Online Semi-Supervisado ---")
    
    # 1. Cargar datos
    X, y, classes = get_prepared_data()
    
    # 2. Configurar el modelo
    clusterer = OnlineClusterer(n_clusters=3)
    
    # Tomamos una pequeña semilla para la parte SEMI-SUPERVISADA (primeros 15 datos)
    X_seed = X[:15]
    clusterer.initialize_with_semi_supervision(X_seed)
    
    # 3. Flujo Online (empezamos desde el dato 15 en adelante)
    print("Iniciando procesamiento de flujo de datos...")
    data_stream = stream_data(X[15:], y[15:])
    
    assignments = []
    for i, (point, label) in enumerate(data_stream):
        # El modelo aprende del punto y nos dice en qué grupo lo puso
        cluster_id = clusterer.partial_update(point)
        assignments.append(cluster_id)
        
        if i % 30 == 0:
            print(f"Instancia {i}: Asignada al Cluster {cluster_id}")

    print(f"--- Etapa 2 finalizada: {len(assignments)} puntos procesados online ---")

    print("--- Etapa 3: Clustering con Restricción de Tamaño ---")
    
    X, y, classes = get_prepared_data()
    clusterer = OnlineClusterer(n_clusters=3)
    
    # 1. Definir la restricción (Ej: Máximo 50 por grupo para Iris)
    # Meta 2 del PDF: Modificar el algoritmo para restringir tamaños 
    manager = SizeConstrainedManager(n_clusters=3, max_capacity=50)
    
    # Inicialización semi-supervisada (primeros 15 datos)
    X_seed = X[:15]
    clusterer.initialize_with_semi_supervision(X_seed)
    # Actualizamos el conteo del manager con la semilla
    for i in range(15):
        initial_id = clusterer.model.predict(X_seed[i].reshape(1,-1))[0]
        manager.counts[initial_id] += 1
    
    # 2. Procesamiento con RESTRICCIÓN
    print(f"Capacidad máxima por cluster: {manager.max_capacity}")
    data_stream = stream_data(X[15:], y[15:])
    
    final_assignments = []
    for i, (point, label) in enumerate(data_stream):
        # Primero el modelo se actualiza (Online)
        clusterer.partial_update(point)
        
        # Luego el Manager decide si ese cluster es válido o debe reasignar
        valid_cluster_id = manager.get_best_available_cluster(point, clusterer.model)
        final_assignments.append(valid_cluster_id)

    # 3. Verificación de Cardinalidad (Requisito de Evaluación Meta 3 )
    print("\nResultados de la Restricción (Cardinalidad):")
    for cid, count in manager.counts.items():
        print(f"Cluster {cid}: {count} instancias (Límite 50)")

    print(f"--- Etapa 3 finalizada: {len(final_assignments)} puntos procesados con restricción ---")
    print("--- Etapa 4: Evaluación y Resultados Finales ---")
    
    # 1. Preparación de datos (Meta 1 y 2) 
    X, y, classes = get_prepared_data()
    clusterer = OnlineClusterer(n_clusters=3)
    manager = SizeConstrainedManager(n_clusters=3, max_capacity=50)
    
    # Semilla Semi-supervisada
    X_seed = X[:15]
    y_seed = y[:15] # Guardamos para la métrica externa
    clusterer.initialize_with_semi_supervision(X_seed)
    
    for i in range(15):
        initial_id = clusterer.model.predict(X_seed[i].reshape(1,-1))[0]
        manager.counts[initial_id] += 1
    
    # 2. Procesamiento Online con Restricción (Meta 2 de Materia 2) 
    stream_assignments = []
    for point, label in stream_data(X[15:], y[15:]):
        clusterer.partial_update(point)
        valid_id = manager.get_best_available_cluster(point, clusterer.model)
        stream_assignments.append(valid_id)
    
    # Combinamos semilla + streaming para evaluación total
    total_assignments = [clusterer.model.predict(p.reshape(1,-1))[0] for p in X[:15]] + stream_assignments
    
    # 3. Cálculo de Métricas (Meta 3 de Materia 2) 
    ari, sil = evaluate_model(X, total_assignments, y)
    
    print("\n--- INFORME DE RENDIMIENTO ---")
    print(f"Métrica Externa (Adjusted Rand Index): {ari:.4f}  (Ideal: 1.0)")
    print(f"Métrica Interna (Silhouette Score): {sil:.4f} (Ideal: 1.0)")
    print("-" * 30)
    
    # 4. Visualización (Para Meta 4 y Presentación) [cite: 27, 31]
    plot_results(X, total_assignments, clusterer.get_centroids())
    print("Gráfico guardado en 'results/plots/final_clustering.png'")
if __name__ == "__main__":
    main()