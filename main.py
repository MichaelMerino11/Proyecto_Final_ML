import os
import pandas as pd

from src.data_loader import get_prepared_data
from src.online.controlled_online_kmeans import ControlledOnlineKMeans
from src.semi.seeded_centroids import stratified_seed_indices, centroids_from_labeled
from src.eval.metrics_evaluator import evaluate_model


def ensure_dirs():
    os.makedirs("results/tables", exist_ok=True)


def main():
    ensure_dirs()

    # 1) Datos (Iris escalado)
    X, y, classes = get_prepared_data()
    k = 3
    max_size = 50

    results = []

    # A) Baseline: online SIN restricción (random init)
    model_online = ControlledOnlineKMeans(n_clusters=k, random_state=42)
    model_online.init_random(X)
    y_pred_online = model_online.fit_predict_stream(X, max_size=None)

    m_online = evaluate_model(X, y_pred_online, y)
    results.append({"model": "online_controlled_no_constraint", **m_online})
    print("✅ Online controlado SIN restricción:", m_online)

    # B) Online + restricción (random init)
    model_con = ControlledOnlineKMeans(n_clusters=k, random_state=42)
    model_con.init_random(X)
    y_pred_con = model_con.fit_predict_stream(X, max_size=max_size)

    m_con = evaluate_model(X, y_pred_con, y)
    results.append({"model": f"online_controlled_constrained_max{max_size}", **m_con})
    print("✅ Online controlado CON restricción:", m_con)
    print("Tamaños:", model_con.counts.tolist())

    # Verificación fuerte de cardinalidad
    if (model_con.counts > max_size).any():
        raise RuntimeError("❌ Se violó la restricción de tamaño.")
    print("✅ Restricción cumplida (0 violaciones).")

    # C) Semi-supervisado + restricción (seed por clase)
    for per_class in [2, 5, 10]:
        labeled_idx = stratified_seed_indices(y, per_class=per_class, random_state=42)
        seed_centroids = centroids_from_labeled(X, y, labeled_idx, n_clusters=k)

        model_semi = ControlledOnlineKMeans(n_clusters=k, random_state=42)
        model_semi.init_with_centroids(seed_centroids)

        y_pred_semi = model_semi.fit_predict_stream(X, max_size=max_size)
        m_semi = evaluate_model(X, y_pred_semi, y)

        results.append({"model": f"semi_seeded_{per_class}perclass_max{max_size}", **m_semi})
        print(f"✅ Semi-supervisado ({per_class} por clase) + restricción:", m_semi)
        print("Tamaños:", model_semi.counts.tolist())

        if (model_semi.counts > max_size).any():
            raise RuntimeError("❌ Violación de tamaño en semi-supervisado.")

    # Guardar resultados
    df = pd.DataFrame(results)
    out_path = "results/tables/final_results.csv"
    df.to_csv(out_path, index=False)
    print("\n📁 Guardado:", out_path)
    print(df)


if __name__ == "__main__":
    main()
