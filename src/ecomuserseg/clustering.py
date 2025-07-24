import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, adjusted_rand_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from scipy.cluster.hierarchy import linkage, dendrogram

def evaluate_clustering(X, name="Dataset", k_range=(3, 12)):
    """
    Évalue différents nombres de clusters avec la méthode du coude et le coefficient de silhouette
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Données à clustériser
    name : str
        Nom du dataset pour l'affichage
    k_range : tuple
        Range des valeurs K à tester
    
    Returns
    -------
    int
        Nombre optimal de clusters
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Méthode du coude - Distortion
    kmeans_viz = Pipeline([
        ("scaler", MinMaxScaler()),
        ("kmeans", KElbowVisualizer(KMeans(random_state=42), K=k_range, metric='distortion', ax=axes[0]))
    ])
    kmeans_viz.fit(X)
    optimal_k_distortion = kmeans_viz.named_steps['kmeans'].elbow_value_
    kmeans_viz.named_steps['kmeans'].finalize()
    
    # 2. Méthode du coude - Silhouette
    kmeans_viz_sil = Pipeline([
        ("scaler", MinMaxScaler()),
        ("kmeans", KElbowVisualizer(KMeans(random_state=42), K=k_range, metric='silhouette', ax=axes[1]))
    ])
    kmeans_viz_sil.fit(X)
    optimal_k_silhouette = kmeans_viz_sil.named_steps['kmeans'].elbow_value_
    kmeans_viz_sil.named_steps['kmeans'].finalize()
    
    # 3. Visualisation du coefficient de silhouette pour K optimal
    best_k = optimal_k_silhouette if optimal_k_silhouette else optimal_k_distortion
    sil_viz = Pipeline([
        ("scaler", MinMaxScaler()),
        ("kmeans", SilhouetteVisualizer(KMeans(n_clusters=best_k, random_state=42), ax=axes[2]))
    ])
    sil_viz.fit(X)
    sil_viz.named_steps['kmeans'].finalize()
    
    plt.suptitle(f'Optimisation K-Means - {name}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return best_k

def create_kmeans_model(X, n_clusters, random_state=42):
    """
    Crée et entraîne un modèle K-Means avec preprocessing.
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Données à clustériser
    n_clusters : int
        Nombre de clusters
    random_state : int
        Graine aléatoire
    
    Returns
    -------
    Pipeline
        Modèle entraîné
    np.array
        Labels des clusters
    float
        Score de silhouette
    """
    model = Pipeline([
        ("scaler", MinMaxScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10))
    ])
    
    model.fit(X)
    labels = model.named_steps['kmeans'].labels_
    
    X_scaled = model.named_steps['scaler'].transform(X)
    sil_score = silhouette_score(X_scaled, labels)
    
    return model, labels, sil_score

def test_initialization_stability(X, n_clusters, n_iterations=10):
    """
    Test de la stabilité du clustering à l'initialisation
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Données à clustériser
    n_clusters : int
        Nombre de clusters
    n_iterations : int
        Nombre d'itérations à tester
    
    Returns
    -------
    tuple
        (ari_scores, sil_scores)
    """
    # Labels de référence
    ref_model = Pipeline([
        ("scaler", MinMaxScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
    ])
    ref_model.fit(X)
    ref_labels = ref_model.named_steps['kmeans'].labels_
    
    print("Test de stabilité à l'initialisation")
    print("="*50)
    print(f"{'Itération':<10}{'ARI':<10}{'Silhouette':<12}{'Inertie':<12}")
    print("-"*50)
    
    ari_scores = []
    sil_scores = []
    
    for i in range(n_iterations):
        model = Pipeline([
            ("scaler", MinMaxScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=i, n_init=1))
        ])
        model.fit(X)
        labels = model.named_steps['kmeans'].labels_
        
        ari = adjusted_rand_score(ref_labels, labels)
        sil = silhouette_score(model.named_steps['scaler'].transform(X), labels)
        inertia = model.named_steps['kmeans'].inertia_
        
        ari_scores.append(ari)
        sil_scores.append(sil)
        
        print(f"{i:<10}{ari:<10.3f}{sil:<12.3f}{inertia:<12.0f}")
    
    print("-"*50)
    print(f"{'Moyenne':<10}{np.mean(ari_scores):<10.3f}{np.mean(sil_scores):<12.3f}")
    print(f"{'Écart-type':<10}{np.std(ari_scores):<10.3f}{np.std(sil_scores):<12.3f}")
    
    return ari_scores, sil_scores

def test_dbscan(X, eps_range=None):
    """
    Test de différentes valeurs d'epsilon pour DBSCAN
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Données à clustériser
    eps_range : np.array
        Range des valeurs epsilon à tester
    
    Returns
    -------
    pd.DataFrame
        Résultats des tests DBSCAN
    """
    if eps_range is None:
        eps_range = np.arange(0.1, 1.0, 0.1)
        
    results = []
    X_scaled = MinMaxScaler().fit_transform(X)
    
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1 and n_noise < len(X) * 0.5:
            sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 else -1
            results.append({
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / len(X),
                'silhouette': sil_score
            })
    
    return pd.DataFrame(results)

def test_hierarchical_clustering(X, max_clusters=8):
    """
    Test de la classification hiérarchique
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Données à clustériser
    max_clusters : int
        Nombre maximum de clusters à tester
    
    Returns
    -------
    tuple
        (best_n_clusters, best_score)
    """
    X_scaled = MinMaxScaler().fit_transform(X)
    
    # Calcul de la matrice de liaison
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # Test différents nombres de clusters
    sil_scores = []
    for n_clusters in range(2, max_clusters + 1):
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = hierarchical.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)
        sil_scores.append(sil_score)
    
    # Dendrogramme
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.title('Dendrogramme (Classification Hiérarchique)')
    plt.xlabel('Échantillons')
    plt.ylabel('Distance')
    
    # Scores de silhouette
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), sil_scores, 'bo-')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score de silhouette')
    plt.title('Scores de silhouette - CAH')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    best_n_clusters = np.argmax(sil_scores) + 2
    best_score = max(sil_scores)
    
    return best_n_clusters, best_score

def visualize_intercluster_distance(X, n_clusters):
    """
    Visualise les distances inter-clusters.
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Données à clustériser
    n_clusters : int
        Nombre de clusters
    """
    distance_viz = Pipeline([
        ("scaler", MinMaxScaler()),
        ("kmeans", InterclusterDistance(KMeans(n_clusters=n_clusters, random_state=42)))
    ])
    distance_viz.fit(X)
    distance_viz.named_steps['kmeans'].show()
