import pytest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from ecomuserseg.simulation import simulate_data_evolution, detect_data_drift, evaluate_cluster_stability

def test_simulate_data_evolution():
    """Test de la simulation d'évolution des données."""
    # Créer des données de test
    original_data = pd.DataFrame({
        'total_spend': np.random.normal(100, 20, 100),
        'nb_orders': np.random.poisson(2, 100),
        'mean_review_score': np.random.uniform(3, 5, 100),
        'haversine_distance': np.random.normal(500, 100, 100)
    })
    
    # Simuler l'évolution
    new_data = simulate_data_evolution(original_data, n_new_customers=50, drift_intensity=0.1)
    
    # Vérifications
    assert len(new_data) == 50
    assert set(new_data.columns) == set(original_data.columns)
    assert all(new_data['mean_review_score'].between(1, 5))
    assert all(new_data['nb_orders'] >= 1)
    assert all(new_data['total_spend'] >= 0)

def test_detect_data_drift():
    """Test de la détection de dérive."""
    # Données originales
    original_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(10, 2, 100)
    })
    
    # Nouvelles données sans dérive
    new_data_no_drift = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(10, 2, 100)
    })
    
    # Nouvelles données avec dérive
    new_data_drift = pd.DataFrame({
        'feature1': np.random.normal(2, 1, 100),  # Dérive de la moyenne
        'feature2': np.random.normal(10, 2, 100)
    })
    
    # Test sans dérive
    results_no_drift = detect_data_drift(original_data, new_data_no_drift)
    
    # Test avec dérive
    results_drift = detect_data_drift(original_data, new_data_drift)
    
    # Vérifications
    assert isinstance(results_no_drift, dict)
    assert isinstance(results_drift, dict)
    assert 'feature1' in results_drift
    assert 'ks_statistic' in results_drift['feature1']
    assert 'p_value' in results_drift['feature1']
    assert 'is_drift' in results_drift['feature1']

def test_evaluate_cluster_stability():
    """Test de l'évaluation de la stabilité des clusters."""
    # Créer des données avec clusters
    np.random.seed(42)
    original_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'cluster': np.random.choice([0, 1, 2], 100)
    })
    
    new_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 50),
        'feature2': np.random.normal(0, 1, 50)
    })
    
    # Créer un modèle simple
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('kmeans', KMeans(n_clusters=3, random_state=42))
    ])
    model.fit(original_data[['feature1', 'feature2']])
    
    # Évaluer la stabilité
    results = evaluate_cluster_stability(original_data, new_data, model)
    
    # Vérifications
    assert isinstance(results, dict)
    assert 'scenario' in results
    assert 'new_clusters' in results
    assert 'silhouette_new' in results
    assert 'silhouette_combined' in results
    assert len(results['new_clusters']) == 50
