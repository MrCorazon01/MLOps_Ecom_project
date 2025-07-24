import pytest
import pandas as pd
import numpy as np
from ecomuserseg.business_analysis import create_business_interpretation, create_cluster_profiles

def test_create_business_interpretation():
    """Test de la création d'interprétation métier."""
    # Créer des données de test
    data = pd.DataFrame({
        'cluster': [0, 0, 1, 1, 2, 2],
        'total_spend': [100, 150, 500, 600, 50, 80],
        'nb_orders': [2, 3, 5, 6, 1, 1],
        'mean_review_score': [4.5, 4.0, 4.8, 4.9, 2.0, 2.5],
        'mean_delivery_days': [5, 6, 7, 8, 10, 12],
        'haversine_distance': [100, 120, 200, 220, 50, 60]
    })
    
    interpretation = create_business_interpretation(data)
    
    # Vérifications
    assert isinstance(interpretation, dict)
    assert len(interpretation) == 3  # 3 clusters
    
    for cluster_id in [0, 1, 2]:
        assert cluster_id in interpretation
        assert 'nom' in interpretation[cluster_id]
        assert 'taille' in interpretation[cluster_id]
        assert 'pourcentage' in interpretation[cluster_id]
        assert 'caracteristiques' in interpretation[cluster_id]

def test_create_cluster_profiles():
    """Test de la création des profils de clusters."""
    # Créer des données de test
    data = pd.DataFrame({
        'cluster': [0, 0, 1, 1],
        'feature1': [10, 20, 100, 200],
        'feature2': [5, 15, 50, 150]
    })
    
    profiles = create_cluster_profiles(data, ['feature1', 'feature2'])
    
    # Vérifications
    assert isinstance(profiles, pd.DataFrame)
    assert len(profiles) == 2  # 2 clusters
    assert list(profiles.columns) == ['feature1', 'feature2']
    assert profiles.loc[0, 'feature1'] == 15  # Moyenne de 10 et 20
    assert profiles.loc[1, 'feature1'] == 150  # Moyenne de 100 et 200

def test_empty_dataframe():
    """Test avec un DataFrame vide."""
    data = pd.DataFrame(columns=['cluster', 'total_spend'])
    
    interpretation = create_business_interpretation(data)
    
    # Doit retourner un dictionnaire vide
    assert isinstance(interpretation, dict)
    assert len(interpretation) == 0
