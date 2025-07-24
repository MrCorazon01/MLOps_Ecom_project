import pytest
import pandas as pd
import numpy as np
from ecomuserseg.monitoring import recommend_update_strategy, create_monitoring_dashboard, create_production_config

def test_recommend_update_strategy():
    """Test des recommandations de stratégie de mise à jour."""
    # Données de test
    drift_results = {
        'Scenario1': {
            'var1': {'is_drift': True},
            'var2': {'is_drift': False},
            'var3': {'is_drift': True}
        }
    }
    
    stability_results = {
        'Scenario1': {'silhouette_combined': 0.22}
    }
    
    quality_df = pd.DataFrame({
        'Scenario': ['Original', 'Scenario1'],
        'Silhouette_Combined': [0.229, 0.22]
    })
    
    recommendations = recommend_update_strategy(drift_results, stability_results, quality_df)
    
    # Vérifications
    assert isinstance(recommendations, dict)
    assert 'Scenario1' in recommendations
    assert 'drift_ratio' in recommendations['Scenario1']
    assert 'strategy' in recommendations['Scenario1']
    assert recommendations['Scenario1']['drift_ratio'] == 2/3  # 2 dérives sur 3 variables

def test_create_monitoring_dashboard():
    """Test de la création du tableau de bord."""
    metrics, thresholds = create_monitoring_dashboard()
    
    # Vérifications
    assert isinstance(metrics, dict)
    assert isinstance(thresholds, dict)
    assert 'Qualité du modèle' in metrics
    assert 'Silhouette dégradation' in thresholds

def test_create_production_config():
    """Test de la création de la configuration de production."""
    metrics = {'test_metric': 'description'}
    thresholds = {'test_threshold': 0.1}
    recommendations = {'scenario1': {'strategy': 'test'}}
    
    config = create_production_config(metrics, thresholds, recommendations)
    
    # Vérifications
    assert isinstance(config, dict)
    assert 'monitoring_metrics' in config
    assert 'alert_thresholds' in config
    assert 'drift_detection_params' in config
    assert config['monitoring_metrics'] == metrics
