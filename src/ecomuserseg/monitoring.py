import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'settings'))

def recommend_update_strategy(drift_results, stability_results, quality_df=None):
    """
    Recommande une stratégie de mise à jour basée sur l'analyse de dérive et stabilité
    
    Parameters
    ----------
    drift_results : dict
        Résultats de l'analyse de dérive par scénario
    stability_results : dict
        Résultats de l'analyse de stabilité par scénario
    quality_df : pd.DataFrame, optional
        DataFrame avec l'évolution de la qualité
    
    Returns
    -------
    dict
        Recommandations par scénario
    """
    strategies = {}
    
    for scenario in drift_results.keys():
        n_drifts = sum([r['is_drift'] for r in drift_results[scenario].values()])
        total_vars = len(drift_results[scenario])
        drift_ratio = n_drifts / total_vars
        
        # Calcul de la dégradation silhouette
        if quality_df is not None:
            original_silhouette = quality_df[quality_df['Scenario'] == 'Original']['Silhouette_Combined'].iloc[0]
            current_silhouette = stability_results[scenario]['silhouette_combined']
            silhouette_degradation = original_silhouette - current_silhouette
        else:
            silhouette_degradation = 0.0  # Valeur par défaut si pas de données
        
        # Critères adaptés aux données
        if silhouette_degradation < 0.01:  # Dégradation négligeable
            strategy = "✅ MAINTENIR - Modèle stable"
            action = "Continuer le monitoring. Performance excellente."
            
        elif silhouette_degradation < 0.05:  # Dégradation faible
            strategy = "⚠️ MONITORING RENFORCÉ"
            action = "Surveillance accrue mais pas de mise à jour nécessaire."
            
        elif silhouette_degradation < 0.1:  # Dégradation modérée
            strategy = "🔄 MISE À JOUR RECOMMANDÉE"
            action = "Réentraînement avec nouvelles données recommandé."
            
        else:  # Dégradation significative
            strategy = "🚨 REFONTE NÉCESSAIRE"
            action = "Revoir complètement la segmentation."
        
        strategies[scenario] = {
            'drift_ratio': drift_ratio,
            'silhouette_degradation': silhouette_degradation,
            'strategy': strategy,
            'action': action
        }
    
    return strategies

def create_monitoring_dashboard(alert_thresholds=None):
    """
    Crée un tableau de bord de monitoring pour la production
    
    Parameters
    ----------
    alert_thresholds : dict, optional
        Seuils d'alerte personnalisés
    
    Returns
    -------
    tuple
        (monitoring_metrics, alert_thresholds)
    """
    # Métriques clés à surveiller
    monitoring_metrics = {
        'Qualité du modèle': {
            'Silhouette Score': 'Cohésion interne des clusters (>0.2 recommandé)',
            'Inertie': 'Compacité des clusters (évolution relative)',
            'Taille des clusters': 'Distribution équilibrée (aucun cluster <5%)'
        },
        'Dérive des données': {
            'Variables en dérive': 'Nombre de variables avec p-value <0.05',
            'Score de dérive KS': 'Moyenne des statistiques KS',
            'Dérive géographique': 'Changement dans la répartition des états'
        },
        'Performance métier': {
            'Nouveaux clients/mois': 'Volume d\'intégration des nouvelles données',
            'Stabilité des segments': 'Variation des tailles de clusters',
            'Utilisabilité marketing': 'Feedback des équipes utilisatrices'
        }
    }
    
    # Seuils d'alerte par défaut
    if alert_thresholds is None:
        alert_thresholds = {
            'Silhouette dégradation': 0.01,
            'Variables en dérive (%)': 80,         
            'Variation taille cluster (%)': 15,    
            'Score KS moyen': 0.3                  
        }
    
    return monitoring_metrics, alert_thresholds

def simulate_auto_update_pipeline(new_data, original_model, original_data, threshold_config=None):
    """
    Simule un pipeline de mise à jour automatique
    
    Parameters
    ----------
    new_data : pd.DataFrame
        Nouvelles données
    original_model : Pipeline
        Modèle original
    original_data : pd.DataFrame
        Données originales
    threshold_config : dict, optional
        Configuration des seuils
    
    Returns
    -------
    str
        Décision du pipeline
    """
    from .simulation import detect_data_drift, evaluate_cluster_stability
    
    if threshold_config is None:
        threshold_config = {'drift_threshold': 0.3, 'silhouette_threshold': 0.05}
    
    print("PIPELINE DE MISE À JOUR AUTOMATIQUE")
    print("=" * 40)
    
    # Étape 1: Détection de dérive
    print("1. Détection de dérive...")
    drift_detected = detect_data_drift(original_data, new_data, significance_level=0.05)
    n_drifts = sum([r['is_drift'] for r in drift_detected.values()])
    drift_ratio = n_drifts / len(drift_detected)
    
    # Étape 2: Évaluation de la qualité
    print("2. Évaluation de la qualité...")
    stability_result = evaluate_cluster_stability(original_data, new_data, original_model, "Auto-update")
    
    # Calcul de la dégradation réelle (simulation)
    original_silhouette = 0.229  # Valeur de référence du projet
    silhouette_degradation = original_silhouette - stability_result['silhouette_combined']
    
    # Étape 3: Décision basée sur l'impact réel
    print("3. Prise de décision...")
    
    if silhouette_degradation > threshold_config.get('silhouette_threshold', 0.05):
        decision = "RÉENTRAÎNEMENT"
        print(f"   Décision: {decision} (dégradation {silhouette_degradation:.3f} > seuil)")
    elif silhouette_degradation > 0.01:
        decision = "MONITORING RENFORCÉ"
        print(f"   Décision: {decision} (dégradation {silhouette_degradation:.3f} modérée)")
    else:
        decision = "MAINTIEN"
        print(f"   Décision: {decision} (dégradation négligeable: {silhouette_degradation:.3f})")
        print(f"   Note: {drift_ratio:.1%} dérive détectée mais sans impact opérationnel")
    
    return decision

def create_production_config(monitoring_metrics, alert_thresholds, recommendations):
    """
    Crée la configuration complète pour la production
    
    Parameters
    ----------
    monitoring_metrics : dict
        Métriques de monitoring
    alert_thresholds : dict
        Seuils d'alerte
    recommendations : dict
        Recommandations par scénario
    
    Returns
    -------
    dict
        Configuration complète
    """
    try:
        from params import CLUSTERING_PARAMS
        random_state = CLUSTERING_PARAMS["RANDOM_STATE"]
    except ImportError:
        random_state = 42
    
    production_config = {
        'monitoring_metrics': monitoring_metrics,
        'alert_thresholds': alert_thresholds,
        'update_strategies': recommendations,
        'drift_detection_params': {
            'significance_level': 0.05,
            'test_method': 'kolmogorov_smirnov',
            'min_sample_size': 1000
        },
        'model_update_triggers': {
            'high_drift_threshold': 0.3,
            'silhouette_degradation_threshold': 0.1,
            'cluster_size_variation_threshold': 0.15
        },
        'retraining_schedule': {
            'frequency': 'monthly',
            'min_new_data_volume': 2000,
            'validation_method': 'silhouette_score'
        },
        'random_state': random_state
    }
    
    return production_config

def save_simulation_results(production_config, simulation_report, output_dir='../output'):
    """
    Sauvegarde les résultats de simulation
    
    Parameters
    ----------
    production_config : dict
        Configuration de production
    simulation_report : dict
        Rapport de simulation
    output_dir : str
        Répertoire de sortie
    """
    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde
    with open(f'{output_dir}/production_monitoring_config.pkl', 'wb') as f:
        pickle.dump(production_config, f)
    
    with open(f'{output_dir}/simulation_report.pkl', 'wb') as f:
        pickle.dump(simulation_report, f)
    
    print("Configuration et rapport sauvegardés avec succès!")
    print(f"- Configuration de monitoring: production_monitoring_config.pkl")
    print(f"- Rapport de simulation: simulation_report.pkl")

def analyze_quality_evolution(data_initial, stability_results, scenarios):
    """
    Analyse l'évolution de la qualité du clustering
    
    Parameters
    ----------
    data_initial : pd.DataFrame
        Données initiales
    stability_results : dict
        Résultats de stabilité par scénario
    scenarios : dict
        Scénarios de données
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec l'évolution de la qualité
    """
    # Silhouette score original (simulation)
    original_silhouette = 0.229  # Valeur de référence du projet
    
    # Compilation des résultats
    results_df = pd.DataFrame({
        'Scenario': ['Original'] + list(scenarios.keys()),
        'Silhouette_New': [original_silhouette] + [stability_results[s]['silhouette_new'] for s in scenarios.keys()],
        'Silhouette_Combined': [original_silhouette] + [stability_results[s]['silhouette_combined'] for s in scenarios.keys()]
    })
    
    return results_df
