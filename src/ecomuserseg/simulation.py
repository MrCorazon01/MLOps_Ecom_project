import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'settings'))

def simulate_data_evolution(original_data, n_new_customers=5000, drift_intensity=0.1, random_state=42):
    """
    Simule l'évolution des données clients avec différents scénarios de dérive
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Données originales de référence
    n_new_customers : int
        Nombre de nouveaux clients à simuler
    drift_intensity : float
        Intensité de la dérive (0.0 à 1.0)
    random_state : int
        Graine aléatoire
    
    Returns
    -------
    pd.DataFrame
        Nouvelles données simulées
    """
    # Variables numériques (sans cluster)
    numerical_cols = original_data.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'cluster']
    
    # Statistiques des données originales
    original_stats = original_data[numerical_cols].describe()
    
    # Simulation de nouveaux clients
    np.random.seed(random_state)
    new_customers = []
    
    for i in range(n_new_customers):
        new_customer = {}
        
        for col in numerical_cols:
            mean_orig = original_stats.loc['mean', col]
            std_orig = original_stats.loc['std', col]
            
            # Simulation avec dérive progressive
            if col in ['total_spend', 'nb_orders']:
                # Augmentation légère des dépenses et commandes (croissance e-commerce)
                drift_factor = 1 + drift_intensity * np.random.normal(0.5, 0.2)
                new_value = np.random.normal(mean_orig * drift_factor, std_orig)
            elif col == 'mean_review_score':
                # Légère dégradation de la satisfaction
                drift_factor = 1 - drift_intensity * np.random.normal(0.1, 0.05)
                new_value = np.random.normal(mean_orig * drift_factor, std_orig)
            elif col == 'haversine_distance':
                # Expansion géographique
                drift_factor = 1 + drift_intensity * np.random.normal(0.3, 0.1)
                new_value = np.random.normal(mean_orig * drift_factor, std_orig)
            else:
                # Variation normale
                drift_factor = 1 + drift_intensity * np.random.normal(0, 0.1)
                new_value = np.random.normal(mean_orig * drift_factor, std_orig)
            
            # Contraintes réalistes
            if col == 'mean_review_score':
                new_value = np.clip(new_value, 1, 5)
            elif col in ['nb_orders', 'mean_nb_items']:
                new_value = max(1, int(new_value))
            elif col in ['total_spend', 'freight_ratio']:
                new_value = max(0, new_value)
            
            new_customer[col] = new_value
        
        new_customers.append(new_customer)
    
    new_data = pd.DataFrame(new_customers)
    
    # Ajout de variables catégorielles simulées
    if 'customer_state' in original_data.columns:
        states = original_data['customer_state'].value_counts()
        new_data['customer_state'] = np.random.choice(
            states.index, 
            size=len(new_data), 
            p=states.values/states.sum()
        )
    
    return new_data

def detect_data_drift(original_data, new_data, significance_level=0.05):
    """
    Détecte la dérive des données en comparant les distributions
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Données originales
    new_data : pd.DataFrame
        Nouvelles données
    significance_level : float
        Seuil de significativité pour les tests statistiques
    
    Returns
    -------
    dict
        Résultats des tests de dérive pour chaque variable
    """
    # Variables numériques communes
    numerical_cols = [col for col in original_data.select_dtypes(include=['number']).columns 
                     if col in new_data.columns and col != 'cluster']
    
    drift_results = {}
    
    print(f"{'Variable':<25}{'Test KS':<12}{'P-value':<12}{'Dérive?':<10}")
    print("-" * 60)
    
    for col in numerical_cols:
        # Test de Kolmogorov-Smirnov pour comparer les distributions
        ks_stat, p_value = ks_2samp(original_data[col].dropna(), new_data[col].dropna())
        
        is_drift = p_value < significance_level
        drift_results[col] = {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'is_drift': is_drift
        }
        
        status = "OUI" if is_drift else "NON"
        print(f"{col:<25}{ks_stat:<12.4f}{p_value:<12.4f}{status:<10}")
    
    # Résumé
    n_drifts = sum([result['is_drift'] for result in drift_results.values()])
    print(f"\nRésumé: {n_drifts}/{len(numerical_cols)} variables présentent une dérive significative")
    
    return drift_results

def evaluate_cluster_stability(original_data, new_data, model, scenario_name="New Data"):
    """
    Évalue la stabilité des clusters avec les nouvelles données
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Données originales avec clusters
    new_data : pd.DataFrame
        Nouvelles données
    model : Pipeline
        Modèle de clustering entraîné
    scenario_name : str
        Nom du scénario pour l'affichage
    
    Returns
    -------
    dict
        Résultats de l'évaluation de stabilité
    """
    # Variables utilisées pour le clustering
    numerical_cols = original_data.select_dtypes(include=['number']).columns
    
    try:
        from params import FEATURE_ENGINEERING_PARAMS
        categories_col = FEATURE_ENGINEERING_PARAMS["CATEGORY_COLUMNS"]
    except ImportError:
        categories_col = [col for col in original_data.columns if col in [
            'books_cds_media', 'fashion_clothing_accessories', 'flowers_gifts',
            'groceries_food_drink', 'health_beauty', 'home_furniture', 
            'other', 'sport', 'technology', 'toys_baby'
        ]]
    
    feature_cols = [col for col in numerical_cols if col not in categories_col and col != 'cluster']
    
    # Prédiction des clusters pour les nouvelles données
    new_clusters = model.predict(new_data[feature_cols])
    
    # Distribution des clusters
    original_dist = original_data['cluster'].value_counts(normalize=True).sort_index()
    new_dist = pd.Series(new_clusters).value_counts(normalize=True).sort_index()
    
    # Métriques de qualité
    X_new_scaled = model.named_steps['scaler'].transform(new_data[feature_cols])
    silhouette_new = silhouette_score(X_new_scaled, new_clusters)
    
    # Données combinées pour analyse globale
    combined_data = pd.concat([
        original_data[feature_cols].assign(cluster=original_data['cluster'], source='original'),
        new_data[feature_cols].assign(cluster=new_clusters, source='new')
    ])
    
    X_combined_scaled = model.named_steps['scaler'].transform(combined_data[feature_cols])
    silhouette_combined = silhouette_score(X_combined_scaled, combined_data['cluster'])
    
    return {
        'scenario': scenario_name,
        'new_clusters': new_clusters,
        'original_distribution': original_dist,
        'new_distribution': new_dist,
        'silhouette_new': silhouette_new,
        'silhouette_combined': silhouette_combined,
        'combined_data': combined_data
    }

def generate_multiple_scenarios(original_data, scenarios_config=None):
    """
    Génère plusieurs scénarios de dérive avec différentes intensités
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Données originales
    scenarios_config : dict
        Configuration des scénarios (nom: paramètres)
    
    Returns
    -------
    dict
        Dictionnaire des scénarios générés
    """
    if scenarios_config is None:
        scenarios_config = {
            'Faible dérive': {'n_new_customers': 3000, 'drift_intensity': 0.05},
            'Dérive modérée': {'n_new_customers': 3000, 'drift_intensity': 0.15},
            'Forte dérive': {'n_new_customers': 3000, 'drift_intensity': 0.30}
        }
    
    scenarios = {}
    for name, config in scenarios_config.items():
        scenarios[name] = simulate_data_evolution(original_data, **config)
    
    return scenarios
