import pandas as pd
import numpy as np

def analyze_cluster(cluster_id, data_clustered, cluster_col='cluster'):
    """
    Analyse détaillée d'un cluster spécifique
    
    Parameters
    ----------
    cluster_id : int
        ID du cluster à analyser
    data_clustered : pd.DataFrame
        DataFrame avec les données et les clusters
    cluster_col : str
        Nom de la colonne contenant les clusters
    """
    cluster_data = data_clustered[data_clustered[cluster_col] == cluster_id]
    total_customers = len(data_clustered)
    cluster_size = len(cluster_data)
    
    print(f"\n{'='*50}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*50}")
    print(f"Taille: {cluster_size} clients ({cluster_size/total_customers*100:.1f}% du total)")
    
    # Statistiques principales
    print(f"\nCARACTÉRISTIQUES PRINCIPALES:")
    
    # Variables à analyser 
    variables_to_analyze = {
        'total_spend': 'Dépense totale moyenne',
        'nb_orders': 'Nombre moyen de commandes',
        'mean_review_score': 'Note moyenne',
        'mean_delivery_days': 'Délai de livraison moyen',
        'haversine_distance': 'Distance moyenne au siège',
        'customer_state': 'État principal',
        'favorite_sale_month': 'Mois d\'achat préféré'
    }
    
    for var, description in variables_to_analyze.items():
        if var in cluster_data.columns:
            if var in ['customer_state', 'favorite_sale_month']:
                # Variables catégorielles
                mode_val = cluster_data[var].mode()
                if not mode_val.empty:
                    print(f"- {description}: {mode_val.iloc[0]}")
            else:
                # Variables numériques
                mean_val = cluster_data[var].mean()
                if var == 'mean_review_score':
                    print(f"- {description}: {mean_val:.2f}/5")
                elif var in ['total_spend']:
                    print(f"- {description}: {mean_val:.2f} BRL")
                elif var in ['mean_delivery_days', 'haversine_distance']:
                    unit = 'jours' if 'days' in var else 'km'
                    print(f"- {description}: {mean_val:.1f} {unit}")
                else:
                    print(f"- {description}: {mean_val:.1f}")

def create_business_interpretation(data_clustered, cluster_col='cluster'):
    """
    Crée une interprétation métier des clusters
    
    Parameters
    ----------
    data_clustered : pd.DataFrame
        DataFrame avec les données et les clusters
    cluster_col : str
        Nom de la colonne contenant les clusters
    
    Returns
    -------
    dict
        Dictionnaire avec l'interprétation métier de chaque cluster
    """
    interpretations = {}
    
    # Calcul des médianes et moyennes globales pour comparaison
    global_stats = {}
    if 'total_spend' in data_clustered.columns:
        global_stats['spend_median'] = data_clustered['total_spend'].median()
        global_stats['spend_mean'] = data_clustered['total_spend'].mean()
    if 'nb_orders' in data_clustered.columns:
        global_stats['orders_median'] = data_clustered['nb_orders'].median()
    if 'mean_review_score' in data_clustered.columns:
        global_stats['rating_median'] = data_clustered['mean_review_score'].median()
    if 'mean_delivery_days' in data_clustered.columns:
        global_stats['delivery_median'] = data_clustered['mean_delivery_days'].median()
    if 'haversine_distance' in data_clustered.columns:
        global_stats['distance_median'] = data_clustered['haversine_distance'].median()
    
    # Collecte des moyennes par cluster pour comparaison relative
    cluster_stats = {}
    for cluster_id in sorted(data_clustered[cluster_col].unique()):
        cluster_data = data_clustered[data_clustered[cluster_col] == cluster_id]
        cluster_stats[cluster_id] = {
            'spend': cluster_data['total_spend'].mean() if 'total_spend' in cluster_data.columns else 0,
            'rating': cluster_data['mean_review_score'].mean() if 'mean_review_score' in cluster_data.columns else 0,
            'delivery': cluster_data['mean_delivery_days'].mean() if 'mean_delivery_days' in cluster_data.columns else 0,
            'distance': cluster_data['haversine_distance'].mean() if 'haversine_distance' in cluster_data.columns else 0
        }
    
    # Tri des clusters par dépense pour hiérarchisation
    clusters_by_spend = sorted(cluster_stats.items(), key=lambda x: x[1]['spend'], reverse=True)
    
    for cluster_id in sorted(data_clustered[cluster_col].unique()):
        cluster_data = data_clustered[data_clustered[cluster_col] == cluster_id]
        
        # Caractéristiques moyennes
        cluster_size = len(cluster_data)
        
        # Variables pour l'interprétation
        avg_spend = cluster_stats[cluster_id]['spend']
        avg_orders = cluster_data['nb_orders'].mean() if 'nb_orders' in cluster_data.columns else 0
        avg_rating = cluster_stats[cluster_id]['rating']
        avg_delivery = cluster_stats[cluster_id]['delivery']
        avg_distance = cluster_stats[cluster_id]['distance']
        
        value_type = "Clients Standards"  # par défaut
        
        # 1. D'abord identifier les clients à problème (priorité absolue)
        if avg_rating < 2.5:  # Très insatisfaits
            if avg_delivery > global_stats.get('delivery_median', 10) * 1.8:
                value_type = "Clients à Risque - Livraison"
            else:
                value_type = "Clients Insatisfaits"
        
        # 2. Puis classifier selon performance combinée (satisfaction + dépense + livraison)
        elif avg_rating >= 4.5 and avg_delivery <= global_stats.get('delivery_median', 10):
            # Très satisfaits ET livraison rapide
            spend_rank = [i for i, (cid, _) in enumerate(clusters_by_spend) if cid == cluster_id][0]
            if spend_rank == 0:  # Plus gros dépensier
                value_type = "Clients VIP Premium"
            elif avg_spend > global_stats.get('spend_mean', 100):
                value_type = "Clients VIP"
            else:
                value_type = "Clients Satisfaits"
        
        # 3. Clients avec livraison problématique mais toujours satisfaits
        elif avg_delivery > global_stats.get('delivery_median', 10) * 1.4:
            if avg_distance > global_stats.get('distance_median', 500) * 1.3:
                value_type = "Clients Éloignés"
            elif avg_rating >= 4.0:  # Satisfaits malgré les délais
                value_type = "Clients Tolérants"
            else:
                value_type = "Clients Livraison Lente"
        
        # 4. Classification par niveau de dépense et engagement
        elif avg_spend > global_stats.get('spend_median', 100) * 1.5:
            if avg_rating >= 4.0:
                spend_rank = [i for i, (cid, _) in enumerate(clusters_by_spend) if cid == cluster_id][0]
                if spend_rank <= 1:  # Top 2 des dépensiers
                    value_type = "Clients Premium"
                else:
                    value_type = "Gros Acheteurs"
            else:
                value_type = "Gros Acheteurs Mitigés"
        
        # 5. Clients moyens et occasionnels
        elif avg_spend < global_stats.get('spend_median', 100) * 0.9:
            if avg_rating >= 4.5:
                value_type = "Clients Satisfaits Économes"
            else:
                value_type = "Clients Occasionnels"
        
        # 6. Reste = clients standards
        else:
            if avg_rating >= 4.2:
                value_type = "Clients Réguliers Satisfaits"
            else:
                value_type = "Clients Standards"
        
        interpretations[cluster_id] = {
            'nom': value_type,
            'taille': cluster_size,
            'pourcentage': cluster_size / len(data_clustered) * 100,
            'caracteristiques': {
                'depense_moyenne': avg_spend,
                'nb_commandes': avg_orders,
                'satisfaction': avg_rating,
                'delai_livraison': avg_delivery,
                'distance': avg_distance
            }
        }
    
    return interpretations

def print_business_summary(business_interpretation):
    """
    Affiche un résumé de l'interprétation métier.
    
    Parameters
    ----------
    business_interpretation : dict
        Dictionnaire avec l'interprétation métier
    """
    print("INTERPRÉTATION MÉTIER DES CLUSTERS")
    print("="*60)
    
    for cluster_id, info in business_interpretation.items():
        print(f"\nCLUSTER {cluster_id}: {info['nom']}")
        print(f"   Taille: {info['taille']} clients ({info['pourcentage']:.1f}%)")
        print(f"   Dépense moyenne: {info['caracteristiques']['depense_moyenne']:.2f} BRL")
        print(f"   Commandes moyennes: {info['caracteristiques']['nb_commandes']:.1f}")
        print(f"   Satisfaction: {info['caracteristiques']['satisfaction']:.2f}/5")
        print(f"   Délai livraison: {info['caracteristiques']['delai_livraison']:.1f} jours")
        
        # Ajout d'une recommandation actionnable
        if "Risque" in info['nom'] or "Insatisfaits" in info['nom']:
            print(f"   ⚠️  ATTENTION: Segment prioritaire pour améliorer l'expérience client")
        elif "VIP" in info['nom'] or "Premium" in info['nom']:
            print(f"   ⭐ OPPORTUNITÉ: Segment à valoriser avec des offres premium")
        elif "Éloignés" in info['nom']:
            print(f"   📍 ACTION: Optimiser la logistique pour ce segment géographique")

def create_cluster_profiles(data_clustered, numerical_cols, cluster_col='cluster'):
    """
    Crée les profils moyens des clusters.
    
    Parameters
    ----------
    data_clustered : pd.DataFrame
        DataFrame avec les données et les clusters
    numerical_cols : list
        Liste des colonnes numériques à analyser
    cluster_col : str
        Nom de la colonne contenant les clusters
    
    Returns
    -------
    pd.DataFrame
        Profils moyens des clusters
    """
    return data_clustered.groupby(cluster_col)[numerical_cols].mean()

def save_clustering_results(model, data_clustered, cluster_profiles, business_interpretation, 
                          optimal_k, sil_score, output_dir='../output'):
    """
    Sauvegarde tous les résultats du clustering.
    
    Parameters
    ----------
    model : Pipeline
        Modèle de clustering entraîné
    data_clustered : pd.DataFrame
        Données avec clusters
    cluster_profiles : pd.DataFrame
        Profils des clusters
    business_interpretation : dict
        Interprétation métier
    optimal_k : int
        Nombre optimal de clusters
    sil_score : float
        Score de silhouette
    output_dir : str
        Répertoire de sortie
    """
    import pickle
    import os
    
    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde du modèle
    with open(f'{output_dir}/kmeans_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Sauvegarde des données avec clusters
    data_clustered.to_csv(f'{output_dir}/customers_with_clusters.csv')
    
    # Sauvegarde des profils de clusters
    cluster_profiles.to_csv(f'{output_dir}/cluster_profiles.csv')
    
    # Résumé des résultats
    results_summary = {
        'optimal_k': optimal_k,
        'silhouette_score': sil_score,
        'cluster_sizes': data_clustered['cluster'].value_counts().to_dict(),
        'business_interpretation': business_interpretation
    }
    
    with open(f'{output_dir}/clustering_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print("Modèle et résultats sauvegardés avec succès!")
    print(f"- Modèle K-Means: {optimal_k} clusters")
    print(f"- Score de silhouette: {sil_score:.3f}")
    print(f"- {len(data_clustered)} clients segmentés")
