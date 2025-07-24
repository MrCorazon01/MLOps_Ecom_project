import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'settings'))

def setup_mlflow_experiment(experiment_name="ecomuserseg_clustering"):
    """
    Configure l'expérience MLflow pour le projet de segmentation
    
    Parameters
    ----------
    experiment_name : str
        Nom de l'expérience MLflow
    
    Returns
    -------
    str
        ID de l'expérience
    """
    try:
        # Définir le répertoire MLflow dans le dossier ecomuserseg
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        mlruns_path = os.path.abspath(os.path.join(project_root, 'mlruns'))

        print(f"MLflow tracking URI: file://{mlruns_path}")
        
        # Configurer l'URI de tracking MLflow
        mlflow.set_tracking_uri(f"file://{mlruns_path}")
        
        # Créer le répertoire mlruns s'il n'existe pas
        os.makedirs(mlruns_path, exist_ok=True)
        
        # Créer ou récupérer l'expérience
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"Nouvelle expérience créée: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Expérience existante trouvée: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    except Exception as e:
        print(f"Erreur lors de la configuration MLflow: {e}")
        import traceback
        traceback.print_exc()
        return None

def log_data_preprocessing(data_initial_stats, data_final, run_name=None):
    """
    Log les métriques du prétraitement des données
    
    Parameters
    ----------
    data_initial_stats : pd.DataFrame
        Statistiques des données initiales (DataFrame avec colonnes 'dataset' et 'rows')
    data_final : pd.DataFrame
        Données finales après prétraitement
    run_name : str
        Nom du run MLflow
    """
    try:
        # S'assurer qu'on est dans la bonne expérience
        current_experiment = mlflow.get_experiment_by_name("ecomuserseg_preprocessing")
        if current_experiment:
            mlflow.set_experiment("ecomuserseg_preprocessing")
        
        with mlflow.start_run(run_name=run_name or f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            print(f"Démarrage du run MLflow: {run.info.run_id}")
            
            # Métriques de données
            initial_total_rows = data_initial_stats['rows'].sum()
            mlflow.log_metric("initial_total_rows", initial_total_rows)
            mlflow.log_metric("final_rows", len(data_final))
            mlflow.log_metric("final_columns", data_final.shape[1])
            mlflow.log_metric("data_reduction_ratio", len(data_final) / initial_total_rows)
            
            # Log des stats par dataset initial
            for _, row in data_initial_stats.iterrows():
                mlflow.log_metric(f"initial_{row['dataset']}_rows", row['rows'])
            
            # Qualité des données
            missing_values = data_final.isnull().sum().sum()
            mlflow.log_metric("missing_values_final", missing_values)
            mlflow.log_metric("missing_percentage_final", (missing_values / (len(data_final) * data_final.shape[1])) * 100)
            
            # Variables numériques vs catégorielles
            numeric_cols = len(data_final.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(data_final.select_dtypes(exclude=[np.number]).columns)
            mlflow.log_metric("numeric_variables", numeric_cols)
            mlflow.log_metric("categorical_variables", categorical_cols)
            
            # Tags
            mlflow.set_tag("stage", "preprocessing")
            mlflow.set_tag("data_version", datetime.now().strftime('%Y%m%d'))
            mlflow.set_tag("project", "ecomuserseg")
            
            print(f"Métriques loggées avec succès dans le run: {run.info.run_id}")
            return run.info.run_id
            
    except Exception as e:
        print(f"Erreur lors du logging MLflow: {e}")
        import traceback
        traceback.print_exc()
        return None

def log_clustering_experiment(model, X, labels, silhouette_score, optimal_k, 
                            cluster_profiles=None, business_interpretation=None, 
                            algorithm="kmeans", run_name=None):
    """
    Log une expérience de clustering complète
    
    Parameters
    ----------
    model : Pipeline
        Modèle de clustering entraîné
    X : pd.DataFrame
        Données d'entrée
    labels : np.array
        Labels des clusters
    silhouette_score : float
        Score de silhouette
    optimal_k : int
        Nombre optimal de clusters
    cluster_profiles : pd.DataFrame, optional
        Profils des clusters
    business_interpretation : dict, optional
        Interprétation métier
    algorithm : str
        Algorithme utilisé
    run_name : str
        Nom du run
    """
    with mlflow.start_run(run_name=run_name or f"{algorithm}_{optimal_k}clusters_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        
        # Paramètres du modèle
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("n_clusters", optimal_k)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        
        if hasattr(model.named_steps.get('kmeans', {}), 'random_state'):
            mlflow.log_param("random_state", model.named_steps['kmeans'].random_state)
        
        # Métriques de performance
        mlflow.log_metric("silhouette_score", silhouette_score)
        mlflow.log_metric("inertia", getattr(model.named_steps.get('kmeans', {}), 'inertia_', 0))
        
        # Distribution des clusters
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        for i, count in enumerate(cluster_counts):
            mlflow.log_metric(f"cluster_{i}_size", count)
            mlflow.log_metric(f"cluster_{i}_percentage", count / len(labels) * 100)
        
        # Enregistrer le modèle
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="clustering_model",
            registered_model_name=f"ecomuserseg_{algorithm}_model"
        )
        
        # Artifacts supplémentaires
        if cluster_profiles is not None:
            cluster_profiles.to_csv("cluster_profiles.csv")
            mlflow.log_artifact("cluster_profiles.csv")
            os.remove("cluster_profiles.csv")
        
        if business_interpretation is not None:
            with open("business_interpretation.pkl", "wb") as f:
                pickle.dump(business_interpretation, f)
            mlflow.log_artifact("business_interpretation.pkl")
            os.remove("business_interpretation.pkl")
        
        # Tags
        mlflow.set_tag("stage", "modeling")
        mlflow.set_tag("algorithm", algorithm)
        mlflow.set_tag("model_version", datetime.now().strftime('%Y%m%d'))
        
        return run.info.run_id

def log_model_comparison(results_dict, run_name=None):
    """
    Log la comparaison de plusieurs algorithmes
    
    Parameters
    ----------
    results_dict : dict
        Dictionnaire avec les résultats {algorithm: {'silhouette': score, 'n_clusters': k}}
    run_name : str
        Nom du run
    """
    with mlflow.start_run(run_name=run_name or f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        for algorithm, metrics in results_dict.items():
            mlflow.log_metric(f"{algorithm}_silhouette", metrics['silhouette'])
            mlflow.log_metric(f"{algorithm}_n_clusters", metrics['n_clusters'])
        
        # Meilleur modèle
        best_algorithm = max(results_dict.keys(), key=lambda x: results_dict[x]['silhouette'])
        mlflow.log_param("best_algorithm", best_algorithm)
        mlflow.log_metric("best_silhouette", results_dict[best_algorithm]['silhouette'])
        
        mlflow.set_tag("stage", "model_comparison")

def log_drift_detection(drift_results, stability_results, scenarios, run_name=None):
    """
    Log les résultats de détection de dérive
    
    Parameters
    ----------
    drift_results : dict
        Résultats de détection de dérive
    stability_results : dict
        Résultats de stabilité
    scenarios : dict
        Scénarios testés
    run_name : str
        Nom du run
    """
    with mlflow.start_run(run_name=run_name or f"drift_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        for scenario_name in scenarios.keys():
            # Métriques de dérive
            n_drifts = sum([r['is_drift'] for r in drift_results[scenario_name].values()])
            total_vars = len(drift_results[scenario_name])
            drift_ratio = n_drifts / total_vars
            
            mlflow.log_metric(f"{scenario_name}_drift_ratio", drift_ratio)
            mlflow.log_metric(f"{scenario_name}_n_drifts", n_drifts)
            
            # Métriques de stabilité
            mlflow.log_metric(f"{scenario_name}_silhouette_new", stability_results[scenario_name]['silhouette_new'])
            mlflow.log_metric(f"{scenario_name}_silhouette_combined", stability_results[scenario_name]['silhouette_combined'])
        
        mlflow.set_tag("stage", "monitoring")
        mlflow.set_tag("n_scenarios", len(scenarios))

def get_best_model(experiment_name="ecomuserseg_clustering", metric="silhouette_score"):
    """
    Récupère le meilleur modèle selon une métrique
    
    Parameters
    ----------
    experiment_name : str
        Nom de l'expérience
    metric : str
        Métrique pour sélectionner le meilleur modèle
    
    Returns
    -------
    mlflow.entities.model_registry.ModelVersion
        Meilleur modèle
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    # Rechercher le meilleur run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'modeling'",
        order_by=[f"metrics.{metric} DESC"]
    )
    
    if not runs.empty:
        best_run_id = runs.iloc[0]['run_id']
        return mlflow.sklearn.load_model(f"runs:/{best_run_id}/clustering_model")
    
    return None

def create_model_registry(run_id, model_info, production_criteria=None, model_name="ecomuserseg_production_model"):
    """
    Enregistre un modèle dans le registre MLflow avec validation des critères
    
    Parameters
    ----------
    run_id : str
        ID du run MLflow contenant le modèle
    model_info : dict
        Informations du modèle (algorithm, n_clusters, silhouette_score, etc.)
    production_criteria : dict, optional
        Critères de validation pour la production
    model_name : str
        Nom du modèle dans le registre
    
    Returns
    -------
    dict
        Résultats de l'enregistrement
    """
    try:
        print("=== ENREGISTREMENT POUR LA PRODUCTION ===")
        
        # Affichage des informations du modèle
        print("Informations du modèle sélectionné pour la production:")
        for key, value in model_info.items():
            print(f"  • {key}: {value}")
        
        # Critères par défaut si non spécifiés
        if production_criteria is None:
            production_criteria = {
                'min_silhouette': 0.2,
                'min_clusters': 2,
                'min_samples': 1000
            }
        
        # Validation des critères de production
        production_ready = (
            model_info.get('silhouette_score', 0) >= production_criteria['min_silhouette'] and
            model_info.get('n_clusters', 0) >= production_criteria['min_clusters'] and
            model_info.get('training_samples', 0) >= production_criteria['min_samples']
        )
        
        if production_ready:
            print("MODÈLE VALIDÉ POUR LA PRODUCTION")
            print("Prêt pour déploiement avec les métriques actuelles")
            target_stage = "Production"
        else:
            print("MODÈLE À AMÉLIORER")
            print("Vérifier les critères de qualité avant production")
            target_stage = "Staging"
        
        # Enregistrement dans le registre MLflow
        model_uri = f"runs:/{run_id}/clustering_model"
        
        try:
            # Enregistrer le modèle
            model_version = mlflow.register_model(model_uri, model_name)
            version_number = model_version.version
            print(f"✓ Modèle enregistré: {model_name} v{version_number}")
            
            # Définir le stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version_number,
                stage=target_stage
            )
            print(f"✓ Stage défini: {target_stage}")
            
            # Ajouter des tags
            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="algorithm",
                value=model_info.get('algorithm', 'Unknown')
            )
            
            client.set_model_version_tag(
                name=model_name,
                version=version_number,
                key="creation_date",
                value=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Description du modèle
            description = (f"Modèle {model_info.get('algorithm', 'ML')} avec "
                          f"{model_info.get('n_clusters', 'N')} clusters, "
                          f"score silhouette: {model_info.get('silhouette_score', 'N/A'):.3f}")
            
            client.update_model_version(
                name=model_name,
                version=version_number,
                description=description
            )
            
            return {
                'success': True,
                'model_name': model_name,
                'version': version_number,
                'stage': target_stage,
                'production_ready': production_ready,
                'model_uri': f"models:/{model_name}/{version_number}"
            }
            
        except Exception as e:
            print(f"❌ Erreur lors de l'enregistrement: {e}")
            return {
                'success': False,
                'error': str(e),
                'production_ready': production_ready
            }
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return {'success': False, 'error': str(e)}
