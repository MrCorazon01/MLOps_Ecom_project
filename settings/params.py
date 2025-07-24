"""Paramètres du projet de segmentation client Olist."""

# Paramètres des données
DATA_PARAMS = {
    "ROOT_PATH": '../input/brazilian-ecommerce/',
    "OUTPUT_PATH": '../input/client-segmentation/',
    "FINAL_DATASET_NAME": 'olist-customers-segmentation.csv',
    
    # Fichiers de données
    "DATA_FILES": {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv", 
        "order_items": "olist_order_items_dataset.csv",
        "order_payments": "olist_order_payments_dataset.csv",
        "order_reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "categories": "product_category_name_translation.csv",
        "geolocation": "olist_geolocation_dataset.csv"
    },
    
    # Coordonnées du siège d'Olist pour calcul de distance
    "OLIST_COORDINATES": {
        "latitude": -25.43045,
        "longitude": -49.29207
    },
    
    # Colonnes à supprimer lors du prétraitement
    "COLUMNS_TO_DROP": {
        "products": ["product_category_name", "product_weight_g", "product_length_cm", 
                    "product_height_cm", "product_width_cm"],
        "orders": ["seller_id", "shipping_limit_date", "order_approved_at", 
                  "order_delivered_carrier_date", "order_estimated_delivery_date"]
    },
    
    # Colonnes datetime à convertir
    "DATETIME_COLUMNS": ["order_purchase_timestamp", "order_delivered_customer_date"],
    
    # Variables pour l'imputation
    "IMPUTATION_COLUMNS": ["mean_payment_sequential", "mean_payment_installments", 
                          "mean_review_score", "mean_delivery_days"]
}

# Paramètres du feature engineering
FEATURE_ENGINEERING_PARAMS = {
    # Mapping des catégories produits
    "PRODUCT_CATEGORIES": {
        "fashion_clothing_accessories": r"fashio|luggage",
        "health_beauty": r"health|beauty|perfum", 
        "toys_baby": r"toy|baby|diaper",
        "books_cds_media": r"book|cd|dvd|media",
        "groceries_food_drink": r"grocer|food|drink",
        "technology": r"phon|compu|tablet|electro|consol",
        "home_furniture": r"home|furnitur|garden|bath|house|applianc",
        "flowers_gifts": r"flow|gift|stuff",
        "sport": r"sport"
    },
    
    # Variables catégories produits finales
    "CATEGORY_COLUMNS": [
        'books_cds_media', 'fashion_clothing_accessories', 'flowers_gifts',
        'groceries_food_drink', 'health_beauty', 'home_furniture', 
        'other', 'sport', 'technology', 'toys_baby'
    ],
    
    # Variables pour l'analyse des outliers
    "OUTLIER_ANALYSIS_FEATURES": [
        'total_spend', 'nb_orders', 'mean_payment_sequential',
        'mean_payment_installments', 'mean_review_score', 'mean_delivery_days'
    ]
}

# Paramètres du clustering  
CLUSTERING_PARAMS = {
    "OPTIMAL_K_RANGE": (3, 12),
    "RANDOM_STATE": 42,
    "N_INIT": 10,
    "STABILITY_TEST_ITERATIONS": 10,
    
    # Paramètres DBSCAN
    "DBSCAN_EPS_RANGE": (0.1, 1.0, 0.1),  # start, stop, step
    "DBSCAN_MIN_SAMPLES": 5,
    
    # Paramètres classification hiérarchique
    "HIERARCHICAL_MAX_CLUSTERS": 8,
    "HIERARCHICAL_LINKAGE": 'ward',
    
    # Seuils de qualité
    "MIN_SILHOUETTE_SCORE": 0.2,
    "MAX_NOISE_RATIO": 0.5  # Pour DBSCAN
}

# Paramètres de visualisation
VISUALIZATION_PARAMS = {
    "FIGURE_SIZE_LARGE": (18, 8),
    "FIGURE_SIZE_MEDIUM": (12, 6), 
    "FIGURE_SIZE_SMALL": (8, 6),
    "COLOR_PALETTE": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
    "PRIMARY_COLOR": "#00d994",
    "PLOT_STYLE": 'seaborn-v0_8'
}

# Paramètres d'évaluation métier
BUSINESS_PARAMS = {
    # Seuils pour l'interprétation métier
    "HIGH_RATING_THRESHOLD": 4.0,
    "LOW_RATING_THRESHOLD": 3.5,
    "HIGH_SPEND_PERCENTILE": 0.5,  # Médiane
    "HIGH_ORDERS_PERCENTILE": 0.5,
    "HIGH_DISTANCE_PERCENTILE": 0.5,
    
    # Types de clients
    "CUSTOMER_TYPES": {
        "premium": "Clients Premium",
        "loyal": "Clients Fidèles", 
        "dissatisfied": "Clients Insatisfaits",
        "distant": "Clients Éloignés",
        "occasional": "Clients Occasionnels",
        "standard": "Clients Standards"
    }
}

# Chemins de sauvegarde
OUTPUT_PARAMS = {
    "MODEL_OUTPUT_DIR": '../output',
    "MODEL_FILENAME": 'kmeans_model.pkl',
    "CLUSTERED_DATA_FILENAME": 'customers_with_clusters.csv',
    "CLUSTER_PROFILES_FILENAME": 'cluster_profiles.csv', 
    "RESULTS_SUMMARY_FILENAME": 'clustering_results.pkl'
}

# Graine aléatoire globale
RANDOM_STATE = 42

# Paramètres de simulation et monitoring
SIMULATION_PARAMS = {
    # Configuration des scénarios de dérive
    "SCENARIOS_CONFIG": {
        'Faible dérive': {'n_new_customers': 3000, 'drift_intensity': 0.05},
        'Dérive modérée': {'n_new_customers': 3000, 'drift_intensity': 0.15},
        'Forte dérive': {'n_new_customers': 3000, 'drift_intensity': 0.30}
    },
    
    # Paramètres de détection de dérive
    "DRIFT_DETECTION": {
        "significance_level": 0.05,
        "test_method": "kolmogorov_smirnov",
        "min_sample_size": 1000
    },
    
    # Seuils d'alerte pour le monitoring
    "ALERT_THRESHOLDS": {
        "silhouette_degradation": 0.01,
        "variables_drift_percentage": 80,
        "cluster_size_variation": 15,
        "ks_score_mean": 0.3
    },
    
    # Configuration du pipeline automatique
    "AUTO_UPDATE_CONFIG": {
        "drift_threshold": 0.3,
        "silhouette_threshold": 0.05,
        "monitoring_threshold": 0.01
    }
}

# Paramètres de maintenance du modèle
MAINTENANCE_PARAMS = {
    "retraining_schedule": {
        "frequency": "monthly",
        "min_new_data_volume": 2000,
        "validation_method": "silhouette_score"
    },
    
    "model_update_triggers": {
        "high_drift_threshold": 0.3,
        "silhouette_degradation_threshold": 0.1,
        "cluster_size_variation_threshold": 0.15
    },
    
    "quality_reference": {
        "baseline_silhouette": 0.229,
        "minimum_acceptable_silhouette": 0.2
    }
}
