import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'settings'))
from params import FEATURE_ENGINEERING_PARAMS

def calculate_delivery_delay(data):
    """
    Calcule le délai de livraison en jours.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame avec colonnes order_delivered_customer_date et order_purchase_timestamp
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec colonne delivery_delta_days ajoutée
    """
    data_copy = data.copy()
    data_copy["delivery_delta_days"] = (data_copy.order_delivered_customer_date - data_copy.order_purchase_timestamp).dt.round('1d').dt.days
    data_copy.drop("order_delivered_customer_date", axis=1, inplace=True)
    return data_copy

def categorize_products(data):
    """
    Regroupe les catégories produits en grandes familles selon les paramètres.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame avec colonne product_category_name
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec colonne product_category ajoutée
    """
    data_copy = data.copy()
    
    # Récupération des patterns depuis les paramètres
    categories = FEATURE_ENGINEERING_PARAMS["PRODUCT_CATEGORIES"]
    
    # Construction de la logique de catégorisation
    conditions = []
    choices = []
    
    for category, pattern in categories.items():
        condition = data_copy['product_category_name'].str.contains(pattern, na=False)
        conditions.append(condition)
        choices.append(category)
    
    # Application de la logique avec np.select
    data_copy['product_category'] = np.select(conditions, choices, default='other')
    data_copy.drop("product_category_name", axis=1, inplace=True)
    
    return data_copy

def calculate_categories_per_customer(data):
    """
    Calcule la répartition des achats par catégorie pour chaque client.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame avec colonnes customer_unique_id, product_category, order_item_id
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec répartition des catégories par client
    """
    categories_customers = data.groupby(["customer_unique_id", "product_category"]).agg({"order_item_id": "count"}).unstack()
    categories_customers.columns = categories_customers.columns.droplevel(0)
    categories_customers.fillna(0, inplace=True)
    categories_customers["total_items"] = categories_customers.sum(axis=1)
    for col in categories_customers.columns:
        if col != "total_items":
            categories_customers[col] = categories_customers[col] / categories_customers["total_items"]
    categories_customers.reset_index(inplace=True)
    return categories_customers

def calculate_products_per_order(data):
    """
    Calcule le nombre moyen d'articles par commande pour chaque client.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame avec colonnes customer_unique_id, order_id, order_item_id
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec nombre moyen d'items par commande
    """
    products_per_order = data.groupby(["customer_unique_id", "order_id"]).agg({"order_item_id": "count"})
    products_per_order = products_per_order.groupby("customer_unique_id").agg({"order_item_id": "mean"})
    return products_per_order

def calculate_purchase_recency(data):
    """
    Calcule la récurrence d'achat pour chaque client.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame avec colonnes customer_unique_id, order_purchase_timestamp
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec délai total entre première et dernière commande
    """
    recurencies = data.groupby("customer_unique_id").agg({"order_purchase_timestamp": ["min","max"]})
    recurencies.columns = recurencies.columns.droplevel(0)
    max_date = data["order_purchase_timestamp"].max()
    recurencies["order_total_delay"] = [(y[1] - y[0]).round('1d').days if y[1] != y[0] else (max_date - y[0]).round('1d').days for x,y in recurencies.iterrows()]
    recurencies.drop(["min", "max"], axis=1, inplace=True)
    return recurencies

def aggregate_customer_data(data):
    """
    Agrège toutes les données par client.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame des données détaillées
    
    Returns
    -------
    pd.DataFrame
        DataFrame agrégé par client
    """
    data_agg = data.groupby("customer_unique_id").agg({
        "order_id": "nunique",
        "price": "sum",
        "freight_value": "sum",
        "nb_payment_sequential": "mean",
        "sum_payment_installments": "mean",
        "review_score": "mean",
        "delivery_delta_days": "mean",
        "sale_month": lambda x: x.value_counts().index[0] if x.notnull().any() else np.nan
    })
    data_agg = data_agg.rename(columns={
        "order_id": "nb_orders",
        "price": "total_spend",
        "freight_value": "total_freight",
        "nb_payment_sequential": "mean_payment_sequential",
        "sum_payment_installments": "mean_payment_installments",
        "review_score": "mean_review_score",
        "delivery_delta_days": "mean_delivery_days",
        "sale_month": "favorite_sale_month"
    })
    return data_agg

def impute_missing_values(data_agg, columns_to_impute=None):
    """
    Impute les valeurs manquantes par la moyenne.
    
    Parameters
    ----------
    data_agg : pd.DataFrame
        DataFrame avec potentielles valeurs manquantes
    columns_to_impute : list, optional
        Liste des colonnes à imputer. Si None, utilise les paramètres par défaut.
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec valeurs imputées
    """
    if columns_to_impute is None:
        # Importation locale pour éviter les imports circulaires
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'settings'))
            from params import DATA_PARAMS
            columns_to_impute = DATA_PARAMS["IMPUTATION_COLUMNS"]
        except ImportError:
            # Fallback vers les valeurs par défaut
            columns_to_impute = ["mean_payment_sequential", "mean_payment_installments", 
                               "mean_review_score", "mean_delivery_days"]
    
    data_copy = data_agg.copy()
    for col in columns_to_impute:
        if col in data_copy.columns:
            data_copy[col].fillna(data_copy[col].mean(), inplace=True)
    
    return data_copy
