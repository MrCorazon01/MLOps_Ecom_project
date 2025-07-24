import pandas as pd
import numpy as np

def merge_products_categories(products, categories_en):
    """
    Fusionne les produits avec leurs catégories traduites et nettoie les colonnes.
    
    Parameters
    ----------
    products : pd.DataFrame
        DataFrame des produits
    categories_en : pd.DataFrame
        DataFrame des catégories traduites
    
    Returns
    -------
    pd.DataFrame
        DataFrame des produits avec catégories traduites
    """
    products = pd.merge(products, categories_en, how="left", on="product_category_name")
    del_features_list = ["product_category_name", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
    products.drop(del_features_list, axis=1, inplace=True)
    products = products.rename(columns={"product_category_name_english": "product_category_name"})
    return products

def merge_orders_items(order_items, orders):
    """
    Fusionne les commandes avec les items et filtre les commandes livrées.
    
    Parameters
    ----------
    order_items : pd.DataFrame
        DataFrame des items de commande
    orders : pd.DataFrame
        DataFrame des commandes
    
    Returns
    -------
    pd.DataFrame
        DataFrame fusionné filtré sur les commandes livrées
    """
    order_items = pd.merge(order_items, orders, how="left", on="order_id")
    del_features_list = ["seller_id", "shipping_limit_date", "order_approved_at", "order_delivered_carrier_date", "order_estimated_delivery_date"]
    order_items.drop(del_features_list, axis=1, inplace=True)
    # Garder uniquement les commandes livrées
    order_items = order_items[order_items["order_status"] == "delivered"]
    return order_items

def convert_datetime_columns(df, datetime_cols):
    """
    Convertit les colonnes spécifiées au format datetime.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à modifier
    datetime_cols : list
        Liste des noms de colonnes à convertir
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec colonnes converties
    """
    df_copy = df.copy()
    for col in datetime_cols:
        df_copy[col] = df_copy[col].astype('datetime64[ns]')
    return df_copy

def add_payment_info(order_items, order_payments):
    """
    Ajoute les informations de paiement agrégées aux items de commande.
    
    Parameters
    ----------
    order_items : pd.DataFrame
        DataFrame des items de commande
    order_payments : pd.DataFrame
        DataFrame des paiements
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec informations de paiement agrégées
    """
    group_payments = order_payments.groupby(by="order_id").agg({
        "payment_sequential": 'count',
        "payment_installments": 'sum'})
    order_items = pd.merge(order_items, group_payments, how="left", on="order_id")
    order_items = order_items.rename(columns={
        "payment_sequential": "nb_payment_sequential",
        "payment_installments": "sum_payment_installments"})
    return order_items

def add_review_info(order_items, order_reviews):
    """
    Ajoute les informations d'avis agrégées aux items de commande.
    
    Parameters
    ----------
    order_items : pd.DataFrame
        DataFrame des items de commande
    order_reviews : pd.DataFrame
        DataFrame des avis
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec informations d'avis agrégées
    """
    group_reviews = order_reviews.groupby("order_id").agg({
        "review_id": "count",
        "review_score": "mean"})
    order_items = pd.merge(order_items, group_reviews, how="left", on="order_id")
    order_items = order_items.rename(columns={"review_id": "is_reviewed"})
    order_items["is_reviewed"] = np.where(order_items["is_reviewed"] == 1, True, False)
    return order_items
