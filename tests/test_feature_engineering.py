import pytest
import pandas as pd
import numpy as np
from ecomuserseg.feature_engineering import (calculate_delivery_delay, categorize_products, 
                                           impute_missing_values)

def test_calculate_delivery_delay():
    """Test du calcul du délai de livraison."""
    # Créer un DataFrame de test
    data = pd.DataFrame({
        'order_purchase_timestamp': pd.to_datetime(['2023-01-01', '2023-01-05']),
        'order_delivered_customer_date': pd.to_datetime(['2023-01-10', '2023-01-15'])
    })
    
    result = calculate_delivery_delay(data)
    
    assert 'delivery_delta_days' in result.columns
    assert 'order_delivered_customer_date' not in result.columns
    assert result['delivery_delta_days'].iloc[0] == 9
    assert result['delivery_delta_days'].iloc[1] == 10

def test_categorize_products():
    """Test de la catégorisation des produits."""
    data = pd.DataFrame({
        'product_category_name': ['fashion_bags', 'health_beauty', 'toys_kids', 'computers']
    })
    
    result = categorize_products(data)
    
    assert 'product_category' in result.columns
    assert 'product_category_name' not in result.columns
    assert result['product_category'].iloc[0] == 'fashion_clothing_accessories'
    assert result['product_category'].iloc[1] == 'health_beauty'
    assert result['product_category'].iloc[2] == 'toys_baby'
    assert result['product_category'].iloc[3] == 'technology'

def test_impute_missing_values():
    """Test de l'imputation des valeurs manquantes."""
    data = pd.DataFrame({
        'mean_review_score': [4.5, np.nan, 3.0],
        'mean_delivery_days': [10, 15, np.nan],
        'other_column': [1, 2, 3]
    })
    
    result = impute_missing_values(data, ['mean_review_score', 'mean_delivery_days'])
    
    # Vérifier qu'il n'y a plus de valeurs manquantes dans les colonnes spécifiées
    assert not result['mean_review_score'].isnull().any()
    assert not result['mean_delivery_days'].isnull().any()
    
    # Vérifier que les valeurs imputées sont cohérentes
    assert result['mean_review_score'].iloc[1] == 3.75  # moyenne de 4.5 et 3.0
    assert result['mean_delivery_days'].iloc[2] == 12.5  # moyenne de 10 et 15
