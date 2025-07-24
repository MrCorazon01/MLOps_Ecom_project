import pytest
import pandas as pd
import numpy as np
from ecomuserseg.utils import haversine_distance, check_datasets_info

def test_haversine_distance():
    """Test de la fonction haversine_distance."""
    # Test avec des coordonnées connues (Paris -> Londres)
    lat1, lng1 = 48.8566, 2.3522  # Paris
    lat2, lng2 = 51.5074, -0.1278  # Londres
    
    distance = haversine_distance(lat1, lng1, lat2, lng2)
    
    # La distance Paris-Londres est d'environ 344 km
    assert 340 <= distance <= 350
    
def test_haversine_distance_same_point():
    """Test avec le même point (distance = 0)."""
    distance = haversine_distance(0, 0, 0, 0)
    assert distance == 0

def test_check_datasets_info(capsys):
    """Test de la fonction check_datasets_info."""
    # Créer des datasets de test
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'x': [1, 1, 2], 'y': [None, 2, 3]})
    
    datasets = {'test1': df1, 'test2': df2}
    
    check_datasets_info(datasets)
    captured = capsys.readouterr()
    
    assert 'test1' in captured.out
    assert 'test2' in captured.out
    assert 'NA=0' in captured.out  # df1 n'a pas de valeurs manquantes
    assert 'NA=1' in captured.out  # df2 a 1 valeur manquante
