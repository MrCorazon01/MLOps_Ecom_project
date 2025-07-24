import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def plot_radar_clusters(cluster_data, title="Comparaison des clusters"):
    """
    Crée un radar plot pour comparer les profils des clusters
    
    Parameters
    ----------
    cluster_data : pd.DataFrame
        Données des clusters avec clusters en index et variables en colonnes
    title : str
        Titre du graphique
    """
    # Normalisation des données pour le radar plot
    scaler_radar = MinMaxScaler()
    cluster_data_scaled = pd.DataFrame(
        scaler_radar.fit_transform(cluster_data), 
        index=cluster_data.index, 
        columns=cluster_data.columns
    )
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, cluster_id in enumerate(cluster_data_scaled.index):
        fig.add_trace(go.Scatterpolar(
            r=cluster_data_scaled.loc[cluster_id].values,
            theta=cluster_data_scaled.columns,
            fill='toself',
            name=f'Cluster {cluster_id}',
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        title_font_size=16
    )
    
    fig.show()

def plot_pca_analysis(X, labels=None, n_components=None):
    """
    Analyse PCA avec visualisation de la variance expliquée et projection des clusters.
    
    Parameters
    ----------
    X : pd.DataFrame or np.array
        Données à analyser
    labels : np.array, optional
        Labels des clusters pour la coloration
    n_components : int, optional
        Nombre de composantes pour la PCA (si None, utilise toutes les composantes)
    
    Returns
    -------
    tuple
        (X_pca, variance_explained, n_components_95)
    """
    # PCA pour réduction dimensionnelle
    pca_pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=n_components, random_state=42))
    ])
    
    X_pca = pca_pipeline.fit_transform(X)
    pca_model = pca_pipeline.named_steps['pca']
    
    variance_explained = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    # Graphique de la variance expliquée
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(variance_explained) + 1), variance_explained * 100)
    plt.xlabel('Composante principale')
    plt.ylabel('Variance expliquée (%)')
    plt.title('Variance expliquée par composante')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'bo-')
    plt.axhline(y=95, color='r', linestyle='--', label='95% de variance')
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance cumulée (%)')
    plt.title('Variance cumulée')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Nombre de composantes pour 95% de variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Nombre de composantes pour 95% de variance: {n_components_95}")
    
    # Visualisation des clusters si labels fournis
    if labels is not None:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC1 ({variance_explained[0]*100:.1f}% de variance)')
        plt.ylabel(f'PC2 ({variance_explained[1]*100:.1f}% de variance)')
        plt.title('Clusters projetés sur les 2 premières composantes principales')
        plt.colorbar(scatter, label='Cluster')
        plt.show()
    
    return X_pca, variance_explained, n_components_95

def plot_cluster_distribution(data_clustered, cluster_col='cluster'):
    """
    Affiche la distribution des clusters.
    
    Parameters
    ----------
    data_clustered : pd.DataFrame
        DataFrame avec colonne cluster
    cluster_col : str
        Nom de la colonne contenant les clusters
    """
    plt.figure(figsize=(8, 6))
    cluster_counts = data_clustered[cluster_col].value_counts().sort_index()
    
    bars = plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel('Cluster')
    plt.ylabel('Nombre de clients')
    plt.title('Distribution des clients par cluster')
    
    # Ajouter les pourcentages sur les barres
    total = len(data_clustered)
    for bar, count in zip(bars, cluster_counts.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
