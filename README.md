# EcomUserSeg

EcomUserSeg est un projet MLOps destiné à la segmentation des utilisateurs pour les plateformes e-commerce. Il a été initialisé avec [Cookiecutter](https://cookiecutter.readthedocs.io/) pour fournir une structure modulaire et reproductible, adaptée aux workflows de science des données et de machine learning.

## Objectifs du projet

- Fournir des outils de segmentation des utilisateurs (RFM, clustering, etc.)
- Faciliter l'analyse des données e-commerce
- Structurer le projet pour le développement collaboratif, la reproductibilité et le déploiement MLOps
- Proposer des notebooks pour l'exploration, la modélisation et la visualisation

## Structure du projet

Le projet est organisé comme suit :

```
ecomuserseg/
│
├── src/                   # Code source principal (modules, scripts, pipelines ML)
│   └── ecomuserseg/       
│
├── notebooks/             # Notebooks Jupyter pour l'exploration, la modélisation et la démonstration
│   ├── 01_data_exploration.ipynb
│   ├── 02_segmentation_rfm.ipynb
│   └── 03_simulation_mise_a_jour.ipynb            
│
├── docs/                  # Documentation utilisateur et technique
│   ├── index.md
│   ├── installation.md
│   ├── usage.md
│   └── ...
│
├── tests/                 # Tests unitaires et d'intégration
│
├── input/                 # Fichiers de données d'entrée(datasets) (Lien vers les données Olist)
│
├── output/                # Fichiers de sortie générés (modèles, résultats, rapports) 
│
├── settings/              # Fichiers de configuration et paramètres
│
├── pyproject.toml         # Configuration du projet et des dépendances
├── README.md              # Présentation du projet (ce fichier)
├── .gitignore             # Fichiers et dossiers ignorés par git
└── ...                    # Autres fichiers (CONTRIBUTING.md, etc.)
```

### Détail des dossiers principaux

- **src/ecomuserseg/** : Code source Python (modules de traitement, scripts, pipelines ML, CLI éventuelle).
- **notebooks/** : Notebooks Jupyter pour l'exploration des données, la démonstration des méthodes de segmentation, et la visualisation des résultats.
- **docs/** : Documentation détaillée sur l'installation, l'utilisation, la contribution, etc.
- **tests/** : Scripts de tests pour garantir la qualité et la robustesse du code.

### Prérequis

- Python 3.10 ou supérieur
- [Poetry](https://python-poetry.org/) ou gestionnaire d'environnements virtuels (venv, conda, etc.)
- Dépendances listées dans `pyproject.toml`

### Mise en place

1. **Cloner le dépôt :**
   ```sh
   git clone https://github.com/mrcorazon_01/ecomuserseg.git
   cd ecomuserseg
   ```

2. **Créer un environnement virtuel et installer les dépendances :**
   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt  # ou utiliser poetry/pyproject.toml
   ```

3. **Lancer les notebooks ou scripts selon vos besoins.**

## Utilisation

Vous pouvez utiliser les notebooks pour explorer les données et tester les méthodes de segmentation, ou exécuter les scripts Python présents dans `src/`.

```python
# Exemple d'import d'un module du projet
from ecomuserseg import segmentation
```

## Notebooks

Le dossier `notebooks/` contient plusieurs notebooks illustrant l'utilisation du projet sur des cas concrets :
- Exploration des données brutes
- Segmentation RFM
- Visualisation des clusters
- Analyses avancées

N'hésitez pas à ouvrir ces notebooks pour mieux comprendre les fonctionnalités et adapter les analyses à vos besoins.

## Contribution

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour plus d'informations.

## Licence

Ce projet est sous licence MIT.

## Ressources

- [Documentation](docs/index.md)
- [Dépôt GitHub](https://github.com/mrcorazon_01/ecomuserseg)
