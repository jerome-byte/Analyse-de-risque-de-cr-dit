# Modèle de Scoring Crédit — Prédiction du Risque de Défaut

**Auteur :** AKOTA Yao Jérôme

Projet de Data Science / Machine Learning appliqué au contexte bancaire ouest-africain .
Consultez les images de l'API et du modèle en ligne dans le dossier docs, ou accédez directement à l'interface via le lien indiqué dans le README pour tester l'envoi de fichiers CSV au modèle et observer les réponses.

---

## 📋 Table des matières

- [Contexte](#-contexte)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture du projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
  - [1. Notebook de Data Science](#1-notebook-de-data-science)
  - [2. API FastAPI](#2-api-fastapi)
  - [3. Interface Web](#3-interface-web)
- [Endpoints de l'API](#-endpoints-de-lapi)
- [Format du fichier CSV](#-format-du-fichier-csv)
- [Modèle ML](#-modèle-ml)
- [Déploiement](#-déploiement)
- [Tests](#-tests)
- [Limites et perspectives](#-limites-et-perspectives)
- [Auteur](#-auteur)

---

## Contexte

La **Banque Ouest Africaine de Développement (BOAD)** finance des projets de développement dans les **8 pays de l'UEMOA** : Bénin, Burkina Faso, Côte d'Ivoire, Guinée-Bissau, Mali, Niger, Sénégal et Togo.

Ce projet construit un modèle capable de **prédire, au moment de la demande de prêt, si un emprunteur présente un risque élevé de défaut** (classification binaire : `0` = bon remboursement, `1` = défaut).

### Objectifs

1. **Construire un jeu de données réaliste** simulant 1 000 demandes de prêt dans la zone UEMOA
2. **Explorer et analyser** les facteurs de risque (EDA)
3. **Entraîner et comparer** deux modèles de classification
4. **Déployer une API REST + interface web** pour la mise en production
5. **Formuler des recommandations** pour la gestion du risque de crédit

---

##  Fonctionnalités

###  Modèle de Machine Learning
- ✅ Deux modèles comparés : **Régression Logistique** et **Random Forest**
- ✅ Pipeline complet : EDA, prétraitement, entraînement, évaluation
- ✅ Métriques complètes : accuracy, précision, rappel, F1, AUC-ROC
- ✅ **AUC = 0.972** (Régression Logistique)
- ✅ Modèle sauvegardé en `.pkl` avec métadonnées JSON

###  API REST (FastAPI)
- ✅ Endpoints REST documentés automatiquement (Swagger UI)
- ✅ Validation des entrées avec Pydantic
- ✅ Prédiction unitaire, par lot (JSON) et par fichier CSV
- ✅ Gestion des erreurs centralisée
- ✅ CORS configuré pour les frontends modernes

###  Interface Web
- ✅ Interface professionnelle avec **4 onglets** :
  - **Prédiction individuelle** : formulaire interactif
  - **Import CSV** : upload et analyse en batch (max 1000 clients)
  - **Informations du modèle** : détails techniques et performances
  - **Aide** : documentation intégrée
- ✅ Affichage clair des résultats : **Bon client** 🟢 / **Mauvais client** 🔴
- ✅ Statistiques globales sur les imports CSV
- ✅ Export des résultats en CSV enrichi
- ✅ Design responsive (desktop, tablette, mobile)

###  Artefacts ML
- ✅ Modèle entraîné : `model_lr.pkl`, `model_rf.pkl`
- ✅ Scaler : `scaler.pkl`
- ✅ Métadonnées : `metadata.json` (features, performances, version)

---

##  Architecture du projet

```
scoring-credit-boad/
│
├── 📓 scoring_credit.ipynb           # Notebook de Data Science
├── 🐍 app.py                          # API FastAPI + Interface Web
├── 📄 README.md                       # Ce fichier
├── 📄 requirements.txt                # Dépendances Python
├── 📄 Dockerfile                      # Image Docker
├── 📄 docker-compose.yml              # Orchestration Docker
├── 📄 .gitignore                      # Fichiers ignorés par Git
├── 📄 .dockerignore                   # Fichiers ignorés par Docker
│
├── 📁 model_dossier/                  # Artefacts du modèle
│   ├── model_lr.pkl                  # Régression Logistique
│   ├── model_rf.pkl                  # Random Forest
│   ├── scaler.pkl                    # StandardScaler
│   └── metadata.json                 # Métadonnées du pipeline
│
├── 📁 templates/                      # Templates HTML
│   └── index.html                    # Interface web (4 onglets)
│
├── 📁 static/                         # Fichiers statiques
│   ├── 📁 css/
│   │   └── style.css                 # Styles professionnels
│   └── 📁 js/
│       └── app.js                    # Logique frontend
│
└── 📁 .vscode/                        # Config VS Code
    └── settings.json                 # Interpréteur Python (venv)
```

---

##  Installation

### Prérequis

- **Python 3.11+** (recommandé : 3.11 ou 3.12)
- **pip** et **venv**
- **Git**

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/jerome-byte/Analyse-de-risque-de-cr-dit.git
cd dossier

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement virtuel
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Windows (CMD)
venv\Scripts\activate.bat
# Windows (Git Bash / MINGW64)
source venv/Scripts/activate
# Mac/Linux
source venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. (Optionnel) Régénérer le modèle en exécutant le notebook
jupyter notebook scoring_credit.ipynb
```

### `requirements.txt`

```txt

# Dépendances pour l'API Scoring Crédit

fastapi==0.115.0
uvicorn[standard]==0.32.0
pydantic==2.9.2
joblib==1.4.2
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.9.0
python-multipart==0.0.12
jinja2==3.1.4

# Dépendances pour le notebook
matplotlib==3.9.2
seaborn==0.13.2
jupyter==1.1.1
notebook==7.3.1
```

---

##  Utilisation

### 1. Notebook de Data Science

Le notebook `scoring_credit.ipynb` couvre tout le pipeline :

| # | Étape | Description |
|---|-------|-------------|
| 1 | **Création des données** | 1 000 demandes de prêt simulées |
| 2 | **EDA** | Distribution cible, analyses univariées, bivariées, corrélations |
| 3 | **Prétraitement** | Encodage one-hot, train/test split, standardisation |
| 4 | **Modélisation** | Régression Logistique + Random Forest |
| 5 | **Évaluation** | Matrices de confusion, ROC/AUC, comparaison |
| 6 | **Interprétation** | Facteurs de risque clés, recommandations |
| 7 | **Sauvegarde** | Export des `.pkl` + `metadata.json` |

**Exécution :**
```bash
jupyter notebook scoring_credit.ipynb
# puis Kernel → Restart & Run All
```

> **Note** : le chemin de sauvegarde du CSV dans le notebook est codé en dur pour Windows (`C:\Users\...`). Adaptez-le à votre environnement avant d'exécuter.

### 2. API FastAPI

```bash
# Lancer l'API en mode développement (rechargement auto)
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Lancer en production (avec workers)
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

L'API démarre sur **http://localhost:8000**

**Sortie attendue :**
```

 Démarrage de l'API Scoring Crédit 

✅ Modèle chargé : LogisticRegression
✅ Scaler chargé
✅ AUC du modèle : 0.9721
✅ Nombre de features : 15
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 3. Interface Web

Ouvrez votre navigateur sur **http://localhost:8000**

L'interface propose **4 onglets** :

####  Onglet 1 : Prédiction individuelle
- Formulaire interactif avec les 11 variables du modèle
- Bouton **Évaluer le risque** → résultat instantané :
  - 🟢 **Bon client** (vert) ou 🔴 **Mauvais client** (rouge)
  - Probabilité de défaut + niveau de risque
  - Recommandation pour l'agent de crédit

####  Onglet 2 : Import CSV (Batch)
- **Télécharger le template CSV** pré-rempli
- **Uploader** un fichier CSV (jusqu'à 1000 clients)
- Résultats :
  - Statistiques globales (bons/mauvais clients, taux de défaut)
  - Tableau détaillé client par client
  - **Téléchargement du CSV enrichi** avec les prédictions

####  Onglet 3 : Informations du modèle
- Type de modèle, version, métriques de performance
- Liste complète des features utilisées
- Distinction features numériques / one-hot

####  Onglet 4 : Aide
- Documentation intégrée des colonnes CSV
- Interprétation des résultats
- Liste des endpoints API

---

##  Endpoints de l'API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Interface web (HTML) |
| `GET` | `/api` | Informations générales (JSON) |
| `GET` | `/health` | Vérification de l'état de l'API |
| `GET` | `/model/info` | Informations détaillées du modèle |
| `POST` | `/predict` | Prédiction unitaire (JSON) |
| `POST` | `/predict/batch` | Prédiction par lot (JSON, max 100) |
| `GET` | `/template/csv` | Télécharger un template CSV |
| `POST` | `/predict/csv` | Prédiction depuis un fichier CSV (max 1000) |
| `GET` | `/docs` | Documentation Swagger interactive |

### Exemple : prédiction unitaire

**Requête :**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_pret": 15000000,
    "duree_pret_mois": 60,
    "taux_interet": 8.5,
    "nb_emprunts_anterieurs": 2,
    "score_historique": 650,
    "situation_familiale": "marié",
    "secteur_activite": "commerce",
    "garantie": "oui"
  }'
```

**Réponse :**
```json
{
  "defaut": 0,
  "probabilite_defaut": 0.0834,
  "niveau_risque": "Faible",
  "recommandation": "Accorder le prêt selon les conditions standard.",
  "ratio_endettement": 1.0
}
```

---

## 📄 Format du fichier CSV

Le fichier CSV doit contenir **une ligne d'en-tête** avec les colonnes suivantes :

| Colonne | Type | Plage | Description |
|---------|------|-------|-------------|
| `nom_client` | Texte | libre | Identifiant du client (optionnel) |
| `age` | Entier | 22 – 65 | Âge de l'emprunteur |
| `revenu_mensuel` | Entier | 50 000 – 500 000 | Revenu mensuel en FCFA |
| `montant_pret` | Entier | 500 000 – 50 000 000 | Montant du prêt en FCFA |
| `duree_pret_mois` | Entier | 6 – 120 | Durée en mois |
| `taux_interet` | Décimal | 3 – 15 | Taux annuel en % |
| `nb_emprunts_anterieurs` | Entier | 0 – 5 | Nombre d'emprunts passés |
| `score_historique` | Entier | 300 – 850 | Score de crédit |
| `situation_familiale` | Texte | célibataire / marié / divorcé | Situation familiale |
| `secteur_activite` | Texte | agriculture / commerce / services / industrie / btp | Secteur |
| `garantie` | Texte | oui / non | Présence d'une garantie |

### Exemple de CSV

```csv
nom_client,age,revenu_mensuel,montant_pret,duree_pret_mois,taux_interet,nb_emprunts_anterieurs,score_historique,situation_familiale,secteur_activite,garantie
CLIENT_001,35,250000,15000000,60,8.5,2,650,marié,commerce,oui
CLIENT_002,45,180000,25000000,48,11.0,4,520,célibataire,agriculture,non
CLIENT_003,28,400000,8000000,36,6.0,1,780,marié,services,oui
```

>  **Astuce** : téléchargez le template via `GET /template/csv` ou depuis l'onglet **Import CSV** de l'interface web.

---

##  Modèle ML

### Variables utilisées (15 features)

**8 variables numériques** (standardisées) :
- `age`, `revenu_mensuel`, `montant_pret`, `duree_pret_mois`, `taux_interet`, `nb_emprunts_anterieurs`, `ratio_endettement`, `score_historique`

**7 variables one-hot** (dummies) :
- `situation_familiale_divorcé`, `situation_familiale_marié`
- `secteur_activite_btp`, `secteur_activite_commerce`, `secteur_activite_industrie`, `secteur_activite_services`
- `garantie_oui`

### Performances comparées

| Modèle | Accuracy | Précision | Rappel | F1-Score | AUC |
|--------|----------|-----------|--------|----------|-----|
| **Régression Logistique**  | 0.915 | 0.884 | 0.760 | 0.817 | **0.972** |
| Random Forest | 0.915 | 0.867 | **0.780** | 0.821 | 0.959 |

**Modèle retenu** : **Régression Logistique** (meilleure AUC + meilleure interprétabilité).

### Facteurs de risque principaux

1. **Ratio d'endettement élevé** — prédicteur le plus fort
2. **Score historique bas** — mauvais historique de crédit
3. **Absence de garantie** — facteur protecteur
4. **Montant du prêt élevé + faible revenu**
5. **Secteur d'activité volatil** (agriculture, BTP)

---

## 🐳 Déploiement

### Option 1 : Render.com (recommandé, gratuit)

1. Poussez votre code sur GitHub
2. Allez sur [render.com](https://render.com) → **New → Web Service**
3. Connectez votre repo GitHub
4. Configuration :
   - **Environment** : `Python 3`
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Cliquez sur **Create Web Service**

L'app sera accessible à une URL du type `https:........`

### Option 2 : Docker

**Construire l'image :**
```bash
docker build -t scoring-credit .
```

**Lancer le conteneur :**
```bash
docker run -p 8000:8000 scoring-credit
```

**Avec docker-compose :**
```bash
docker-compose up -d
```


##  Tests

### Test de santé de l'API

```bash
curl http://localhost:8000/health
# → {"status":"healthy","model_loaded":true,"scaler_loaded":true,"version":"1.0.0"}
```

### Test de prédiction rapide

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":35,"revenu_mensuel":250000,"montant_pret":15000000,"duree_pret_mois":60,"taux_interet":8.5,"nb_emprunts_anterieurs":2,"score_historique":650,"situation_familiale":"marié","secteur_activite":"commerce","garantie":"oui"}'
```

### Documentation interactive

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc

---

## ⚠️ Limites et perspectives

### Limites actuelles

1. **Données synthétiques** - ne reflètent pas parfaitement la réalité des portefeuilles de la BOAD
2. **Variables simplifiées** - absence de facteurs macroéconomiques, garanties détaillées, etc.
3. **Pas de validation croisée** - k-fold serait nécessaire pour une évaluation plus robuste
4. **Pas d'authentification API** - à ajouter pour un déploiement public

### Perspectives d'amélioration

- ✅ Intégrer des **données réelles** 
- ✅ Ajouter des **variables macroéconomiques** spécifiques à l'UEMOA
- ✅ Tester des modèles avancés (**XGBoost**, **LightGBM**, réseaux de neurones)
- ✅ Implémenter la **validation croisée k-fold**
- ✅ Ajouter l'**authentification JWT** pour sécuriser l'API
- ✅ Mettre en place un **monitoring** des prédictions en production
- ✅ Créer un **dashboard analytique** (Chart.js / Plotly) pour les équipes de risque
- ✅ Intégrer un **pipeline MLOps** (MLflow, DVC) pour le versioning des modèles

---

##  Technologies utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **Langage** | Python 3.11+ |
| **Data Science** | pandas, numpy, scikit-learn |
| **Visualisation** | matplotlib, seaborn |
| **API REST** | FastAPI, Uvicorn, Pydantic |
| **Templates** | Jinja2 |
| **Frontend** | HTML5, CSS3, JavaScript (vanilla) |
| **UI** | Font Awesome 6 |
| **Déploiement** | Docker, Render, Hugging Face |
| **Sérialisation** | joblib, JSON |

---

## 👤 Auteur

**AKOTA Yao Jérôme**

- 📧 Email : akotayaojerome@gmail.com
- 💼 LinkedIn : www.linkedin.com/in/yao-jérôme-akota 
- 🐙 GitHub :https://github.com/jerome-byte
- 📍 Localisation : Togo / Lome

---


## 🙏 Remerciements

- **BOAD** pour le contexte métier inspirant ce projet
- **UEMOA** pour les données macroéconomiques de référence
- La communauté **FastAPI** et **scikit-learn** pour leurs excellents outils

---

<p align="center">
  <strong> Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐</strong>
</p>

