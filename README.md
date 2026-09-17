# Modèle de Scoring Crédit  Prédiction du Risque de Défaut

**Auteur :** AKOTA Yao Jérôme

Projet de Data Science / Machine Learning appliqué au contexte bancaire ouest-africain (BOAD / UEMOA).

## Contexte

La BOAD finance des projets de développement dans les 8 pays de l'UEMOA. Ce projet a pour but de construire un modèle capable de prédire, au moment de la demande de prêt, si un emprunteur présente un risque élevé de défaut (classification binaire : 0 = bon remboursement, 1 = défaut).

## Contenu du projet

Le notebook `scoring_credit_1_.ipynb` couvre tout le pipeline de Data Science :

1. **Création d'un jeu de données synthétique**  1 000 demandes de prêt simulées (revenus, montant du prêt, historique de crédit, secteur d'activité, etc.), car les données bancaires réelles sont confidentielles.
2. **Analyse exploratoire (EDA)**  distribution de la cible, analyses univariées et bivariées, matrice de corrélation.
3. **Prétraitement**  vérification des valeurs manquantes, encodage one-hot des variables catégorielles, division train/test (80/20) et standardisation (sans fuite de données).
4. **Modélisation** entraînement et évaluation de deux modèles :
   - Régression Logistique (interprétable)
   - Random Forest (basé sur les arbres)
5. **Évaluation**  matrices de confusion, rapports de classification, courbes ROC/AUC, comparaison des deux modèles.
6. **Interprétation et recommandations**  facteurs de risque clés et pistes pour la gestion du risque de crédit à la BOAD.

## Variables du jeu de données

| Variable | Type | Description |
|---|---|---|
| age | Numérique | Âge de l'emprunteur (22-65 ans) |
| revenu_mensuel | Numérique | Revenu mensuel en FCFA |
| montant_pret | Numérique | Montant du prêt demandé en FCFA |
| duree_pret_mois | Numérique | Durée de remboursement en mois |
| taux_interet | Numérique | Taux d'intérêt annuel en % |
| nb_emprunts_anterieurs | Numérique | Nombre d'emprunts antérieurs |
| ratio_endettement | Numérique | Ratio dette/revenu |
| score_historique | Numérique | Score d'historique de crédit (300-850) |
| situation_familiale | Catégorielle | Célibataire, Marié, Divorcé |
| secteur_activite | Catégorielle | Agriculture, Commerce, Services, Industrie, BTP |
| garantie | Catégorielle | Présence d'une garantie (Oui/Non) |
| defaut | Cible (0/1) | 1 = défaut, 0 = bon remboursement |

## Principaux résultats

- Les deux modèles obtiennent de très bonnes performances (Accuracy ≈ 0,915 ; AUC > 0,95).
- La Régression Logistique offre la meilleure AUC (0,972) et une meilleure interprétabilité.
- Le Random Forest a un léger avantage en rappel.
- Facteurs de risque principaux : **ratio d'endettement élevé**, **score historique bas**, **absence de garantie**.

## Prérequis

```
python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Installation rapide :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Utilisation

1. Ouvrir le notebook `scoring_credit_1_.ipynb` dans Jupyter.
2. Exécuter les cellules dans l'ordre.
3. Le jeu de données est généré automatiquement (pas de fichier externe requis), puis sauvegardé en CSV.

> Note : le chemin de sauvegarde du CSV dans le notebook est codé en dur pour Windows (`C:\Users\...`). Pensez à l'adapter à votre environnement avant d'exécuter cette cellule.

