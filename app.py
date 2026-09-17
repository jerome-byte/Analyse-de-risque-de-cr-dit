"""

API FastAPI - Modèle de Scoring Crédit (Système Bancaire Ouest-Africain)

Auteur : AKOTA Yao Jérôme
Projet : Prédiction du risque de défaut de crédit (contexte UEMOA)

Cette API expose le modèle de Régression Logistique entraîné dans le
notebook scoring_credit.ipynb pour prédire le risque de défaut d'un
emprunteur au moment de la demande de prêt.

Endpoints :
    GET  /                  → Interface web (page HTML avec onglets)
    GET  /api               → Informations générales de l'API (JSON)
    GET  /health            → Vérification de l'état de l'API
    GET  /model/info        → Informations sur le modèle
    POST /predict           → Prédiction pour un emprunteur
    POST /predict/batch     → Prédiction pour plusieurs emprunteurs
    GET  /template/csv      → Télécharger un template CSV
    POST /predict/csv       → Prédiction depuis un fichier CSV
    GET  /docs              → Documentation interactive Swagger

"""


# 1. IMPORTS

import io
import os
import json
from typing import List, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    status,
    UploadFile,
    File,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator



# 2. CONFIGURATION

# Chemin vers les artefacts du modèle (modifiable via variable d'environnement)
MODEL_DIR = os.environ.get("MODEL_DIR", "model_dossier")

MODEL_PATH = os.path.join(MODEL_DIR, "model_lr.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")



# 3. CHARGEMENT DU MODÈLE AU DÉMARRAGE

print("=" * 70)
print(" Démarrage de l'API Scoring Crédit ")
print("=" * 70)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Modèle chargé : {metadata['metrics']['model']}")
    print(f"✅ Scaler chargé")
    print(f"✅ AUC du modèle : {metadata['metrics']['auc']:.4f}")
    print(f"✅ Nombre de features : {len(metadata['feature_names'])}")
except FileNotFoundError as e:
    print(f"❌ ERREUR : fichier introuvable → {e}")
    print("   Exécutez d'abord la cellule de sauvegarde dans le notebook.")
    raise

# Récupération des métadonnées utiles
FEATURE_NAMES = metadata["feature_names"]
NUMERICAL_COLS = metadata["numerical_cols"]
DUMMY_COLS = metadata["dummy_cols"]
CATEGORICAL_INFO = metadata["categorical_info"]



# 4. INITIALISATION DE L'APPLICATION FASTAPI

app = FastAPI(
    title="API Scoring Crédit",
    description=(
        "API de prédiction du risque de défaut de crédit pour le "
        "système bancaire ouest-africain. "
        "Modèle : Régression Logistique (AUC = 0.972)."
    ),
    version=metadata["version"],
    contact={
        "name": "AKOTA Yao Jérôme",
        "email": "akotayaojerome@gmail.com",
    },
)

# Créer les dossiers si nécessaire
os.makedirs("templates", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurer Jinja2 pour les templates HTML
templates = Jinja2Templates(directory="templates")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 5. SCHÉMAS PYDANTIC (VALIDATION DES ENTRÉES)

class CreditApplication(BaseModel):
    """Schéma d'entrée : une demande de prêt."""

    # Variables numériques
    age: int = Field(..., ge=22, le=65, description="Âge (22-65 ans)", examples=[35])
    revenu_mensuel: int = Field(..., ge=50000, le=500000,
                                 description="Revenu mensuel en FCFA", examples=[250000])
    montant_pret: int = Field(..., ge=500000, le=50000000,
                               description="Montant du prêt en FCFA", examples=[15000000])
    duree_pret_mois: int = Field(..., ge=6, le=120,
                                  description="Durée en mois (6-120)", examples=[60])
    taux_interet: float = Field(..., ge=3.0, le=15.0,
                                 description="Taux annuel en % (3-15)", examples=[8.5])
    nb_emprunts_anterieurs: int = Field(..., ge=0, le=5,
                                         description="Nombre d'emprunts antérieurs", examples=[2])
    score_historique: int = Field(..., ge=300, le=850,
                                   description="Score de crédit (300-850)", examples=[650])

    # Variables catégorielles
    situation_familiale: Literal["célibataire", "marié", "divorcé"] = Field(
        ..., description="Situation familiale"
    )
    secteur_activite: Literal[
        "agriculture", "commerce", "services", "industrie", "btp"
    ] = Field(..., description="Secteur d'activité")
    garantie: Literal["oui", "non"] = Field(
        ..., description="Présence d'une garantie"
    )

    @field_validator("situation_familiale", "secteur_activite", "garantie")
    @classmethod
    def lowercase_categorical(cls, v: str) -> str:
        """Normalise les valeurs catégorielles en minuscules."""
        return v.lower() if isinstance(v, str) else v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "revenu_mensuel": 250000,
                "montant_pret": 15000000,
                "duree_pret_mois": 60,
                "taux_interet": 8.5,
                "nb_emprunts_anterieurs": 2,
                "score_historique": 650,
                "situation_familiale": "marié",
                "secteur_activite": "commerce",
                "garantie": "oui",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schéma de sortie : prédiction et interprétation."""
    defaut: int = Field(..., description="1 = défaut, 0 = bon remboursement")
    probabilite_defaut: float = Field(..., description="Probabilité (0 à 1)")
    niveau_risque: str = Field(..., description="'Faible', 'Moyen' ou 'Élevé'")
    recommandation: str = Field(..., description="Recommandation pour l'agent")
    ratio_endettement: float = Field(..., description="Ratio d'endettement calculé")


class BatchPredictionRequest(BaseModel):
    """Schéma d'entrée pour la prédiction par lot."""
    demandes: List[CreditApplication] = Field(
        ..., min_length=1, max_length=100,
        description="Liste de 1 à 100 demandes"
    )


class BatchPredictionResponse(BaseModel):
    """Schéma de sortie pour la prédiction par lot."""
    predictions: List[PredictionResponse]
    total: int
    nb_defauts: int
    taux_defaut_predit: float


# 6. FONCTIONS UTILITAIRES DU PIPELINE

def compute_ratio_endettement(
    revenu_mensuel: int, montant_pret: int, duree_pret_mois: int
) -> float:
    """
    Calcule le ratio d'endettement (identique au notebook) :
        ratio = montant_pret / (revenu_mensuel * duree_pret_mois)
    Arrondi à 4 décimales.
    """
    ratio = montant_pret / (revenu_mensuel * duree_pret_mois)
    return round(ratio, 4)


def preprocess_application(demande: CreditApplication):
    """
    Transforme une demande brute en DataFrame prêt pour le modèle.

    Étapes (identiques au notebook) :
      1. Calcul du ratio_endettement
      2. Encodage one-hot (drop_first=True)
      3. Ré-indexation sur FEATURE_NAMES
      4. Standardisation des colonnes numériques uniquement

    Retourne : (DataFrame prêt, ratio_endettement)
    """
    # 1) Ratio d'endettement
    ratio = compute_ratio_endettement(
        demande.revenu_mensuel, demande.montant_pret, demande.duree_pret_mois
    )

    # 2) Dictionnaire brut
    raw = {
        "age": demande.age,
        "revenu_mensuel": demande.revenu_mensuel,
        "montant_pret": demande.montant_pret,
        "duree_pret_mois": demande.duree_pret_mois,
        "taux_interet": demande.taux_interet,
        "nb_emprunts_anterieurs": demande.nb_emprunts_anterieurs,
        "ratio_endettement": ratio,
        "score_historique": demande.score_historique,
        "situation_familiale": demande.situation_familiale,
        "secteur_activite": demande.secteur_activite,
        "garantie": demande.garantie,
    }
    df = pd.DataFrame([raw])

    # 3) Encodage one-hot
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 4) Ré-indexation sur l'ordre exact des features du modèle
    df_encoded = df_encoded.reindex(columns=FEATURE_NAMES, fill_value=0)

    # Conversion explicite des dummies en int (0/1)
    for col in DUMMY_COLS:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(int)

    # 5) Standardisation des colonnes numériques uniquement
    df_encoded[NUMERICAL_COLS] = scaler.transform(df_encoded[NUMERICAL_COLS])

    return df_encoded, ratio


def interpret_prediction(prob: float):
    """
    Traduit la probabilité de défaut en niveau de risque + recommandation.

    Seuils (contexte bancaire prudent) :
        - prob < 0.30          → Faible
        - 0.30 ≤ prob < 0.60   → Moyen
        - prob ≥ 0.60          → Élevé
    """
    if prob < 0.30:
        return ("Faible", "Accorder le prêt selon les conditions standard.")
    elif prob < 0.60:
        return ("Moyen", "Exiger des garanties supplémentaires et/ou ajuster le taux.")
    else:
        return ("Élevé",
                "Risque trop élevé : refuser le prêt ou exiger une garantie "
                "très solide avec un suivi renforcé.")



# 7. ENDPOINTS — INTERFACE WEB ET INFORMATIONS

@app.get("/", response_class=HTMLResponse, tags=["Interface Web"])
async def index(request: Request):
    """Page d'accueil : interface web avec onglets."""
    # Compatibilité Starlette < 0.45 et >= 0.45
    try:
        return templates.TemplateResponse(request, "index.html")
    except TypeError:
        return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api", tags=["Information"])
def api_info():
    """Informations générales de l'API (JSON)."""
    return {
        "api": "Scoring Crédit",
        "version": metadata["version"],
        "description": "Prédiction du risque de défaut de crédit",
        "auteur": "AKOTA Yao Jérôme",
        "documentation": "/docs",
        "endpoints": {
            "interface_web": "GET /",
            "health": "GET /health",
            "model_info": "GET /model/info",
            "predict": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "template_csv": "GET /template/csv",
            "predict_csv": "POST /predict/csv",
        },
    }


@app.get("/health", tags=["Information"])
def health_check():
    """Vérification de l'état de l'API (pour load balancers)."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "version": metadata["version"],
    }


@app.get("/model/info", tags=["Information"])
def model_info():
    """Retourne les informations détaillées sur le modèle."""
    return {
        "model_type": metadata["metrics"]["model"],
        "version": metadata["version"],
        "performance": metadata["metrics"],
        "features": FEATURE_NAMES,
        "numerical_features": NUMERICAL_COLS,
        "dummy_features": DUMMY_COLS,
        "categorical_features": CATEGORICAL_INFO,
    }


# 8. ENDPOINTS — PRÉDICTION UNITAIRES ET BATCH (JSON)

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prédiction"],
    summary="Prédire le risque de défaut pour un emprunteur",
)
def predict(demande: CreditApplication):
    """Prédit si un emprunteur va faire défaut."""
    try:
        X, ratio = preprocess_application(demande)
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1])
        niveau, reco = interpret_prediction(prob)

        return PredictionResponse(
            defaut=pred,
            probabilite_defaut=round(prob, 4),
            niveau_risque=niveau,
            recommandation=reco,
            ratio_endettement=ratio,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction : {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prédiction"],
    summary="Prédire le risque pour plusieurs emprunteurs (JSON)",
)
def predict_batch(request: BatchPredictionRequest):
    """Prédit le risque pour une liste de demandes (max 100)."""
    try:
        predictions = []
        for demande in request.demandes:
            X, ratio = preprocess_application(demande)
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0, 1])
            niveau, reco = interpret_prediction(prob)

            predictions.append(
                PredictionResponse(
                    defaut=pred,
                    probabilite_defaut=round(prob, 4),
                    niveau_risque=niveau,
                    recommandation=reco,
                    ratio_endettement=ratio,
                )
            )

        nb_defauts = sum(p.defaut for p in predictions)
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            nb_defauts=nb_defauts,
            taux_defaut_predit=round(nb_defauts / len(predictions), 4),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction par lot : {str(e)}",
        )


# 9. ENDPOINTS — UPLOAD CSV

@app.get(
    "/template/csv",
    tags=["CSV"],
    summary="Télécharger un modèle de fichier CSV",
)
def download_template():
    """Retourne un fichier CSV modèle avec 3 exemples."""
    template_data = pd.DataFrame([
        {
            "nom_client": "CLIENT_001", "age": 35, "revenu_mensuel": 250000,
            "montant_pret": 15000000, "duree_pret_mois": 60, "taux_interet": 8.5,
            "nb_emprunts_anterieurs": 2, "score_historique": 650,
            "situation_familiale": "marié", "secteur_activite": "commerce",
            "garantie": "oui",
        },
        {
            "nom_client": "CLIENT_002", "age": 45, "revenu_mensuel": 180000,
            "montant_pret": 25000000, "duree_pret_mois": 48, "taux_interet": 11.0,
            "nb_emprunts_anterieurs": 4, "score_historique": 520,
            "situation_familiale": "célibataire", "secteur_activite": "agriculture",
            "garantie": "non",
        },
        {
            "nom_client": "CLIENT_003", "age": 28, "revenu_mensuel": 400000,
            "montant_pret": 8000000, "duree_pret_mois": 36, "taux_interet": 6.0,
            "nb_emprunts_anterieurs": 1, "score_historique": 780,
            "situation_familiale": "marié", "secteur_activite": "services",
            "garantie": "oui",
        },
    ])

    csv_buffer = io.StringIO()
    template_data.to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=template_scoring_credit.csv"},
    )


@app.post(
    "/predict/csv",
    tags=["CSV"],
    summary="Prédire le risque pour tous les clients d'un fichier CSV",
)
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload d'un CSV pour prédire en batch.
    Colonnes attendues : voir /template/csv
    """
    # Validation extension
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier doit être au format CSV (.csv)",
        )

    # Lecture du fichier
    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(content), encoding="latin-1", sep=";")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Impossible de lire le fichier CSV : {str(e)}",
        )

    # Vérification des colonnes requises
    required_cols = [
        "age", "revenu_mensuel", "montant_pret", "duree_pret_mois",
        "taux_interet", "nb_emprunts_anterieurs", "score_historique",
        "situation_familiale", "secteur_activite", "garantie",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Colonnes manquantes : {missing_cols}. "
                   f"Téléchargez le template via /template/csv.",
        )

    if len(df) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier ne doit pas contenir plus de 1000 lignes.",
        )

    # Prédictions ligne par ligne
    resultats = []
    errors = []

    for idx, row in df.iterrows():
        try:
            demande = CreditApplication(
                age=int(row["age"]),
                revenu_mensuel=int(row["revenu_mensuel"]),
                montant_pret=int(row["montant_pret"]),
                duree_pret_mois=int(row["duree_pret_mois"]),
                taux_interet=float(row["taux_interet"]),
                nb_emprunts_anterieurs=int(row["nb_emprunts_anterieurs"]),
                score_historique=int(row["score_historique"]),
                situation_familiale=str(row["situation_familiale"]).lower(),
                secteur_activite=str(row["secteur_activite"]).lower(),
                garantie=str(row["garantie"]).lower(),
            )
            X, ratio = preprocess_application(demande)
            pred = int(model.predict(X)[0])
            prob = float(model.predict_proba(X)[0, 1])
            niveau, reco = interpret_prediction(prob)

            nom = str(row.get("nom_client", f"Ligne_{idx + 1}"))

            resultats.append({
                "nom_client": nom,
                "age": int(row["age"]),
                "revenu_mensuel": int(row["revenu_mensuel"]),
                "montant_pret": int(row["montant_pret"]),
                "score_historique": int(row["score_historique"]),
                "ratio_endettement": ratio,
                "defaut": pred,
                "probabilite_defaut": round(prob, 4),
                "niveau_risque": niveau,
                "statut": "Mauvais client" if pred == 1 else "Bon client",
                "recommandation": reco,
            })
        except Exception as e:
            errors.append({"ligne": int(idx) + 2, "erreur": str(e)})

    # Statistiques globales (HORS de la boucle for !)
    df_result = pd.DataFrame(resultats)
    nb_total = len(resultats)
    nb_mauvais = int(df_result["defaut"].sum()) if nb_total > 0 else 0
    nb_bons = nb_total - nb_mauvais

    summary = {
        "total_clients": nb_total,
        "bons_clients": nb_bons,
        "mauvais_clients": nb_mauvais,
        "taux_defaut": round(nb_mauvais / nb_total, 4) if nb_total > 0 else 0,
        "probabilite_moyenne": (
            round(float(df_result["probabilite_defaut"].mean()), 4)
            if nb_total > 0 else 0
        ),
        "risque_faible": int((df_result["niveau_risque"] == "Faible").sum()) if nb_total > 0 else 0,
        "risque_moyen": int((df_result["niveau_risque"] == "Moyen").sum()) if nb_total > 0 else 0,
        "risque_eleve": int((df_result["niveau_risque"] == "Élevé").sum()) if nb_total > 0 else 0,
    }

    # Générer un CSV enrichi
    csv_out = io.StringIO()
    df_result.to_csv(csv_out, index=False, encoding="utf-8")
    csv_content = csv_out.getvalue()

    return JSONResponse(content={
        "success": True,
        "predictions": resultats,
        "summary": summary,
        "errors": errors,
        "csv_result": csv_content,
    })
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Retourne une réponse vide pour éviter les erreurs 404 dans les logs."""
    from fastapi.responses import Response
    return Response(status_code=204)


# 10. POINT D'ENTRÉE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)