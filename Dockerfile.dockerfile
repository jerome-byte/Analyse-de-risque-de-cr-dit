
# Image Docker pour l'API Scoring Crédit BOAD

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="AKOTA Yao Jérôme"
LABEL description="API Scoring Crédit BOAD - Prédiction du risque de défaut"

# Répertoire de travail
WORKDIR /app

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application et les artefacts du modèle
COPY app.py .
COPY model_artifacts/ ./model_artifacts/

# Exposer le port
EXPOSE 8000

# Variables d'environnement
ENV MODEL_DIR=/app/model_artifacts
ENV PYTHONUNBUFFERED=1

# Commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]