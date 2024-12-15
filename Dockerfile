# Utilisation de l'image Python comme base
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY model1.keras /app/model1.keras

# Installer les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'app Flask va tourner
EXPOSE 5000

# Commande pour lancer l'application Flask
CMD ["python", "app.py"]
