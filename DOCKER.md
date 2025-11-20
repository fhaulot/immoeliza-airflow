# 🐳 Docker Setup Guide

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Compose                         │
├──────────────┬──────────────┬──────────────┬────────────┤
│   Scraper    │    Model     │     API      │  Frontend  │
│   (Chrome)   │  Training    │  (FastAPI)   │ (Streamlit)│
│   Port: -    │  Port: -     │  Port: 8000  │ Port: 8501 │
└──────────────┴──────────────┴──────────────┴────────────┘
```

## 🚀 Quick Start

### Option 1: Build et Run tout en une commande

```bash
docker-compose up --build
```

### Option 2: Build puis Run séparément

```bash
# Build toutes les images
docker-compose build

# Run tous les services
docker-compose up
```

### Option 3: Run en arrière-plan (detached mode)

```bash
docker-compose up -d
```

## 📊 Services disponibles

Une fois lancé, vous aurez accès à :

- **API Backend** : http://localhost:8000
- **API Docs (Swagger)** : http://localhost:8000/docs
- **Frontend Streamlit** : http://localhost:8501

## 🔍 Commandes utiles

### Voir les logs

```bash
# Tous les services
docker-compose logs -f

# Service spécifique
docker-compose logs -f api
docker-compose logs -f frontend
docker-compose logs -f scraper
docker-compose logs -f model-training
```

### Status des conteneurs

```bash
docker-compose ps
```

### Arrêter les services

```bash
# Arrêt gracieux
docker-compose down

# Arrêt + suppression des volumes
docker-compose down -v
```

### Rebuild un service spécifique

```bash
docker-compose build api
docker-compose up -d api
```

### Entrer dans un conteneur

```bash
docker-compose exec api bash
docker-compose exec frontend bash
```

## 🛠️ Workflow de développement

### 1. Développement local avec hot-reload

Pour le développement, vous pouvez monter votre code en volume :

```yaml
# Dans docker-compose.yml (pour dev)
services:
  api:
    volumes:
      - ./deployment:/app/deployment  # Hot reload
```

### 2. Rebuild après changement de dépendances

Si vous modifiez `pyproject.toml` ou `uv.lock` :

```bash
docker-compose build --no-cache
docker-compose up
```

### 3. Tester un service individuellement

```bash
# Build et run seulement l'API
docker-compose up api

# Build et run seulement le frontend
docker-compose up frontend
```

## 📦 Structure des volumes

Les données persistantes sont stockées dans :

```
./data/                          # Données scrapées
./model/trained_models/          # Modèles ML entraînés
./model/processed_data/          # Données preprocessées
./analyse/                       # Analyses
```

## 🐛 Troubleshooting

### Le scraper ne fonctionne pas (Chrome)

```bash
# Vérifier les logs
docker-compose logs scraper

# Rebuild avec cache cleared
docker-compose build --no-cache scraper
```

### L'API ne démarre pas

```bash
# Vérifier que le modèle est bien entraîné
docker-compose logs model-training

# Vérifier les fichiers requis
docker-compose exec api ls -la model/trained_models/
```

### Port déjà utilisé

```bash
# Changer le port dans docker-compose.yml
ports:
  - "8001:8000"  # Au lieu de 8000:8000
```

## 🔧 Configuration avancée

### Variables d'environnement

Créez un fichier `.env` à la racine :

```env
# .env
PYTHONUNBUFFERED=1
API_PORT=8000
FRONTEND_PORT=8501
```

Puis dans `docker-compose.yml` :

```yaml
env_file:
  - .env
```

### Production deployment

Pour la production, ajoutez :

```yaml
services:
  api:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

## 📊 Ordre d'exécution

Docker Compose lance les services dans cet ordre :

1. **Scraper** : Collecte les données
2. **Model Training** : Entraîne le modèle (dépend du scraper)
3. **API** : Lance l'API (dépend du modèle)
4. **Frontend** : Lance le dashboard (dépend de l'API)

## 🎯 Tests

### Tester l'API

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "HOUSE",
    "bedroomCount": 3,
    "habitableSurface": 150,
    "postCode": 1000
  }'
```

### Tester le Frontend

Ouvrez http://localhost:8501 dans votre navigateur.

## 🚀 CI/CD

Exemple pour GitHub Actions :

```yaml
# .github/workflows/docker.yml
name: Docker Build

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker images
        run: docker-compose build
      - name: Run tests
        run: docker-compose up -d
```

## 📝 Notes importantes

- Le premier build peut prendre 5-10 minutes (installation de toutes les dépendances)
- Les builds suivants seront beaucoup plus rapides grâce au cache Docker
- UV rend l'installation des dépendances 10-100x plus rapide que pip
- Les modèles entraînés sont persistés dans des volumes

## 🎉 C'est tout !

Une seule commande pour tout lancer :

```bash
docker-compose up --build
```

Puis accédez à http://localhost:8501 pour voir le dashboard ! 🏡
