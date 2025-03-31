# Chatbot RAG (Retrieval Augmented Generation)

Ce projet implémente un chatbot utilisant l'approche RAG (Retrieval Augmented Generation), qui enrichit les réponses du modèle avec des données contextuelles.

## Installation

```bash	
git clone https://github.com/Nadir-Bsd/RAG_LLM_Python.git
```

```bash
pip install -r requirements.txt
```

## Prérequis

- Python version 3.8 ou supérieure
- Dépendances listées dans `requirements.txt`
- Clé API Mistral AI disponible [ici](https://console.mistral.ai/api-keys) (compte requis)
- Clé API Langsmith disponible [ici](https://smith.langchain.com/) (compte requis)

## Configuration

1. Dupliquez le fichier `.env.example` et renommez-le `.env`.
2. Configurez vos clés API dans le fichier `.env`.

## Utilisation
1. Placez vos documents dans le dossier `data/`. Le programme n'accepte que les fichiers .JPG.

2. Exécutez le script de création de la base de données :


linux:
   ```bash
   python3 indexationPhase.py
   ```

windows:
   ```bash
   python indexationPhase.py
   ```

3. Parler avec le chatbot :


linux: 
   ```bash
   python3 chatbot.py
   ```

windows:
   ```bash
   python chatbot.py
   ```

## Structure du projet

- `data/` : Contient les données source.
- `chroma/` : Base de données vectorielle.
- `indexationPhase.py` : Script de création de la base de données.
- `chatbot.py` : Script d'interrogation du chatbot.