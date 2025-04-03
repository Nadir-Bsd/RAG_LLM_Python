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
1. Placez vos documents dans le dossier `data/`. Le programme n'accepte que les fichiers .PDF.

2. Exécutez le script de création de la base de données :

- attendre que le script fasse la DB (ça peux être long)

## commands

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

for the RAG:
   ```bash
   -use rag: "the question"
   ```
(utilisez-le lorsque vous parlez déjà avec le chatbot)

to stop:
   ```bash
   "exit"
   ```

## Structure du projet

- `data/` : Contient les données source (bien pense a gitIgnore si donnée sensible).
- `chroma/` : Base de données vectorielle.
- `indexationPhase.py` : Script de création de la base de données.
- `chatbot.py` : Script d'interrogation du chatbot.

## Attention !!!

une erreurs possible est que la fenetre contextuelle soit dépassée (trop de tokens):
```bash
   raise httpx.HTTPStatusError(
      httpx.HTTPStatusError: Error response 429 while fetching https://api.mistral.ai/v1/chat/completions: {"message":"Requests rate limit exceeded"}
   )
```
