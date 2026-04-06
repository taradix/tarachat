# TaraChat Backend

FastAPI-based backend implementing RAG (Retrieval-Augmented Generation) with CroissantLLM.

## Features

- FastAPI REST API
- CroissantLLM for French language support
- FAISS vector store for efficient similarity search
- LangChain for RAG orchestration
- Multilingual embeddings

## Setup

### With Docker (Recommended)

```bash
cd ..
docker-compose up backend
```

### Local Development

```bash
# Install dependencies
poetry install

# Run the server
poetry run uvicorn app.main:app --reload
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /chat` - Chat with the bot
- `POST /documents` - Add document to knowledge base
- `GET /documents/count` - Get document count

## Configuration

See `.env.example` for configuration options.

## Initialize Sample Data

```bash
python scripts/init_data.py
```
