# TaraChat Usage Guide

## Quick Start

### 1. Start the Application

```bash
# Using Make
make up

# Or using docker-compose directly
docker-compose up --build
```

The first startup will take several minutes as it downloads and initializes the CroissantLLM model (~2-3 GB).

### 2. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 3. Monitor Startup

Check the logs to monitor the initialization:

```bash
make logs
# or
docker-compose logs -f backend
```

Wait until you see "RAG system initialized successfully" in the backend logs.

## Using the Chatbot

### Basic Chat

1. Open http://localhost:5173 in your browser
2. Wait for the status bar to show "Ready" (green indicator)
3. Type your message in the input field at the bottom
4. Press Enter or click "Send"

### Adding Documents

You can add documents to enhance the chatbot's knowledge:

1. Click "Upload Document" button in the header
2. Paste your document content
3. Optionally add metadata as JSON
4. Click "Upload"

### Example Queries (French)

Since CroissantLLM is optimized for French:

- "Parle-moi de Paris"
- "Qu'est-ce que la Tour Eiffel?"
- "Quelle est la capitale de la France?"
- "Décris la cuisine française"

### Viewing Sources

When the chatbot responds using RAG, you'll see a "Sources" dropdown showing the document chunks used to generate the response.

## API Usage

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Parle-moi de Paris",
    "conversation_history": []
  }'
```

### Add Document

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your document content here",
    "metadata": {"title": "Example"}
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Management Commands

### View Logs

```bash
make logs
```

### Stop the Application

```bash
make down
```

### Clean Everything (including data)

```bash
make clean
```

### Initialize Sample Data

```bash
make init-data
```

## Configuration

### Backend Configuration

Edit `backend/.env`:

```env
# Model settings
MODEL_NAME=croissantllm/CroissantLLMChat-v0.1
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# RAG settings
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=3
```

### Frontend Configuration

Edit `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

## Troubleshooting

### Model Takes Too Long to Load

The first time you run the application, it needs to download the CroissantLLM model. This can take 10-15 minutes depending on your internet connection.

### Out of Memory

CroissantLLM requires at least 8GB of RAM. If you're running out of memory:

1. Close other applications
2. Consider using a smaller model
3. Disable GPU if available but causing issues

### Port Already in Use

If ports 5173 or 8000 are already in use, edit `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8001:8000"  # Changed from 8000:8000
  frontend:
    ports:
      - "5174:5173"  # Changed from 5173:5173
```

### Frontend Can't Connect to Backend

Make sure both containers are running:

```bash
docker-compose ps
```

Check backend logs:

```bash
docker-compose logs backend
```

## Development

### Backend Development

```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
# Backend
cd backend
poetry run pytest

# Frontend
cd frontend
npm test
```

## Production Deployment

For production deployment:

1. Set proper CORS origins in `backend/app/main.py`
2. Use production-grade WSGI server (Gunicorn)
3. Set up reverse proxy (Nginx)
4. Enable HTTPS
5. Use environment variables for sensitive configuration
6. Set up persistent volumes for vector store data

## Resources

- [CroissantLLM](https://huggingface.co/croissantllm)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [LangChain Documentation](https://python.langchain.com/)
