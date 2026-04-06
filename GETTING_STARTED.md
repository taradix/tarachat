# Getting Started with TaraChat

## What is TaraChat?

TaraChat is a full-stack chatbot application that implements **Retrieval-Augmented Generation (RAG)** using **CroissantLLM**, a French-optimized language model. The application allows users to chat with an AI assistant that can retrieve and use information from uploaded documents.

## Architecture Overview

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   React     │ ◄─────► │   FastAPI   │ ◄─────► │ CroissantLLM│
│  Frontend   │  REST   │   Backend   │         │     +       │
│ TypeScript  │   API   │   Python    │         │    FAISS    │
└─────────────┘         └─────────────┘         └─────────────┘
  localhost:5173        localhost:8000         Vector Store
```

## Prerequisites

- **Docker** and **Docker Compose** (recommended)
- **At least 8GB RAM** for running the LLM
- **~3GB disk space** for model downloads

OR for local development:

- **Python 3.10+** with Poetry
- **Node.js 18+** with npm

## Quick Start (5 minutes)

### 1. Build and Start

```bash
# Option A: Using Make
make build
make up

# Option B: Using Docker Compose directly
docker-compose up --build -d
```

### 2. Monitor Initialization

The first startup takes **5-15 minutes** to download the CroissantLLM model:

```bash
make logs
# or
docker-compose logs -f backend
```

Wait for: `"RAG system initialized successfully"`

### 3. Access the Application

Open your browser to:
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

### 4. Start Chatting!

Wait for the status bar to show **"Ready"** (green), then try:
- "Parle-moi de Paris" (Tell me about Paris)
- "Qu'est-ce que la Tour Eiffel?" (What is the Eiffel Tower?)

## Key Features

### 1. RAG-Powered Conversations
The chatbot uses retrieved documents to provide accurate, contextual responses.

### 2. Document Upload
Click "Upload Document" to add new knowledge to the chatbot's database.

### 3. Source Citations
Responses include expandable "Sources" showing which documents were used.

### 4. Real-time Status
The status bar shows when the model is ready and vector store is loaded.

### 5. Conversation History
The chatbot maintains context throughout the conversation.

## Project Structure

```
tarachat/
├── backend/              # FastAPI + Python
│   ├── app/
│   │   ├── main.py      # API routes
│   │   ├── rag.py       # RAG implementation
│   │   ├── models.py    # Data models
│   │   └── config.py    # Configuration
│   ├── scripts/
│   │   └── init_data.py # Initialize sample data
│   └── data/
│       └── sample_documents.txt
│
├── frontend/            # React + TypeScript
│   └── src/
│       ├── components/  # React components
│       ├── api.ts       # API client
│       └── types.ts     # TypeScript types
│
└── docker-compose.yml   # Orchestration
```

## Common Commands

```bash
# Start application
make up

# View logs
make logs

# Stop application
make down

# Clean everything (including data)
make clean

# Initialize sample data
make init-data
```

## API Endpoints

### POST /chat
Send a message and get a response with sources.

**Request:**
```json
{
  "message": "Tell me about Paris",
  "conversation_history": []
}
```

**Response:**
```json
{
  "response": "Paris is the capital of France...",
  "sources": ["Paris is the capital...", "The city is located..."]
}
```

### POST /documents
Upload a document to the knowledge base.

**Request:**
```json
{
  "content": "Document text here",
  "metadata": {"title": "Example"}
}
```

### GET /health
Check system status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vector_store_ready": true
}
```

## Configuration

### Backend Settings (backend/.env)
```env
MODEL_NAME=croissantllm/CroissantLLMChat-v0.1
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=3
```

### Frontend Settings (frontend/.env)
```env
VITE_API_URL=http://localhost:8000
```

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **CroissantLLM** - French-optimized language model
- **LangChain** - RAG orchestration framework
- **FAISS** - Efficient vector similarity search
- **Sentence Transformers** - Text embeddings
- **Poetry** - Dependency management

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **Axios** - HTTP client

### Infrastructure
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

## How RAG Works

1. **Document Ingestion**
   - Documents are split into chunks
   - Each chunk is converted to embeddings
   - Embeddings are stored in FAISS vector database

2. **Query Processing**
   - User query is converted to embedding
   - Similar document chunks are retrieved (top K)
   - Retrieved chunks provide context

3. **Response Generation**
   - Context + query are sent to CroissantLLM
   - Model generates response using retrieved information
   - Sources are returned with the response

## Troubleshooting

### Model Loading Takes Forever
- First download takes 10-15 minutes
- Subsequent starts are much faster (~1-2 minutes)

### Out of Memory
- Ensure you have at least 8GB RAM
- Close other applications
- Consider using CPU instead of GPU

### Connection Refused
- Wait for both containers to fully start
- Check logs: `docker-compose logs`
- Verify ports 5173 and 8000 are not in use

### Frontend Can't Reach Backend
- Check both containers are running: `docker-compose ps`
- Verify VITE_API_URL in frontend/.env
- Check CORS settings in backend/app/main.py

## Next Steps

1. **Try the sample queries** included with the application
2. **Upload your own documents** to customize the knowledge base
3. **Explore the API** at http://localhost:8000/docs
4. **Modify the configuration** to tune RAG parameters
5. **Check the code** - it's well-documented and modular!

## Resources

- **CroissantLLM**: https://huggingface.co/croissantllm
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **LangChain**: https://python.langchain.com/
- **React**: https://react.dev/

## Support

For issues or questions:
1. Check the logs: `make logs`
2. Review the documentation in this directory
3. Check the API docs: http://localhost:8000/docs

## License

This project is provided as-is for educational and demonstration purposes.
