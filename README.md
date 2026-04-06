# TaraChat - RAG Chatbot with CroissantLLM

A full-stack chatbot application implementing **Retrieval-Augmented Generation (RAG)** using **CroissantLLM**, a French-optimized language model.

## Features

- **Backend**: FastAPI-based REST API with RAG implementation
- **Frontend**: React + TypeScript chat interface with modern UI
- **LLM**: CroissantLLM for French/multilingual support
- **Vector Store**: FAISS for efficient similarity search
- **Document Upload**: Add text documents or PDF files to enhance knowledge base
- **PDF Support**: Automatic text extraction from PDF files with metadata
- **Source Citations**: View which documents informed each response
- **Real-time Status**: Monitor system health and model loading
- **Containerized**: Docker Compose for easy deployment

## Quick Start

```bash
# Start the application
docker-compose up --build

# Wait for "RAG system initialized successfully" in logs
docker-compose logs -f backend
```

**Access the application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

**First startup takes 5-15 minutes** to download the CroissantLLM model (~3GB).

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete introduction and setup guide
- **[USAGE.md](USAGE.md)** - Detailed usage instructions and examples
- **[DOCUMENT_INGESTION.md](DOCUMENT_INGESTION.md)** - Command-line document management
- **[PDF_UPLOAD.md](PDF_UPLOAD.md)** - PDF upload feature guide
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Technical architecture and code organization
- **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - Quick fix for chat performance issues
- **[PERFORMANCE_ISSUES.md](PERFORMANCE_ISSUES.md)** - Detailed performance analysis and solutions

## Prerequisites

- Docker and Docker Compose
- At least 8GB RAM (for running the LLM)
- ~3GB disk space for model downloads

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **CroissantLLM** - French-optimized language model
- **LangChain** - RAG orchestration
- **FAISS** - Vector similarity search
- **PyPDF2** - PDF text extraction
- **Poetry** - Dependency management

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool
- **Axios** - HTTP client

## Project Structure

```
tarachat/
├── backend/           # FastAPI backend
│   ├── app/          # Application code
│   ├── scripts/      # Utility scripts
│   └── data/         # Sample documents
│
├── frontend/         # React frontend
│   └── src/         # Source code
│       └── components/  # React components
│
└── docker-compose.yml  # Orchestration
```

## Key Commands

```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down

# Clean everything (including data)
docker-compose down -v
```

Or use the Makefile:
```bash
make up           # Start
make logs         # View logs
make down         # Stop
make clean        # Clean all
make ingest-docs  # Ingest documents from data/documents/
make list-docs    # List all documents in vector store
```

## Example Usage

### Chat with the Bot
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Parle-moi de Paris"}'
```

### Upload a Document
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your document text here",
    "metadata": {"title": "Example"}
  }'
```

## Development

### Backend
```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## How RAG Works

1. Documents are chunked and converted to embeddings
2. User queries retrieve relevant document chunks
3. CroissantLLM generates responses using retrieved context
4. Sources are cited with each response

## Troubleshooting

**Model takes too long to load**: First download takes 10-15 minutes

**Out of memory**: Ensure 8GB RAM available, close other applications

**Connection refused**: Wait for both containers to start, check logs

**Frontend can't connect**: Verify both services running with `docker-compose ps`

## Resources

- [CroissantLLM on Hugging Face](https://huggingface.co/croissantllm)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Documentation](https://python.langchain.com/)

## License

This project is provided as-is for educational and demonstration purposes.
