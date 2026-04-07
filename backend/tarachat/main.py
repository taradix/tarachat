import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from tarachat.models import (
    ChatRequest,
    ChatResponse,
    DocumentUpload,
    HealthResponse
)
from tarachat.rag import RAGSystem, rag_system
from tarachat.config import get_settings
from tarachat.pdf_processor import pdf_processor

logger = logging.getLogger(__name__)


def get_rag_system() -> RAGSystem:
    """Dependency that provides the RAG system instance."""
    return rag_system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    resolve = app.dependency_overrides.get(get_rag_system, get_rag_system)
    rag = resolve()

    logger.info("Starting up application...")
    try:
        rag.initialize()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="TaraChat API",
    description="RAG-based chatbot API using CroissantLLM",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to TaraChat API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(rag: RAGSystem = Depends(get_rag_system)):
    """Health check endpoint."""
    is_ready = rag.is_ready()
    return HealthResponse(
        status="healthy" if is_ready else "initializing",
        model_loaded=rag.model is not None,
        vector_store_ready=rag.vector_store is not None
    )


@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest, rag: RAGSystem = Depends(get_rag_system)):
    """Chat endpoint with RAG. Returns a Server-Sent Events stream."""
    if not rag.is_ready():
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready yet. Please try again later."
        )

    history = request.conversation_history or []

    def event_generator():
        try:
            yield from rag.chat_stream(request.message, history)
        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            yield f'data: {{"type": "error", "content": "An internal error occurred."}}\n\n'
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "tarachat.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
