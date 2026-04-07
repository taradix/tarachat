import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    ChatRequest,
    ChatResponse,
    DocumentUpload,
    HealthResponse
)
from app.rag import rag_system
from app.config import get_settings
from app.pdf_processor import pdf_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # Startup
    logger.info("Starting up application...")
    try:
        rag_system.initialize()
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
async def health_check():
    """Health check endpoint."""
    is_ready = rag_system.is_ready()
    return HealthResponse(
        status="healthy" if is_ready else "initializing",
        model_loaded=rag_system.model is not None,
        vector_store_ready=rag_system.vector_store is not None
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """Chat endpoint with RAG."""
    try:
        if not rag_system.is_ready():
            raise HTTPException(
                status_code=503,
                detail="RAG system is not ready yet. Please try again later."
            )

        # Process the chat message
        response, sources = rag_system.chat(request.message)

        return ChatResponse(
            response=response,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
