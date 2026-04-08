import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from tarachat.config import get_settings
from tarachat.models import ChatRequest, HealthResponse
from tarachat.rag import RAGProtocol, RAGSystem, _detect_device

logger = logging.getLogger(__name__)


def get_rag_system(request: Request) -> RAGProtocol:
    """Dependency that provides the RAG system instance."""
    return request.app.state.rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    if not hasattr(app.state, "rag"):  # pragma: no cover
        app.state.rag = RAGSystem.create(
            settings=get_settings(), device=_detect_device(),
        )
    yield
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
    return HealthResponse(status="healthy")


@app.post("/chat", tags=["Chat"])
async def chat(request: ChatRequest, rag: RAGProtocol = Depends(get_rag_system)):
    """Chat endpoint with RAG. Returns a Server-Sent Events stream."""
    history = request.conversation_history or []

    def event_generator():
        try:
            yield from rag.chat(request.message, history)
        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            yield 'data: {"type": "error", "content": "An internal error occurred."}\n\n'
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
