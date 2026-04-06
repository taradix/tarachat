from pydantic import BaseModel, Field
from typing import List, Optional


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message", min_length=1)
    conversation_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Previous conversation history"
    )


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Assistant response")
    sources: List[str] = Field(
        default_factory=list,
        description="Source documents used for RAG"
    )


class DocumentUpload(BaseModel):
    """Document upload model."""
    content: str = Field(..., description="Document content")
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Document metadata"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    vector_store_ready: bool
