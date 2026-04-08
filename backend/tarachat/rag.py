
import json
import logging
from collections.abc import Generator
from pathlib import Path
from threading import Thread
from typing import Any, Protocol, runtime_checkable

import faiss
import torch
from attrs import define
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from tarachat.config import Settings

logger = logging.getLogger(__name__)


@runtime_checkable
class RAGProtocol(Protocol):
    """Public interface for the RAG system.

    Implemented by RAGSystem (production) and FakeRAGSystem (tests).
    """

    model: Any
    vector_store: Any

    def add_documents(
        self, texts: list[str], metadatas: list[dict] | None = None,
    ) -> None: ...

    def retrieve_documents(
        self, query: str, k: int | None = None,
    ) -> list[Document]: ...

    def chat(
        self, message: str, conversation_history: list[dict] | None = None,
    ) -> Generator[str, None, None]: ...

    def create_empty_vector_store(self) -> Any: ...


def _detect_device() -> str:
    """Detect the best available compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _create_empty_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Create a new empty FAISS vector store."""
    sample_embedding = embeddings.embed_query("sample")
    dimension = len(sample_embedding)
    index = faiss.IndexFlatL2(dimension)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )


def _load_vector_store(
    path: Path, embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Load an existing vector store or create a new empty one."""
    if path.exists() and (path / "index.faiss").exists():
        logger.info("Loading existing vector store...")
        try:
            return FAISS.load_local(
                str(path), embeddings, allow_dangerous_deserialization=True,
            )
        except TypeError:
            logger.info("Using older FAISS.load_local signature...")
            return FAISS.load_local(str(path), embeddings)

    logger.info("Creating new empty vector store...")
    vector_store = _create_empty_vector_store(embeddings)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))
    return vector_store


@define
class RAGSystem:
    """RAG system using CroissantLLM and FAISS."""

    settings: Settings
    device: str
    embeddings: Any
    vector_store: Any
    tokenizer: Any
    model: Any

    @classmethod
    def create(cls, settings: Settings, device: str) -> "RAGSystem":
        """Create a fully initialized RAG system."""
        logger.info(f"Using device: {device}")
        logger.info("Initializing RAG system...")

        logger.info(f"Loading embedding model: {settings.embedding_model}")
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
        )

        vector_store = _load_vector_store(
            Path(settings.vector_store_path), embeddings,
        )

        logger.info(f"Loading language model: {settings.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            model = model.to(device)

        logger.info("RAG system initialized successfully")
        return cls(
            settings=settings,
            device=device,
            embeddings=embeddings,
            vector_store=vector_store,
            tokenizer=tokenizer,
            model=model,
        )

    def create_empty_vector_store(self) -> FAISS:
        """Create a new empty FAISS vector store."""
        return _create_empty_vector_store(self.embeddings)

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None):
        """Add documents to the vector store."""
        if not texts:
            return

        logger.info(f"Adding {len(texts)} documents to vector store...")

        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
        )

        documents = []
        for i, text in enumerate(texts):
            chunks = text_splitter.split_text(text)
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

            for j, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={**metadata, "chunk": j}
                )
                documents.append(doc)

        # Add to vector store
        self.vector_store.add_documents(documents)

        # Save vector store
        vector_store_path = Path(self.settings.vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(vector_store_path))

        logger.info(f"Added {len(documents)} chunks to vector store")

    def retrieve_documents(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve relevant documents for a query."""
        if k is None:
            k = self.settings.top_k

        if self.vector_store.index.ntotal == 0:
            return []

        results = self.vector_store.similarity_search(query, k=k)
        return results

    def _build_prompt(
        self,
        query: str,
        context_docs: list[Document],
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Build the LLM prompt from context, history, and query."""
        context = "\n\n".join([doc.page_content for doc in context_docs])

        history_text = ""
        if conversation_history:
            recent = conversation_history[-6:]
            history_lines = []
            for msg in recent:
                role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.get('content', '')}")
            history_text = "\n".join(history_lines)

        if history_text:
            return f"""Voici du contexte pertinent :

{context}

Historique de la conversation :
{history_text}

Question : {query}

Réponse :"""
        else:
            return f"""Voici du contexte pertinent :

{context}

Question : {query}

Réponse :"""

    def _build_demo_response(self, docs: list[Document]) -> str:
        """Build a demo-mode response from retrieved documents (no LLM)."""
        if docs:
            snippets = [doc.page_content[:300].strip() for doc in docs[:2]]
            response = f"Voici ce que j'ai trouvé dans les documents:\n\n{snippets[0]}"
            if len(snippets) > 1:
                response += f"\n\n{snippets[1]}"
            return response
        return "Désolé, je n'ai pas trouvé d'informations pertinentes dans la base de connaissances pour répondre à votre question."

    def _extract_sources(self, docs: list[Document]) -> list[str]:
        """Extract source previews from retrieved documents."""
        return [doc.page_content[:100] + "..." for doc in docs]

    def _tokenize_prompt(self, prompt: str) -> dict:
        """Tokenize a prompt and move tensors to the target device."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _generation_kwargs(self, inputs: dict, max_length: int) -> dict:
        """Build the shared generation kwargs for model.generate()."""
        return {
            **inputs,
            "max_new_tokens": max_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def _stream_tokens(
        self,
        query: str,
        context_docs: list[Document],
        conversation_history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response, yielding tokens as they are produced."""
        max_length = self.settings.max_tokens
        prompt = self._build_prompt(query, context_docs, conversation_history)
        inputs = self._tokenize_prompt(prompt)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = {**self._generation_kwargs(inputs, max_length), "streamer": streamer}

        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        yield from streamer

        thread.join()

    def chat(self, message: str, conversation_history: list[dict] | None = None) -> Generator[str, None, None]:
        """Process a chat message with RAG, yielding SSE events as tokens stream."""
        docs = self.retrieve_documents(message)
        sources = self._extract_sources(docs)

        if self.settings.demo_mode:
            logger.info("Using demo mode (fast RAG-only responses)")
            response = self._build_demo_response(docs)
            yield f"data: {json.dumps({'type': 'token', 'content': response})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Normal mode: stream tokens
        logger.info("Using full LLM mode with streaming")
        for token in self._stream_tokens(message, docs, conversation_history):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield "data: [DONE]\n\n"
