
import json
import logging
import re
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from tarachat.config import Settings

logger = logging.getLogger(__name__)

_STOP_PHRASES = [
    "Question :",
    "Sources :",
    "Références :",
    "Notes et références",
]


class _StopOnPhrase(StoppingCriteria):
    """Stop generation when the decoded tail contains any stop phrase."""

    def __init__(self, tokenizer: AutoTokenizer, stop_phrases: list[str]) -> None:
        self._tokenizer = tokenizer
        self._stop_phrases = stop_phrases
        self._check_len = max(
            len(tokenizer.encode(p, add_special_tokens=False)) + 2
            for p in stop_phrases
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        tail = self._tokenizer.decode(input_ids[0, -self._check_len :], skip_special_tokens=True)
        return any(p in tail for p in self._stop_phrases)


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


def _source_ref(doc: Document) -> str:
    """Build a citation reference like ``file.pdf#page=5`` from a Document."""
    filename = doc.metadata.get("filename", "")
    m = re.search(r"\[Page (\d+)\]", doc.page_content)
    page = int(m.group(1)) if m else 1
    return f"{filename}#page={page}"


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
        context_parts = []
        for doc in context_docs:
            ref = _source_ref(doc)
            text = re.sub(r"\[Page \d+\]\n?", "", doc.page_content).strip()
            context_parts.append(f"[{ref}]: {text}")
        context = "\n\n".join(context_parts)

        citation_instruction = (
            "Cite tes sources entre crochets, par exemple [fichier.pdf#page=3]. "
            "Ne combine pas les sources : [a.pdf#page=1][b.pdf#page=2]."
        )

        history_text = ""
        if conversation_history:
            recent = conversation_history[-6:]
            history_lines = []
            for msg in recent:
                role = "Utilisateur" if msg.role == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.content}")
            history_text = "\n".join(history_lines)

        if history_text:
            return f"""{citation_instruction}

Voici du contexte pertinent :

{context}

Historique de la conversation :
{history_text}

Question : {query}

Réponse :"""
        else:
            return f"""{citation_instruction}

Voici du contexte pertinent :

{context}

Question : {query}

Réponse :"""

    def _build_demo_response(self, docs: list[Document]) -> str:
        """Build a demo-mode response from retrieved documents (no LLM)."""
        if docs:
            parts = []
            for doc in docs[:2]:
                ref = _source_ref(doc)
                text = re.sub(r"\[Page \d+\]\n?", "", doc.page_content).strip()
                snippet = text[:300].strip()
                parts.append(f"{snippet} [{ref}]")
            return "Voici ce que j'ai trouvé dans les documents:\n\n" + "\n\n".join(parts)
        return "Désolé, je n'ai pas trouvé d'informations pertinentes dans la base de connaissances pour répondre à votre question."

    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        """Extract structured source info from retrieved documents.

        Each source includes the filename, starting page number, and
        a short text snippet suitable for highlighting in the PDF.
        """
        sources: list[dict] = []
        for doc in docs:
            filename = doc.metadata.get("filename", "")
            # Extract first [Page N] marker from chunk content
            m = re.search(r"\[Page (\d+)\]", doc.page_content)
            page = int(m.group(1)) if m else 1
            # Use first ~120 chars of actual text (skip the [Page N] marker) as highlight
            text = re.sub(r"\[Page \d+\]\n?", "", doc.page_content).strip()
            snippet = text[:120].strip()
            sources.append({"filename": filename, "page": page, "snippet": snippet})
        return sources

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
            "stopping_criteria": StoppingCriteriaList([
                _StopOnPhrase(self.tokenizer, _STOP_PHRASES),
            ]),
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

        buffer = ""
        max_hold = max(len(p) for p in _STOP_PHRASES)
        for token in streamer:
            buffer += token
            # Stop yielding if any stop phrase appears in accumulated text
            hit = next((p for p in _STOP_PHRASES if p in buffer), None)
            if hit:
                clean = buffer[: buffer.index(hit)].rstrip()
                if clean:
                    yield clean
                break
            # Yield everything except a trailing window that could match
            safe = len(buffer) - max_hold
            if safe > 0:
                yield buffer[:safe]
                buffer = buffer[safe:]

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
