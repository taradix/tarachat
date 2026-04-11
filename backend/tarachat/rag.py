
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
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
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
    "---",
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
    ) -> Generator[dict, None, None]: ...

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
        return FAISS.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True,
        )

    logger.info("Creating new empty vector store...")
    vector_store = _create_empty_vector_store(embeddings)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))
    return vector_store


def _source_ref(doc: Document) -> str:
    """Build a citation reference like ``file.pdf#page=5`` from a Document."""
    filename = doc.metadata.get("filename", "")
    page = doc.metadata.get("page")
    if page is None:
        m = re.search(r"\[Page (\d+)\]", doc.page_content)
        page = int(m.group(1)) if m else 1
    return f"{filename}#page={page}"


def _split_by_pages(text: str) -> list[tuple[int, str]]:
    """Split extracted text into ``(page_number, page_text)`` pairs.

    Expects text produced by :func:`pdf.extract_text` with ``[Page N]``
    markers.  Text before the first marker (if any) is assigned to page 1.
    """
    parts = re.split(r"\[Page (\d+)\]\n?", text)
    # parts = ['preamble', '1', 'text…', '2', 'text…', …]
    sections: list[tuple[int, str]] = []
    if parts[0].strip():
        sections.append((1, parts[0].strip()))
    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if page_text:
            sections.append((page_num, page_text))
    return sections


@define
class RAGSystem:
    """RAG system using CroissantLLM and FAISS."""

    settings: Settings
    device: str
    embeddings: Any
    vector_store: Any
    tokenizer: Any
    model: Any
    text_splitter: Any

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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )

        logger.info("RAG system initialized successfully")
        return cls(
            settings=settings,
            device=device,
            embeddings=embeddings,
            vector_store=vector_store,
            tokenizer=tokenizer,
            model=model,
            text_splitter=text_splitter,
        )

    def create_empty_vector_store(self) -> FAISS:
        """Create a new empty FAISS vector store."""
        return _create_empty_vector_store(self.embeddings)

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None):
        """Add documents to the vector store."""
        if not texts:
            return

        logger.info(f"Adding {len(texts)} documents to vector store...")

        documents = []
        for i, text in enumerate(texts):
            base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

            # Split by page first so every chunk keeps its page number
            for page_num, page_text in _split_by_pages(text):
                chunks = self.text_splitter.split_text(page_text)
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={**base_metadata, "chunk": j, "page": page_num},
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
        """Retrieve relevant documents for a query, filtered by similarity threshold."""
        if k is None:
            k = self.settings.top_k

        if self.vector_store.index.ntotal == 0:
            return []

        results = self.vector_store.similarity_search_with_score(query, k=k)
        threshold = self.settings.similarity_threshold
        filtered = []
        for doc, score in results:
            logger.debug(f"Doc score={score:.4f} threshold={threshold} file={doc.metadata.get('filename','?')} page={doc.metadata.get('page','?')}")
            if threshold is None or score <= threshold:
                filtered.append(doc)
        logger.info(f"Retrieved {len(results)} docs, {len(filtered)} within threshold {threshold}")
        return filtered

    def _build_prompt(
        self,
        query: str,
        context_docs: list[Document],
        conversation_history: list[dict] | None = None,
    ):
        """Build the LLM prompt from context, history, and query."""
        context_parts = []
        for doc in context_docs:
            ref = _source_ref(doc)
            context_parts.append(f"[{ref}]: {doc.page_content}")
        context = "\n\n".join(context_parts)

        system = (
            "Tu es un assistant qui répond aux questions sur les règlements municipaux. "
            "Réponds UNIQUEMENT à partir des extraits fournis dans le contexte. "
            "N'utilise jamais tes propres connaissances générales. "
            "Si le contexte ne contient pas la réponse, dis-le explicitement. "
            "Cite tes sources sous la forme [fichier.pdf#page=N]."
        )
        user_content = f"Contexte :\n{context}\n\nQuestion : {query}"

        messages = [{"role": "system", "content": system}]
        if conversation_history:
            history_size = self.settings.conversation_history_size
            for msg in conversation_history[-history_size:]:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_demo_response(self, docs: list[Document]) -> str:
        """Build a demo-mode response from retrieved documents (no LLM)."""
        if docs:
            parts = []
            for doc in docs[:2]:
                ref = _source_ref(doc)
                snippet = doc.page_content[:300].strip()
                parts.append(f"{snippet} [{ref}]")
            return "Voici ce que j'ai trouvé dans les documents:\n\n" + "\n\n".join(parts)
        return "Désolé, je n'ai pas trouvé d'informations pertinentes dans la base de connaissances pour répondre à votre question."

    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        """Extract structured source info from retrieved documents.

        Deduplicates by (filename, page) and collects multiple highlight
        snippets per source for richer PDF highlighting.
        """
        seen: dict[tuple[str, int], list[str]] = {}
        order: list[tuple[str, int]] = []
        for doc in docs:
            filename = doc.metadata.get("filename", "")
            page = doc.metadata.get("page")
            if page is None:
                m = re.search(r"\[Page (\d+)\]", doc.page_content)
                page = int(m.group(1)) if m else 1
            snippet = doc.page_content[:120].strip()
            key = (filename, page)
            if key not in seen:
                seen[key] = []
                order.append(key)
            if snippet and snippet not in seen[key]:
                seen[key].append(snippet)
        return [
            {"filename": f, "page": p, "highlights": seen[(f, p)]}
            for f, p in order
        ]

    def _tokenize_prompt(self, prompt: str) -> dict:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _generation_kwargs(self, inputs: dict, max_length: int) -> dict:
        """Build the shared generation kwargs for model.generate()."""
        return {
            **inputs,
            "max_new_tokens": max_length,
            "temperature": 0.3,
            "top_p": 0.85,
            "do_sample": True,
            "num_beams": 1,
            "repetition_penalty": 1.2,
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

    def chat(self, message: str, conversation_history: list[dict] | None = None) -> Generator[dict, None, None]:
        """Process a chat message with RAG, yielding event dicts as tokens stream."""
        docs = self.retrieve_documents(message)
        sources = self._extract_sources(docs)

        if self.settings.demo_mode:
            logger.info("Using demo mode (fast RAG-only responses)")
            response = self._build_demo_response(docs)
            yield {"type": "token", "content": response}
            yield {"type": "sources", "sources": sources}
            return

        if not docs:
            logger.info("No relevant documents found; skipping LLM")
            no_info = "Je n'ai pas trouvé d'informations pertinentes dans les documents pour répondre à cette question."
            yield {"type": "token", "content": no_info}
            yield {"type": "sources", "sources": []}
            return

        # Normal mode: stream tokens
        logger.info("Using full LLM mode with streaming")
        for token in self._stream_tokens(message, docs, conversation_history):
            yield {"type": "token", "content": token}

        yield {"type": "sources", "sources": sources}
