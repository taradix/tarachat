
import json
import logging
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import faiss
import torch
from attrs import define, field
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tarachat.config import Settings

logger = logging.getLogger(__name__)

_STOP_PHRASES = [
    "Question :",
    "Sources :",
    "Références :",
    "Notes et références",
    "---",
]

_NO_ANSWER = (
    "Je n'ai pas trouvé d'informations pertinentes dans les documents "
    "pour répondre à cette question."
)


@runtime_checkable
class RAGProtocol(Protocol):
    """Public interface for the RAG pipeline.

    Implemented by RAGPipeline (production) and FakeRAGSystem (tests).
    """

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> None: ...
    def retrieve_documents(self, query: str, k: int | None = None) -> list[Document]: ...
    def reset_vector_store(self) -> None: ...
    def chat(self, message: str, conversation_history: list[dict] | None = None) -> Generator[dict, None, None]: ...


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


def _load_vector_store(path: Path, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Load an existing vector store or create a new empty one."""
    if path.exists() and (path / "index.faiss").exists():
        logger.info("Loading existing vector store...")
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)

    logger.info("Creating new empty vector store...")
    vector_store = _create_empty_vector_store(embeddings)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))
    return vector_store


def _source_ref(doc: Document) -> str:
    """Build a citation reference like ``file.pdf#page=5`` from a Document.

    >>> from langchain_core.documents import Document
    >>> _source_ref(Document(page_content="x", metadata={"filename": "reg.pdf", "page": 3}))
    'reg.pdf#page=3'
    """
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


def _rrf_merge(
    ranked_lists: list[list[Document]],
    top_k: int,
    weights: list[float],
    rrf_k: int = 60,
) -> list[Document]:
    """Merge ranked document lists using weighted Reciprocal Rank Fusion.

    Each document's score is the weighted sum of ``weight / (rrf_k + rank)``
    across all lists, where rank is 1-based.  Documents appearing in multiple
    lists accumulate score from each.

    >>> from langchain_core.documents import Document
    >>> a = Document(page_content="alpha")
    >>> b = Document(page_content="beta")
    >>> c = Document(page_content="gamma")
    >>> result = _rrf_merge([[a, b], [b, c]], top_k=2, weights=[0.5, 0.5])
    >>> [d.page_content for d in result]
    ['beta', 'alpha']
    """
    scores: dict[int, float] = {}
    docs: dict[int, Document] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, doc in enumerate(ranked, start=1):
            key = hash(doc.page_content)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            docs[key] = doc
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [docs[k] for k in ordered[:top_k]]


@define
class Reranker:
    """Cross-encoder reranker that rescores (query, document) pairs.

    Typical usage: retrieve a large candidate set (e.g. top-20 with hybrid
    retrieval), then rerank to the final top-k.  The cross-encoder scores each
    ``(query, document)`` pair jointly, which is more accurate than embedding
    distance but too expensive to run on the full corpus.
    """

    model: Any  # sentence_transformers.CrossEncoder

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """Return the *top_k* docs sorted by descending cross-encoder score."""
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


def _extract_sources(docs: list[Document]) -> list[dict]:
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


@define
class Retriever:
    """Dense + BM25 hybrid retriever backed by a FAISS vector store."""

    settings: Settings
    vector_store: Any
    bm25_retriever: Any = field(default=None)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve relevant documents using hybrid BM25 + dense retrieval.

        When BM25 is available, merges keyword and semantic results with weighted
        Reciprocal Rank Fusion.  Falls back to pure dense retrieval otherwise.
        """
        if k is None:
            k = self.settings.top_k

        if self.vector_store.index.ntotal == 0:
            return []

        if self.bm25_retriever is None:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            threshold = self.settings.similarity_threshold
            return [doc for doc, score in results if threshold is None or score <= threshold]

        self.bm25_retriever.k = k
        bm25_docs = self.bm25_retriever.invoke(query)
        dense_docs = self.vector_store.similarity_search(query, k=k)
        w = self.settings.bm25_weight
        merged = _rrf_merge([bm25_docs, dense_docs], top_k=k, weights=[w, 1.0 - w])
        logger.info(
            f"Hybrid retrieval: BM25={len(bm25_docs)} dense={len(dense_docs)} "
            f"merged={len(merged)} (bm25_weight={w})"
        )
        return merged

    def add_documents(self, documents: list[Document]) -> None:
        """Add pre-split documents, rebuild BM25, and persist the vector store."""
        self.vector_store.add_documents(documents)
        self.bm25_retriever = self._build_bm25_retriever()
        path = Path(self.settings.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(path))

    def _build_bm25_retriever(self) -> Any:
        docs = list(self.vector_store.docstore._dict.values())
        if not docs:
            return None
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = self.settings.top_k
        return retriever


_SYSTEM_PROMPT = (
    "Tu es un assistant qui répond aux questions sur les règlements municipaux de Notre-Dame-du-Laus. "
    "Réponds UNIQUEMENT en te basant sur les extraits fournis dans le contexte. "
    "N'utilise jamais tes propres connaissances générales. "
    "Si la réponse ne se trouve pas dans les extraits, dis-le explicitement. "
    "Réponds en français, de façon concise et directe."
)


@define
class PromptBuilder:
    """Assembles the LLM prompt from context documents and conversation history."""

    settings: Settings

    def build(
        self,
        query: str,
        context_docs: list[Document],
        conversation_history: list[dict] | None = None,
    ) -> list[dict]:
        """Build the messages list for Ollama."""
        context = "\n\n".join(
            f"[{_source_ref(doc)}]: {doc.page_content}" for doc in context_docs
        )
        user_content = f"Contexte :\n{context}\n\nQuestion : {query}"
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if conversation_history:
            for msg in conversation_history[-self.settings.conversation_history_size:]:
                role = msg["role"] if isinstance(msg, dict) else msg.role
                content = msg["content"] if isinstance(msg, dict) else msg.content
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_content})
        return messages


@define
class OllamaGenerator:
    """Streams tokens from an Ollama-served model via its /api/chat endpoint."""

    settings: Settings

    def stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """Send *messages* to Ollama and yield content tokens as they arrive."""
        import httpx
        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": self.settings.max_tokens,
                "num_ctx": 2048,
                "temperature": 0,
                "repeat_penalty": 1.2,
                "stop": _STOP_PHRASES,
            },
        }
        url = f"{self.settings.ollama_url}/api/chat"
        with httpx.Client(timeout=300) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        break
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content


@define
class RAGPipeline:
    """Thin orchestrator: retrieve → prompt → generate."""

    settings: Settings
    text_splitter: Any
    embeddings: Any
    retriever: Retriever
    prompt_builder: Any = field(default=None)  # None when created for ingest-only use
    generator: Any = field(default=None)        # None when created for ingest-only use
    reranker: Any = field(default=None)

    @classmethod
    def _load_embeddings_and_retriever(
        cls, settings: Settings, device: str
    ) -> tuple[Any, Any, Any, Any]:
        """Shared setup: embeddings, vector store, text splitter, BM25 retriever."""
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={
                "device": device,
                "model_kwargs": {"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
            },
        )

        vector_store = _load_vector_store(Path(settings.vector_store_path), embeddings)

        # Token-aware chunking: use the embedding model's tokenizer as the
        # length function so chunk boundaries respect the model's context window.
        token_encoder = embeddings._client.tokenizer
        max_seq_len = embeddings._client.max_seq_length
        chunk_size = settings.chunk_size
        if chunk_size > max_seq_len:
            logger.warning(
                f"chunk_size={chunk_size} exceeds embedding model "
                f"max_seq_length={max_seq_len}; capping at {max_seq_len}"
            )
            chunk_size = max_seq_len

        def _token_length(text: str) -> int:
            return len(token_encoder.encode(text, add_special_tokens=False))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=_token_length,
        )

        # Seed BM25 from the loaded vector store (warm start after re-deploy).
        bm25_retriever = None
        if vector_store.index.ntotal > 0:
            docs = list(vector_store.docstore._dict.values())
            if docs:
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = settings.top_k

        retriever = Retriever(
            settings=settings,
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
        )
        return embeddings, text_splitter, retriever

    @classmethod
    def create(cls, settings: Settings, device: str) -> "RAGPipeline":
        """Create a fully initialized RAG pipeline."""
        logger.info(f"Using device: {device}")
        logger.info("Initializing RAG pipeline...")

        embeddings, text_splitter, retriever = cls._load_embeddings_and_retriever(settings, device)

        reranker = None
        if settings.reranker_model:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {settings.reranker_model}")
            reranker = Reranker(model=CrossEncoder(settings.reranker_model, device=device))

        logger.info(f"Using Ollama model: {settings.ollama_model} at {settings.ollama_url}")
        generator = OllamaGenerator(settings=settings)
        prompt_builder = PromptBuilder(settings=settings)

        logger.info("RAG pipeline initialized successfully")
        return cls(
            settings=settings,
            text_splitter=text_splitter,
            embeddings=embeddings,
            retriever=retriever,
            prompt_builder=prompt_builder,
            generator=generator,
            reranker=reranker,
        )

    @classmethod
    def create_for_ingest(cls, settings: Settings, device: str) -> "RAGPipeline":
        """Create a pipeline with only the embedding model loaded (no LLM).

        Use this for document ingestion to avoid loading the language model
        into memory when only the vector store needs to be updated.
        """
        logger.info(f"Using device: {device}")
        logger.info("Initializing RAG pipeline (ingest mode — no LLM)...")

        embeddings, text_splitter, retriever = cls._load_embeddings_and_retriever(settings, device)

        logger.info("RAG pipeline (ingest mode) initialized successfully")
        return cls(
            settings=settings,
            text_splitter=text_splitter,
            embeddings=embeddings,
            retriever=retriever,
        )

    def add_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """Split texts into chunks and add to the retriever."""
        if not texts:
            return

        logger.info(f"Adding {len(texts)} documents...")

        documents = []
        for i, text in enumerate(texts):
            base_metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            for page_num, page_text in _split_by_pages(text):
                chunks = self.text_splitter.split_text(page_text)
                for j, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={**base_metadata, "chunk": j, "page": page_num},
                    ))

        self.retriever.add_documents(documents)
        logger.info(f"Added {len(documents)} chunks")

    def retrieve_documents(self, query: str, k: int | None = None) -> list[Document]:
        """Delegate to the retriever."""
        return self.retriever.retrieve(query, k=k)

    def reset_vector_store(self) -> None:
        """Replace the retriever's vector store with an empty one and persist it."""
        self.retriever.vector_store = _create_empty_vector_store(self.embeddings)
        path = Path(self.settings.vector_store_path)
        path.mkdir(parents=True, exist_ok=True)
        self.retriever.vector_store.save_local(str(path))

    def chat(
        self, message: str, conversation_history: list[dict] | None = None,
    ) -> Generator[dict, None, None]:
        """Orchestrate retrieval, prompt building, and generation."""
        candidate_k = self.settings.rerank_candidates if self.reranker is not None else self.settings.top_k
        docs = self.retriever.retrieve(message, k=candidate_k)
        if self.reranker is not None:
            docs = self.reranker.rerank(message, docs, top_k=self.settings.top_k)
        sources = _extract_sources(docs)

        if self.settings.demo_mode:
            logger.info("Using demo mode (fast RAG-only responses)")
            if docs:
                parts = [
                    f"[{i+1}/{len(docs)}] [{_source_ref(doc)}]:\n{doc.page_content.strip()}"
                    for i, doc in enumerate(docs)
                ]
                demo_content = "Voici ce que j'ai trouvé dans les documents:\n\n" + "\n\n---\n\n".join(parts)
            else:
                demo_content = (
                    "Désolé, je n'ai pas trouvé d'informations pertinentes dans la base "
                    "de connaissances pour répondre à votre question."
                )
            yield {"type": "token", "content": demo_content}
            yield {"type": "sources", "sources": sources}
            return

        if not docs:
            logger.info("No relevant documents found; skipping LLM")
            yield {"type": "token", "content": _NO_ANSWER}
            yield {"type": "sources", "sources": []}
            return

        logger.info("Using full LLM mode with streaming")
        messages = self.prompt_builder.build(message, docs, conversation_history)
        for token in self.generator.stream(messages):
            yield {"type": "token", "content": token}
        yield {"type": "sources", "sources": sources}
