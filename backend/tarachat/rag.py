
import logging
import re
from collections.abc import Generator
from pathlib import Path
from threading import Thread
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

_NO_ANSWER = (
    "Je n'ai pas trouvé d'informations pertinentes dans les documents "
    "pour répondre à cette question."
)


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
        tail = self._tokenizer.decode(input_ids[0, -self._check_len:], skip_special_tokens=True)
        return any(p in tail for p in self._stop_phrases)


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
    for ranked, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked, start=1):
            key = hash(doc.page_content)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            docs[key] = doc
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [docs[k] for k in ordered[:top_k]]


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


@define
class PromptBuilder:
    """Assembles the LLM prompt from context documents and conversation history."""

    settings: Settings
    tokenizer: Any

    def build(
        self,
        query: str,
        context_docs: list[Document],
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Build a tokenizer-formatted prompt string."""
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


@define
class LLMGenerator:
    """Streams tokens from a causal language model."""

    settings: Settings
    tokenizer: Any
    model: Any
    device: str

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate a streaming response, yielding tokens as they are produced."""
        max_length = self.settings.max_tokens
        inputs = self._tokenize(prompt)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = {**self._generation_kwargs(inputs, max_length), "streamer": streamer}

        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        buffer = ""
        max_hold = max(len(p) for p in _STOP_PHRASES)
        for token in streamer:
            buffer += token
            hit = next((p for p in _STOP_PHRASES if p in buffer), None)
            if hit:
                clean = buffer[: buffer.index(hit)].rstrip()
                if clean:
                    yield clean
                break
            safe = len(buffer) - max_hold
            if safe > 0:
                yield buffer[:safe]
                buffer = buffer[safe:]

        thread.join()

    def demo_response(self, docs: list[Document]) -> str:
        """Build a demo-mode response from retrieved documents (no LLM)."""
        if docs:
            parts = []
            for doc in docs[:2]:
                ref = _source_ref(doc)
                snippet = doc.page_content[:300].strip()
                parts.append(f"{snippet} [{ref}]")
            return "Voici ce que j'ai trouvé dans les documents:\n\n" + "\n\n".join(parts)
        return (
            "Désolé, je n'ai pas trouvé d'informations pertinentes dans la base "
            "de connaissances pour répondre à votre question."
        )

    def _tokenize(self, prompt: str) -> dict:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _generation_kwargs(self, inputs: dict, max_length: int) -> dict:
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


@define
class RAGPipeline:
    """Thin orchestrator: retrieve → prompt → generate."""

    settings: Settings
    text_splitter: Any
    embeddings: Any
    retriever: Retriever
    prompt_builder: PromptBuilder
    generator: LLMGenerator

    @classmethod
    def create(cls, settings: Settings, device: str) -> "RAGPipeline":
        """Create a fully initialized RAG pipeline."""
        logger.info(f"Using device: {device}")
        logger.info("Initializing RAG pipeline...")

        logger.info(f"Loading embedding model: {settings.embedding_model}")
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
        )

        vector_store = _load_vector_store(Path(settings.vector_store_path), embeddings)

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

        # Token-aware chunking: use the embedding model's tokenizer as the
        # length function so chunk boundaries respect the model's context window.
        token_encoder = embeddings.client.tokenizer
        max_seq_len = embeddings.client.max_seq_length
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

        logger.info("RAG pipeline initialized successfully")
        return cls(
            settings=settings,
            text_splitter=text_splitter,
            embeddings=embeddings,
            retriever=Retriever(
                settings=settings,
                vector_store=vector_store,
                bm25_retriever=bm25_retriever,
            ),
            prompt_builder=PromptBuilder(settings=settings, tokenizer=tokenizer),
            generator=LLMGenerator(
                settings=settings,
                tokenizer=tokenizer,
                model=model,
                device=device,
            ),
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
        docs = self.retriever.retrieve(message)
        sources = _extract_sources(docs)

        if self.settings.demo_mode:
            logger.info("Using demo mode (fast RAG-only responses)")
            yield {"type": "token", "content": self.generator.demo_response(docs)}
            yield {"type": "sources", "sources": sources}
            return

        if not docs:
            logger.info("No relevant documents found; skipping LLM")
            yield {"type": "token", "content": _NO_ANSWER}
            yield {"type": "sources", "sources": []}
            return

        logger.info("Using full LLM mode with streaming")
        prompt = self.prompt_builder.build(message, docs, conversation_history)
        for token in self.generator.stream(prompt):
            yield {"type": "token", "content": token}
        yield {"type": "sources", "sources": sources}


# Backward-compatible alias: app.py and ingest.py can keep importing RAGSystem
RAGSystem = RAGPipeline
