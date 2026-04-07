import os
import json
import logging
from typing import List, Tuple, Generator
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import faiss
import torch
from threading import Thread

from app.config import get_settings

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG system using CroissantLLM and FAISS."""

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = None
        self.vector_store = None
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {self.device}")

    def create_empty_vector_store(self) -> FAISS:
        """Create a new empty FAISS vector store."""
        sample_embedding = self.embeddings.embed_query("sample")
        dimension = len(sample_embedding)
        index = faiss.IndexFlatL2(dimension)
        return FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )

    def initialize(self):
        """Initialize the RAG system components."""
        logger.info("Initializing RAG system...")

        # Initialize embeddings
        logger.info(f"Loading embedding model: {self.settings.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model,
            model_kwargs={'device': self.device}
        )

        # Initialize or load vector store
        vector_store_path = Path(self.settings.vector_store_path)
        if vector_store_path.exists() and (vector_store_path / "index.faiss").exists():
            logger.info("Loading existing vector store...")
            try:
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except TypeError:
                logger.info("Using older FAISS.load_local signature...")
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings
                )
        else:
            logger.info("Creating new empty vector store...")
            self.vector_store = self.create_empty_vector_store()
            vector_store_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(vector_store_path))

        # Initialize LLM
        logger.info(f"Loading language model: {self.settings.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        logger.info("RAG system initialized successfully")

    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
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

    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
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
        context_docs: List[Document],
        conversation_history: List[dict] = None,
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

    def _build_demo_response(self, docs: List[Document]) -> str:
        """Build a demo-mode response from retrieved documents (no LLM)."""
        if docs:
            snippets = [doc.page_content[:300].strip() for doc in docs[:2]]
            response = f"Voici ce que j'ai trouvé dans les documents:\n\n{snippets[0]}"
            if len(snippets) > 1:
                response += f"\n\n{snippets[1]}"
            return response
        return "Désolé, je n'ai pas trouvé d'informations pertinentes dans la base de connaissances pour répondre à votre question."

    def _extract_sources(self, docs: List[Document]) -> List[str]:
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

    def generate_response(
        self,
        query: str,
        context_docs: List[Document],
        conversation_history: List[dict] = None,
        max_length: int = None
    ) -> str:
        """Generate a response using the LLM with context."""
        if max_length is None:
            max_length = self.settings.max_tokens

        prompt = self._build_prompt(query, context_docs, conversation_history)
        inputs = self._tokenize_prompt(prompt)

        logger.info(f"Generating response for query: {query[:50]}...")

        with torch.no_grad():
            outputs = self.model.generate(**self._generation_kwargs(inputs, max_length))

        logger.info("Response generation complete")

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated response (after the prompt)
        if "Réponse :" in response:
            response = response.split("Réponse :")[-1].strip()

        return response

    def generate_response_stream(
        self,
        query: str,
        context_docs: List[Document],
        conversation_history: List[dict] = None,
        max_length: int = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response, yielding tokens as they are produced."""
        if max_length is None:
            max_length = self.settings.max_tokens

        prompt = self._build_prompt(query, context_docs, conversation_history)
        inputs = self._tokenize_prompt(prompt)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = {**self._generation_kwargs(inputs, max_length), "streamer": streamer}

        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()

        for token_text in streamer:
            yield token_text

        thread.join()

    def chat(self, message: str, conversation_history: List[dict] = None) -> Tuple[str, List[str]]:
        """Process a chat message with RAG."""

        # Retrieve relevant documents
        docs = self.retrieve_documents(message)

        if self.settings.demo_mode:
            logger.info("Using demo mode (fast RAG-only responses)")
            return self._build_demo_response(docs), self._extract_sources(docs)

        # Normal mode: Full LLM generation (slow on CPU)
        logger.info("Using full LLM mode (slow on CPU without GPU)")
        response = self.generate_response(message, docs, conversation_history)
        return response, self._extract_sources(docs)

    def chat_stream(self, message: str, conversation_history: List[dict] = None) -> Generator[str, None, None]:
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
        for token in self.generate_response_stream(message, docs, conversation_history):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        yield "data: [DONE]\n\n"

    def is_ready(self) -> bool:
        """Check if the RAG system is ready."""
        return (
            self.embeddings is not None
            and self.vector_store is not None
            and self.model is not None
            and self.tokenizer is not None
        )


# Global RAG system instance
rag_system = RAGSystem()
