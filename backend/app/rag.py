import os
import logging
from typing import List, Tuple
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
            # Try with allow_dangerous_deserialization parameter for newer versions
            try:
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except TypeError:
                # Fallback for older versions without this parameter
                logger.info("Using older FAISS.load_local signature...")
                self.vector_store = FAISS.load_local(
                    str(vector_store_path),
                    self.embeddings
                )
        else:
            logger.info("Creating new empty vector store...")
            # Create an empty FAISS index using the embedding dimension
            import faiss
            import numpy as np
            sample_embedding = self.embeddings.embed_query("sample")
            dimension = len(sample_embedding)
            index = faiss.IndexFlatL2(dimension)
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )
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

    def generate_response(
        self,
        query: str,
        context_docs: List[Document],
        conversation_history: List[dict] = None,
        max_length: int = 128
    ) -> str:
        """Generate a response using the LLM with context."""

        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in context_docs])

        # Build conversation history section
        history_text = ""
        if conversation_history:
            recent = conversation_history[-6:]
            history_lines = []
            for msg in recent:
                role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
                history_lines.append(f"{role}: {msg.get('content', '')}")
            history_text = "\n".join(history_lines)

        # Create prompt
        if history_text:
            prompt = f"""Voici du contexte pertinent :

{context}

Historique de la conversation :
{history_text}

Question : {query}

Réponse :"""
        else:
            prompt = f"""Voici du contexte pertinent :

{context}

Question : {query}

Réponse :"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logger.info(f"Generating response for query: {query[:50]}...")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_beams=1,  # Disable beam search for faster generation
                pad_token_id=self.tokenizer.eos_token_id
            )

        logger.info("Response generation complete")

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated response (after the prompt)
        if "Réponse :" in response:
            response = response.split("Réponse :")[-1].strip()

        return response

    def chat(self, message: str, conversation_history: List[dict] = None) -> Tuple[str, List[str]]:
        """Process a chat message with RAG."""

        # Retrieve relevant documents
        docs = self.retrieve_documents(message)

        # Demo mode: Fast responses without LLM generation
        if self.settings.demo_mode:
            logger.info("Using demo mode (fast RAG-only responses)")
            if docs and len(docs) > 0:
                # Create a response from the retrieved documents
                context_snippets = []
                for i, doc in enumerate(docs[:2], 1):  # Use top 2 docs
                    snippet = doc.page_content[:300].strip()
                    context_snippets.append(snippet)

                # Simple response construction
                response = f"Voici ce que j'ai trouvé dans les documents:\n\n{context_snippets[0]}"
                if len(context_snippets) > 1:
                    response += f"\n\n{context_snippets[1]}"
            else:
                response = "Désolé, je n'ai pas trouvé d'informations pertinentes dans la base de connaissances pour répondre à votre question."

            # Extract sources
            sources = [doc.page_content[:100] + "..." for doc in docs]
            return response, sources

        # Normal mode: Full LLM generation (slow on CPU)
        logger.info("Using full LLM mode (slow on CPU without GPU)")
        response = self.generate_response(message, docs, conversation_history)

        # Extract sources
        sources = [doc.page_content[:100] + "..." for doc in docs]

        return response, sources

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
