#!/usr/bin/env python3
"""
Document ingestion script for managing the vector store.
Supports both text (.txt) and PDF (.pdf) files.

Usage:
    # Add documents from directory (text files)
    python scripts/ingest_documents.py add --dir data/documents/

    # Add PDF documents from directory
    python scripts/ingest_documents.py add --dir data/documents/ --pattern "*.pdf"

    # Add both text and PDF files
    python scripts/ingest_documents.py add --dir data/documents/ --pattern "*.*"

    # Add a single text document
    python scripts/ingest_documents.py add --file data/mydoc.txt --id mydoc

    # Add a single PDF document
    python scripts/ingest_documents.py add --file data/report.pdf --id report

    # Add with metadata (works for both text and PDF)
    python scripts/ingest_documents.py add --file data/mydoc.pdf --id mydoc --metadata '{"author": "John", "date": "2024"}'

    # Update an existing document (auto-detects file type)
    python scripts/ingest_documents.py update --id mydoc --file data/mydoc_v2.pdf

    # Delete a document
    python scripts/ingest_documents.py delete --id mydoc

    # List all documents
    python scripts/ingest_documents.py list

    # Clear all documents
    python scripts/ingest_documents.py clear

Note: PDF files are automatically processed to extract text content and metadata
      (title, author, subject, creator, number of pages).
"""
import sys
import os
import json
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag import rag_system
from app.pdf_processor import pdf_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentManager:
    """Manages document ingestion and updates in the vector store using SQLite."""

    def __init__(self):
        self.rag_system = rag_system
        self.db_path = Path(rag_system.settings.vector_store_path) / "documents.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_from_json()

    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    content_length INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _migrate_from_json(self):
        """Migrate existing documents_metadata.json to SQLite if present."""
        json_path = Path(self.rag_system.settings.vector_store_path) / "documents_metadata.json"
        if not json_path.exists():
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                old_metadata = json.load(f)

            if old_metadata:
                with sqlite3.connect(self.db_path) as conn:
                    for doc_id, info in old_metadata.items():
                        conn.execute(
                            "INSERT OR IGNORE INTO documents (id, content, metadata, content_length) VALUES (?, ?, ?, ?)",
                            (doc_id, "", json.dumps(info.get('metadata', {})), info.get('content_length', 0))
                        )
                    conn.commit()
                logger.info(f"Migrated {len(old_metadata)} documents from JSON to SQLite")

            # Rename old file to mark it as migrated
            json_path.rename(json_path.with_suffix('.json.bak'))
        except Exception as e:
            logger.warning(f"Failed to migrate from JSON: {e}")

    def _read_file_content(self, file_path: Path) -> tuple[str, Dict]:
        """Read content from a file, supporting both text and PDF formats."""
        file_extension = file_path.suffix.lower()

        if file_extension == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            content, pdf_metadata = pdf_processor.extract_text_from_pdf(pdf_bytes)
            return content, pdf_metadata
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content, {}

    def _doc_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the database."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return row is not None

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """Add a new document to the vector store."""
        if self._doc_exists(doc_id):
            logger.warning(f"Document '{doc_id}' already exists. Use 'update' to modify it.")
            return False

        doc_metadata = metadata or {}
        doc_metadata['doc_id'] = doc_id

        logger.info(f"Adding document '{doc_id}' to vector store...")
        self.rag_system.add_documents([content], [doc_metadata])

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO documents (id, content, metadata, content_length) VALUES (?, ?, ?, ?)",
                (doc_id, content, json.dumps(doc_metadata), len(content))
            )
            conn.commit()

        logger.info(f"✓ Document '{doc_id}' added successfully")
        return True

    def update_document(self, doc_id: str, content: str, metadata: Optional[Dict] = None):
        """Update an existing document by deleting and re-adding it."""
        if not self._doc_exists(doc_id):
            logger.error(f"Document '{doc_id}' not found. Use 'add' to create it.")
            return False

        logger.info(f"Updating document '{doc_id}'...")

        # Get existing metadata as fallback
        if metadata is None:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT metadata FROM documents WHERE id = ?", (doc_id,)).fetchone()
                metadata = json.loads(row[0]) if row else {}

        self.delete_document(doc_id, silent=True)
        return self.add_document(doc_id, content, metadata)

    def delete_document(self, doc_id: str, silent: bool = False):
        """Delete a document from the vector store."""
        if not self._doc_exists(doc_id):
            if not silent:
                logger.error(f"Document '{doc_id}' not found")
            return False

        logger.info(f"Deleting document '{doc_id}'...")
        logger.warning("FAISS doesn't support direct deletion. Rebuilding vector store...")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()

        self._rebuild_vector_store()

        if not silent:
            logger.info(f"✓ Document '{doc_id}' deleted successfully")
        return True

    def _rebuild_vector_store(self):
        """Rebuild the entire vector store from stored content."""
        logger.info("Rebuilding vector store...")

        self.rag_system.vector_store = self.rag_system.create_empty_vector_store()

        # Re-add all remaining documents from SQLite
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, content, metadata FROM documents").fetchall()

        if rows:
            texts = [row[1] for row in rows]
            metadatas = [json.loads(row[2]) for row in rows]
            self.rag_system.add_documents(texts, metadatas)
            logger.info(f"✓ Rebuilt vector store with {len(rows)} documents")
        else:
            # Save empty vector store
            vector_store_path = Path(self.rag_system.settings.vector_store_path)
            self.rag_system.vector_store.save_local(str(vector_store_path))
            logger.info("✓ Vector store cleared (no documents remaining)")

    def list_documents(self):
        """List all documents in the vector store."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, content_length, metadata, content FROM documents").fetchall()

        if not rows:
            logger.info("No documents in vector store")
            return

        logger.info(f"\n{'='*80}")
        logger.info(f"Documents in vector store: {len(rows)}")
        logger.info(f"{'='*80}\n")

        for doc_id, content_length, metadata_json, content in rows:
            preview = content[:200] + '...' if len(content) > 200 else content
            logger.info(f"ID: {doc_id}")
            logger.info(f"  Length: {content_length} characters")
            logger.info(f"  Metadata: {metadata_json}")
            logger.info(f"  Preview: {preview}")
            logger.info("")

    def clear_all(self):
        """Clear all documents from the vector store."""
        logger.warning("This will delete ALL documents from the vector store!")
        response = input("Are you sure? (yes/no): ")

        if response.lower() != 'yes':
            logger.info("Cancelled")
            return

        logger.info("Clearing vector store...")

        self.rag_system.vector_store = self.rag_system.create_empty_vector_store()

        vector_store_path= Path(self.rag_system.settings.vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        self.rag_system.vector_store.save_local(str(vector_store_path))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents")
            conn.commit()

        logger.info("✓ Vector store cleared")

    def add_from_directory(self, directory: Path, pattern: str = "*.txt"):
        """Add all documents from a directory. Supports .txt and .pdf files."""
        files = list(directory.glob(pattern))

        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory}")
            return

        logger.info(f"Found {len(files)} files to ingest")

        for file_path in files:
            doc_id = file_path.stem  # Use filename without extension as ID

            try:
                # Read file content using helper method (supports PDF and text)
                content, extracted_metadata = self._read_file_content(file_path)

                # Combine extracted metadata with source information
                metadata = {
                    'source': str(file_path),
                    'filename': file_path.name,
                    **extracted_metadata
                }

                self.add_document(doc_id, content, metadata)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Manage documents in the vector store',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new document (supports .txt and .pdf files)')
    add_parser.add_argument('--file', type=str, help='Path to document file (.txt or .pdf)')
    add_parser.add_argument('--dir', type=str, help='Directory containing documents')
    add_parser.add_argument('--pattern', type=str, default='*.txt', help='File pattern for directory (default: *.txt, use "*.pdf" for PDFs or "*.*" for all)')
    add_parser.add_argument('--id', type=str, help='Document ID (required for --file)')
    add_parser.add_argument('--metadata', type=str, help='JSON metadata (merged with auto-extracted PDF metadata)')

    # Update command
    update_parser = subparsers.add_parser('update', help='Update an existing document')
    update_parser.add_argument('--id', type=str, required=True, help='Document ID')
    update_parser.add_argument('--file', type=str, required=True, help='Path to new document file')
    update_parser.add_argument('--metadata', type=str, help='JSON metadata')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a document')
    delete_parser.add_argument('--id', type=str, required=True, help='Document ID')

    # List command
    subparsers.add_parser('list', help='List all documents')

    # Clear command
    subparsers.add_parser('clear', help='Clear all documents')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag_system.initialize()

    # Create document manager
    manager = DocumentManager()

    # Execute command
    if args.command == 'add':
        if args.dir:
            dir_path = Path(args.dir)
            if not dir_path.exists():
                logger.error(f"Directory not found: {args.dir}")
                return
            manager.add_from_directory(dir_path, args.pattern)
        elif args.file:
            if not args.id:
                logger.error("--id is required when using --file")
                return

            file_path = Path(args.file)
            if not file_path.exists():
                logger.error(f"File not found: {args.file}")
                return

            # Read file content using helper method (supports PDF and text)
            content, extracted_metadata = manager._read_file_content(file_path)

            # Merge user-provided metadata with extracted metadata
            base_metadata = {'source': str(file_path), 'filename': file_path.name, **extracted_metadata}
            if args.metadata:
                user_metadata = json.loads(args.metadata)
                base_metadata.update(user_metadata)

            manager.add_document(args.id, content, base_metadata)
        else:
            logger.error("Either --file or --dir must be specified")

    elif args.command == 'update':
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {args.file}")
            return

        # Read file content using helper method (supports PDF and text)
        content, extracted_metadata = manager._read_file_content(file_path)

        # Merge extracted metadata with user-provided metadata
        metadata = extracted_metadata.copy()
        if args.metadata:
            user_metadata = json.loads(args.metadata)
            metadata.update(user_metadata)

        manager.update_document(args.id, content, metadata if metadata else None)

    elif args.command == 'delete':
        manager.delete_document(args.id)

    elif args.command == 'list':
        manager.list_documents()

    elif args.command == 'clear':
        manager.clear_all()


if __name__ == "__main__":
    main()
