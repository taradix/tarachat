"""Document ingestion for managing the vector store.

Supports both text (.txt) and PDF (.pdf) files.
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from tarachat.pdf_processor import PDFProcessor, pdf_processor
from tarachat.rag import RAGSystem, rag_system

logger = logging.getLogger(__name__)


class DocumentManager:
    """Manages document ingestion and updates in the vector store using SQLite."""

    def __init__(self, rag: RAGSystem, pdf: PDFProcessor):
        self.rag = rag
        self.pdf = pdf
        self.db_path = Path(rag.settings.vector_store_path) / "documents.db"
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
        json_path = Path(self.rag.settings.vector_store_path) / "documents_metadata.json"
        if not json_path.exists():
            return

        try:
            with open(json_path, encoding='utf-8') as f:
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

            json_path.rename(json_path.with_suffix('.json.bak'))
        except Exception as e:
            logger.warning(f"Failed to migrate from JSON: {e}")

    def _read_file_content(self, file_path: Path) -> tuple[str, dict]:
        """Read content from a file, supporting both text and PDF formats."""
        if file_path.suffix.lower() == '.pdf':
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            return self.pdf.extract_text_from_pdf(pdf_bytes)
        else:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
            return content, {}

    def _doc_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the database."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return row is not None

    def add_document(self, doc_id: str, content: str, metadata: dict | None = None):
        """Add a new document to the vector store."""
        if self._doc_exists(doc_id):
            logger.warning(f"Document '{doc_id}' already exists. Use 'update' to modify it.")
            return False

        doc_metadata = metadata or {}
        doc_metadata['doc_id'] = doc_id

        logger.info(f"Adding document '{doc_id}' to vector store...")
        self.rag.add_documents([content], [doc_metadata])

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO documents (id, content, metadata, content_length) VALUES (?, ?, ?, ?)",
                (doc_id, content, json.dumps(doc_metadata), len(content))
            )
            conn.commit()

        logger.info(f"✓ Document '{doc_id}' added successfully")
        return True

    def update_document(self, doc_id: str, content: str, metadata: dict | None = None):
        """Update an existing document by deleting and re-adding it."""
        if not self._doc_exists(doc_id):
            logger.error(f"Document '{doc_id}' not found. Use 'add' to create it.")
            return False

        logger.info(f"Updating document '{doc_id}'...")

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

        self.rag.vector_store = self.rag.create_empty_vector_store()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT id, content, metadata FROM documents").fetchall()

        if rows:
            texts = [row[1] for row in rows]
            metadatas = [json.loads(row[2]) for row in rows]
            self.rag.add_documents(texts, metadatas)
            logger.info(f"✓ Rebuilt vector store with {len(rows)} documents")
        else:
            vector_store_path = Path(self.rag.settings.vector_store_path)
            self.rag.vector_store.save_local(str(vector_store_path))
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

        self.rag.vector_store = self.rag.create_empty_vector_store()

        vector_store_path = Path(self.rag.settings.vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        self.rag.vector_store.save_local(str(vector_store_path))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM documents")
            conn.commit()

        logger.info("✓ Vector store cleared")

    def init_from_sample_file(self, data_path: Path):
        """Load sample documents split by blank lines."""
        if not data_path.exists():
            logger.warning(f"Sample documents not found at {data_path}")
            logger.info("Vector store initialized but empty.")
            return

        logger.info(f"Loading documents from {data_path}...")
        with open(data_path, encoding="utf-8") as f:
            content = f.read()

        documents = [doc.strip() for doc in content.split("\n\n") if doc.strip()]
        if not documents:
            logger.info("No documents found in sample file.")
            return

        logger.info(f"Adding {len(documents)} documents to vector store...")
        self.rag.add_documents(documents)

        logger.info(f"✓ Loaded {len(documents)} sample documents")

    def add_from_directory(self, directory: Path, pattern: str = "*.txt"):
        """Add all documents from a directory. Supports .txt and .pdf files."""
        files = list(directory.glob(pattern))

        if not files:
            logger.warning(f"No files matching '{pattern}' found in {directory}")
            return

        logger.info(f"Found {len(files)} files to ingest")

        for file_path in files:
            doc_id = file_path.stem

            try:
                content, extracted_metadata = self._read_file_content(file_path)
                metadata = {
                    'source': str(file_path),
                    'filename': file_path.name,
                    **extracted_metadata
                }
                self.add_document(doc_id, content, metadata)
            except Exception:
                logger.exception(f"Error processing {file_path}")


def _run_add(manager, args):
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

        content, extracted_metadata = manager._read_file_content(file_path)
        base_metadata = {'source': str(file_path), 'filename': file_path.name, **extracted_metadata}
        if args.metadata:
            user_metadata = json.loads(args.metadata)
            base_metadata.update(user_metadata)
        manager.add_document(args.id, content, base_metadata)
    else:
        logger.error("Either --file or --dir must be specified")


def _run_update(manager, args):
    file_path = Path(args.file)
    if not file_path.exists():
        logger.error(f"File not found: {args.file}")
        return

    content, extracted_metadata = manager._read_file_content(file_path)
    metadata = extracted_metadata.copy()
    if args.metadata:
        user_metadata = json.loads(args.metadata)
        metadata.update(user_metadata)
    manager.update_document(args.id, content, metadata if metadata else None)


def main():
    parser = argparse.ArgumentParser(
        description='Manage documents in the vector store',
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    add_parser = subparsers.add_parser('add', help='Add a new document')
    add_parser.add_argument('--file', type=str, help='Path to document file')
    add_parser.add_argument('--dir', type=str, help='Directory containing documents')
    add_parser.add_argument('--pattern', type=str, default='*.txt', help='File pattern for directory')
    add_parser.add_argument('--id', type=str, help='Document ID (required for --file)')
    add_parser.add_argument('--metadata', type=str, help='JSON metadata')

    update_parser = subparsers.add_parser('update', help='Update an existing document')
    update_parser.add_argument('--id', type=str, required=True, help='Document ID')
    update_parser.add_argument('--file', type=str, required=True, help='Path to new document file')
    update_parser.add_argument('--metadata', type=str, help='JSON metadata')

    delete_parser = subparsers.add_parser('delete', help='Delete a document')
    delete_parser.add_argument('--id', type=str, required=True, help='Document ID')

    subparsers.add_parser('list', help='List all documents')
    subparsers.add_parser('clear', help='Clear all documents')

    init_parser = subparsers.add_parser('init', help='Load sample documents')
    init_parser.add_argument(
        '--data-path', type=str,
        default=str(Path(__file__).parent.parent / "data" / "sample_documents.txt"),
        help='Path to sample documents file',
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO)

    logger.info("Initializing RAG system...")
    rag_system.initialize()

    manager = DocumentManager(rag_system, pdf_processor)

    commands = {
        'add': lambda: _run_add(manager, args),
        'update': lambda: _run_update(manager, args),
        'delete': lambda: manager.delete_document(args.id),
        'list': manager.list_documents,
        'clear': manager.clear_all,
        'init': lambda: manager.init_from_sample_file(Path(args.data_path)),
    }
    commands[args.command]()
