# Document Ingestion Guide

This guide explains how to populate and manage documents in the TaraChat vector database using the ingestion script instead of the web interface.

## Overview

The `ingest_documents.py` script provides a command-line interface for managing documents in the FAISS vector store. It supports:

- Adding individual documents or entire directories
- Updating existing documents
- Deleting documents
- Listing all documents
- Clearing the vector store

## Prerequisites

Make sure the backend environment is set up:

```bash
cd backend
poetry install
```

## Basic Usage

### 1. Add Documents from a Directory

Add all `.txt` files from a directory:

```bash
poetry run python scripts/ingest_documents.py add --dir data/documents/
```

Add files with a different pattern:

```bash
poetry run python scripts/ingest_documents.py add --dir data/documents/ --pattern "*.md"
```

### 2. Add a Single Document

Add a document with a specific ID:

```bash
poetry run python scripts/ingest_documents.py add --file data/documents/paris.txt --id paris
```

Add with custom metadata:

```bash
poetry run python scripts/ingest_documents.py add \
  --file data/documents/paris.txt \
  --id paris \
  --metadata '{"author": "Tourism Board", "category": "geography", "language": "fr"}'
```

### 3. Update an Existing Document

Update a document's content:

```bash
poetry run python scripts/ingest_documents.py update --id paris --file data/documents/paris_updated.txt
```

Update with new metadata:

```bash
poetry run python scripts/ingest_documents.py update \
  --id paris \
  --file data/documents/paris_updated.txt \
  --metadata '{"author": "Updated Author", "version": "2.0"}'
```

### 4. List All Documents

View all documents in the vector store:

```bash
poetry run python scripts/ingest_documents.py list
```

This shows:
- Document IDs
- Content length
- Metadata
- Content preview

### 5. Delete a Document

Delete a specific document:

```bash
poetry run python scripts/ingest_documents.py delete --id paris
```

**Note:** FAISS doesn't support direct deletion, so this rebuilds the vector store without the deleted document.

### 6. Clear All Documents

Remove all documents from the vector store:

```bash
poetry run python scripts/ingest_documents.py clear
```

You'll be prompted to confirm before deletion.

## Document Organization

### Recommended Directory Structure

```
backend/data/
├── documents/              # Your document collection
│   ├── geography/
│   │   ├── paris.txt
│   │   └── france.txt
│   ├── culture/
│   │   ├── french_cuisine.txt
│   │   └── literature.txt
│   └── history/
│       ├── louvre.txt
│       └── french_revolution.txt
└── sample_documents.txt    # Original sample file
```

### Document ID Strategy

Choose meaningful document IDs that help you identify and update documents:

- Use filename stems: `paris.txt` → ID: `paris`
- Use hierarchical IDs: `geography_paris`, `culture_cuisine`
- Use version numbers: `paris_v1`, `paris_v2`

## Metadata

Metadata helps organize and filter documents. Common metadata fields:

```json
{
  "author": "Author Name",
  "category": "geography",
  "language": "fr",
  "date": "2024-01-15",
  "version": "1.0",
  "source": "path/to/file.txt",
  "tags": ["paris", "tourism", "france"]
}
```

The script automatically adds:
- `doc_id`: The document identifier
- `source`: File path (when using `--file`)
- `filename`: Original filename

## Workflow Examples

### Initial Population

1. Organize your documents in the `data/documents/` directory
2. Ingest all documents:

```bash
poetry run python scripts/ingest_documents.py add --dir data/documents/
```

3. Verify ingestion:

```bash
poetry run python scripts/ingest_documents.py list
```

### Updating Content

When you need to update a document:

1. Edit the source file: `data/documents/paris.txt`
2. Update in vector store:

```bash
poetry run python scripts/ingest_documents.py update --id paris --file data/documents/paris.txt
```

### Managing Versions

Keep different versions of documents:

```bash
# Add version 1
poetry run python scripts/ingest_documents.py add \
  --file docs/guide_v1.txt \
  --id guide_v1 \
  --metadata '{"version": "1.0"}'

# Add version 2
poetry run python scripts/ingest_documents.py add \
  --file docs/guide_v2.txt \
  --id guide_v2 \
  --metadata '{"version": "2.0"}'

# Remove old version
poetry run python scripts/ingest_documents.py delete --id guide_v1
```

## Running in Docker

To use the ingestion script inside the Docker container:

```bash
# Enter the backend container
docker-compose exec backend bash

# Run ingestion commands
python scripts/ingest_documents.py add --dir data/documents/
python scripts/ingest_documents.py list
```

Or run directly:

```bash
docker-compose exec backend python scripts/ingest_documents.py list
```

## Automation with Makefile

Add these commands to your `Makefile` for convenience:

```makefile
.PHONY: ingest-docs list-docs

ingest-docs:
	cd backend && poetry run python scripts/ingest_documents.py add --dir data/documents/

list-docs:
	cd backend && poetry run python scripts/ingest_documents.py list

update-doc:
	@echo "Usage: make update-doc ID=doc_id FILE=path/to/file.txt"
	cd backend && poetry run python scripts/ingest_documents.py update --id $(ID) --file $(FILE)
```

Then use:

```bash
make ingest-docs
make list-docs
make update-doc ID=paris FILE=data/documents/paris.txt
```

## Important Notes

### FAISS Limitations

- **No native deletion**: FAISS doesn't support deleting individual vectors. The script handles this by tracking metadata, but rebuilding requires re-ingesting documents.
- **No native updates**: Updates are handled by deleting and re-adding documents.
- **File-based storage**: The vector store is saved as files (`index.faiss`, `index.pkl`), not a database.

### Metadata Storage

Document metadata is stored in `vector_store/documents_metadata.json`. This file tracks:
- Document IDs
- Custom metadata
- Content previews
- Content lengths

**Important:** Back up this file along with the vector store files.

### Production Considerations

For production use with frequent updates, consider:

1. **Alternative vector stores**: Pinecone, Weaviate, or Qdrant support native deletion and updates
2. **Content backup**: Store original documents separately for rebuilding
3. **Version control**: Keep documents in Git for change tracking
4. **Automated ingestion**: Set up CI/CD pipelines to auto-ingest on document changes

## Troubleshooting

### "Document already exists" error

Use `update` instead of `add`:

```bash
poetry run python scripts/ingest_documents.py update --id mydoc --file mydoc.txt
```

### "Document not found" error

List documents to check the ID:

```bash
poetry run python scripts/ingest_documents.py list
```

### Vector store gets corrupted

Clear and re-ingest:

```bash
poetry run python scripts/ingest_documents.py clear
poetry run python scripts/ingest_documents.py add --dir data/documents/
```

### Memory issues with large documents

Adjust chunk size in `backend/app/config.py`:

```python
chunk_size: int = 256  # Reduce from 512
chunk_overlap: int = 25  # Reduce from 50
```

## Example Complete Workflow

```bash
# 1. Prepare your documents
mkdir -p backend/data/documents
cp your_docs/*.txt backend/data/documents/

# 2. Enter backend directory
cd backend

# 3. Ingest all documents
poetry run python scripts/ingest_documents.py add --dir data/documents/

# 4. Verify ingestion
poetry run python scripts/ingest_documents.py list

# 5. Update a specific document
echo "Updated content" > data/documents/paris.txt
poetry run python scripts/ingest_documents.py update --id paris --file data/documents/paris.txt

# 6. Delete an outdated document
poetry run python scripts/ingest_documents.py delete --id old_doc

# 7. Start the application
cd ..
docker-compose up -d
```

## Next Steps

- See [USAGE.md](USAGE.md) for how to interact with the chatbot
- See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for technical details
- See [PDF_UPLOAD.md](PDF_UPLOAD.md) for PDF document handling
