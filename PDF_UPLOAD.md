# PDF Upload Feature

TaraChat now supports uploading PDF documents to enhance the chatbot's knowledge base.

## Features

- **PDF Text Extraction**: Automatically extracts text from all pages
- **Metadata Extraction**: Captures title, author, subject, and creator from PDF metadata
- **Page Markers**: Text is organized with page numbers for better context
- **Validation**: Ensures only valid PDF files are processed
- **Error Handling**: Clear error messages for invalid or unreadable PDFs

## How to Upload a PDF

### Via Web Interface

1. Click the **"Upload Document"** button in the header
2. Select the **"PDF File"** tab
3. Click **"Choose File"** and select your PDF
4. Click **"Upload"**
5. Wait for the success message

The PDF will be automatically:
- Validated
- Text extracted from all pages
- Split into chunks
- Embedded and stored in the vector database

### Via API

```bash
curl -X POST http://localhost:8000/documents/upload-pdf \
  -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "message": "PDF uploaded and processed successfully",
  "filename": "document.pdf",
  "metadata": {
    "num_pages": 10,
    "file_type": "pdf",
    "title": "Document Title",
    "author": "Author Name",
    "filename": "document.pdf"
  },
  "text_length": 15432
}
```

## API Endpoint

**POST** `/documents/upload-pdf`

- **Content-Type**: `multipart/form-data`
- **Parameter**: `file` (PDF file)
- **Returns**: Upload status and metadata

### Error Responses

**400 Bad Request**
```json
{
  "detail": "Only PDF files are supported"
}
```

**400 Bad Request**
```json
{
  "detail": "Invalid PDF file"
}
```

**400 Bad Request**
```json
{
  "detail": "No text content could be extracted from the PDF"
}
```

**503 Service Unavailable**
```json
{
  "detail": "RAG system is not ready yet. Please try again later."
}
```

## Technical Details

### Backend Implementation

**PDF Processing** (`backend/app/pdf_processor.py`):
- Uses PyPDF2 for PDF parsing
- Extracts text page by page
- Captures PDF metadata (title, author, etc.)
- Validates PDF structure

**API Endpoint** (`backend/app/main.py`):
- File upload handling with FastAPI
- Validation of file type and content
- Integration with RAG system
- Error handling and logging

### Frontend Implementation

**Upload Component** (`frontend/src/components/DocumentUpload.tsx`):
- Tab-based interface (Text / PDF File)
- File input with PDF validation
- Progress indicators
- Success/error feedback

**API Client** (`frontend/src/api.ts`):
- `uploadPDF(file: File)` function
- FormData for file upload
- Type-safe response handling

## Supported PDF Features

✅ **Supported:**
- Standard PDF format (PDF 1.0 - 1.7)
- Text-based PDFs
- Multi-page documents
- PDF metadata
- Encrypted PDFs (if no password required)

❌ **Not Supported:**
- Password-protected PDFs
- Scanned images (OCR not included)
- Complex layouts may have text extraction issues
- Embedded fonts with special encodings

## Best Practices

1. **Use Text-Based PDFs**: Ensure your PDF contains selectable text, not just images
2. **File Size**: Keep PDFs under 10MB for optimal processing
3. **Page Count**: Large PDFs (100+ pages) may take longer to process
4. **Format**: Use standard PDF formats for best results
5. **Content**: Clear, well-formatted text extracts better than complex layouts

## Troubleshooting

### "No text content could be extracted from the PDF"

This usually means:
- The PDF contains only images (scanned document)
- Text is embedded in a non-standard way
- The PDF is corrupted

**Solution**: Try converting the PDF to text first, or use OCR software if it's a scanned document.

### "Invalid PDF file"

This means:
- The file is not a valid PDF
- The PDF structure is corrupted
- The file extension is .pdf but the content is not

**Solution**: Open the PDF in a PDF reader to verify it's valid, or try re-saving it.

### Upload is very slow

For large PDFs:
- Processing time increases with page count
- Check backend logs for progress
- Be patient - the system is extracting and processing all pages

## Example Usage

### Python Script
```python
import requests

# Upload a PDF
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/documents/upload-pdf',
        files=files
    )
    print(response.json())
```

### JavaScript/TypeScript
```typescript
const uploadPDF = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/documents/upload-pdf', {
    method: 'POST',
    body: formData,
  });

  return await response.json();
};
```

### cURL
```bash
# Upload a PDF
curl -X POST http://localhost:8000/documents/upload-pdf \
  -F "file=@document.pdf"

# With verbose output
curl -v -X POST http://localhost:8000/documents/upload-pdf \
  -F "file=@document.pdf"
```

## Processing Flow

1. **Upload**: User selects PDF file
2. **Validation**: File type and PDF structure validated
3. **Extraction**: Text extracted from all pages with PyPDF2
4. **Metadata**: PDF metadata captured (title, author, etc.)
5. **Chunking**: Text split into manageable chunks
6. **Embedding**: Each chunk converted to vector embedding
7. **Storage**: Embeddings stored in FAISS vector database
8. **Ready**: Document available for RAG queries

## Security Considerations

- File size limits should be enforced in production
- PDF validation prevents malicious file uploads
- No file is permanently stored on disk (processed in memory)
- Only text content is extracted and stored
- Metadata is sanitized before storage

## Future Enhancements

Potential improvements:
- OCR support for scanned PDFs
- Multiple file upload at once
- Progress bar for large PDFs
- PDF preview before upload
- Support for other formats (DOCX, TXT, HTML)
