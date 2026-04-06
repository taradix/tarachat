# Troubleshooting Guide

## Chat Interface Issues

### Problem: "Connecting to server..." stuck

**Symptoms**: Frontend shows "Connecting to server..." indefinitely

**Solution**:
```bash
# Check if backend is running
docker compose ps

# Check backend health
curl http://localhost:8000/health

# Check logs
docker compose logs backend --tail 50

# Restart if needed
docker compose restart backend
```

### Problem: Chat never responds / timeouts

**Symptoms**: Chat interface loading indefinitely, no response after 1-2 minutes

**Cause**: CPU-based LLM inference is too slow (3-5+ minutes per response)

**Quick Fix - Enable Demo Mode** (already enabled by default):

Edit `backend/app/config.py`:
```python
demo_mode: bool = True  # Fast RAG-only responses
```

Restart:
```bash
docker compose restart backend
```

**Permanent Solution Options**:
1. **Use GPU** - See [PERFORMANCE_ISSUES.md](PERFORMANCE_ISSUES.md) Option 1
2. **Use smaller model** - See [PERFORMANCE_ISSUES.md](PERFORMANCE_ISSUES.md) Option 2
3. **Use external API** - See [PERFORMANCE_ISSUES.md](PERFORMANCE_ISSUES.md) Option 4

### Problem: "FAISS deserialization" error

**Symptoms**: Backend logs show error about `allow_dangerous_deserialization`

**Status**: ✅ Already fixed in `backend/app/rag.py:46-59`

If you still see this error:
```bash
# Clear vector store and rebuild
docker compose down
rm -rf backend/vector_store
docker compose up -d
```

## Document Ingestion Issues

### Problem: Documents not found in responses

**Symptoms**: Chat returns "Initial document" or "no information found"

**Solution**:
```bash
# 1. Ingest documents
make ingest-docs

# 2. Restart backend to reload vector store
docker compose restart backend

# 3. Test
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"test query"}'
```

### Problem: Ingestion script fails

**Symptoms**: `ingest_documents.py` errors

**Solution**:
```bash
# Run inside container
docker compose exec backend python scripts/ingest_documents.py add --dir data/documents/

# Check if documents exist
docker compose exec backend ls -la data/documents/

# Check logs
docker compose logs backend
```

## Docker Issues

### Problem: Containers won't start

**Solution**:
```bash
# Check what's running
docker compose ps

# View logs
docker compose logs

# Clean restart
docker compose down
docker compose up -d

# Full reset
docker compose down -v
rm -rf backend/vector_store
docker compose up --build -d
```

### Problem: Out of memory

**Symptoms**: Container killed, OOM errors

**Solution**:
```bash
# Check Docker memory limit
docker stats

# Increase Docker memory (Docker Desktop settings)
# Or use smaller model:
```

Edit `backend/app/config.py`:
```python
model_name: str = "google/flan-t5-small"  # Much smaller
```

### Problem: Disk space issues

**Solution**:
```bash
# Check disk space
df -h

# Clean Docker
docker system prune -a

# Remove old images
docker image prune -a
```

## Frontend Issues

### Problem: Frontend can't connect to backend

**Symptoms**: Network errors, CORS errors

**Solution**:
```bash
# Check environment variable
docker compose exec frontend env | grep VITE_API_URL
# Should be: VITE_API_URL=http://localhost:8000

# Restart frontend
docker compose restart frontend

# Check if backend is accessible
curl http://localhost:8000/health
```

### Problem: Frontend shows old data

**Solution**:
```bash
# Clear browser cache
# Or hard refresh (Ctrl+Shift+R / Cmd+Shift+R)

# Restart frontend
docker compose restart frontend
```

## Performance Issues

### Problem: Everything is slow

**Check system resources**:
```bash
# CPU and memory usage
docker stats

# System resources
top
```

**Solutions**:
1. Enable demo mode (default)
2. Close other applications
3. Increase Docker resources
4. Use GPU for LLM
5. Use external API

See [PERFORMANCE_ISSUES.md](PERFORMANCE_ISSUES.md) for detailed solutions.

### Problem: Model download takes forever

**Symptoms**: Backend stuck at "Loading language model"

**Solution**:
```bash
# Be patient - first download takes 10-15 minutes for 3GB model
docker compose logs -f backend

# Check download progress
docker compose exec backend du -sh /root/.cache/huggingface/hub/

# If stuck for >20 minutes, restart
docker compose restart backend
```

## Common Commands

### Check Status
```bash
# All services
docker compose ps

# Backend health
curl http://localhost:8000/health

# Document count
curl http://localhost:8000/documents/count

# List documents
make list-docs
```

### View Logs
```bash
# All logs
docker compose logs

# Backend only
docker compose logs backend

# Follow logs
docker compose logs -f

# Last 50 lines
docker compose logs --tail 50
```

### Restart Services
```bash
# Restart all
docker compose restart

# Restart backend only
docker compose restart backend

# Stop and start
docker compose down
docker compose up -d
```

### Clean Start
```bash
# Soft clean (keeps data)
docker compose down
docker compose up -d

# Hard clean (removes data)
docker compose down -v
rm -rf backend/vector_store
docker compose up -d

# Full rebuild
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

## Getting Help

### Collect Diagnostic Information

```bash
# System info
docker --version
docker compose --version

# Service status
docker compose ps

# Recent logs
docker compose logs --tail 100 > logs.txt

# Vector store status
ls -lah backend/vector_store/

# Container resource usage
docker stats --no-stream
```

### Report an Issue

Include:
1. Symptoms (what's not working)
2. Error messages from logs
3. Output of `docker compose ps`
4. System info (OS, Docker version)
5. Configuration changes made

## Quick Fixes

### "Nothing works"
```bash
docker compose down -v && docker compose up --build -d
```

### "Chat is slow"
Set `demo_mode: bool = True` in `backend/app/config.py`

### "No documents found"
```bash
make ingest-docs && docker compose restart backend
```

### "Backend won't start"
```bash
docker compose logs backend --tail 50
# Check for specific error and search this guide
```

## Still Having Issues?

1. Check [SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md) for known fixes
2. Review [PERFORMANCE_ISSUES.md](PERFORMANCE_ISSUES.md) for performance solutions
3. Check [GETTING_STARTED.md](GETTING_STARTED.md) for setup instructions
4. Ensure all prerequisites are met (Docker, disk space, memory)
