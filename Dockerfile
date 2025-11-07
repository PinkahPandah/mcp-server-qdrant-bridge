FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy local source code
COPY . /app

# Install from local source (not PyPI)
RUN uv pip install --system --no-cache-dir .

# Expose the default port for SSE transport (not used for STDIO but kept for compatibility)
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
ENV COLLECTION_NAME="default-collection"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Run the already-installed package (STDIO mode by default)
# MCPO will communicate via stdin/stdout
CMD ["mcp-server-qdrant"]
