FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    httpx

# Copy application files
COPY src/serve_fastapi.py .
COPY demo/demo_app.py .

# Expose the docs server port
EXPOSE 8000

# Default command
CMD ["python", "serve_fastapi.py", \
     "--input", "demo_app.py", \
     "--model", "swaggerlm", \
     "--port", "8000", \
     "--real-server", "http://host.docker.internal:8001"]
