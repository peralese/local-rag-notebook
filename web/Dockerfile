FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV EMBED_BACKEND=fastembed
ENV EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV VECTOR_DIR=/data/chroma
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
