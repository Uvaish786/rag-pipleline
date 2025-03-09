# FastAPI Qdrant Document Processing

This project is a FastAPI application that processes PDF documents, stores them in a Qdrant vector store, and allows querying the documents using natural language.

## Features

- Upload multiple PDF documents.
- Process and store documents in a Qdrant vector store.
- Query documents using natural language.
- Logging and exception handling.

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the FastAPI application: `python -m app.main`.
4. Access the API at `http://localhost:8000`.

## API Endpoints

- **POST /upload/**: Upload PDF documents.
- **POST /query/**: Query the document collection.

## Logs

Logs are stored in the `logs/app.log` file.