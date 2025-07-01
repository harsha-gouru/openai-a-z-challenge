#!/usr/bin/env python
"""
Amazon Deep Insights - RAG API Module

This module provides a FastAPI-based API for the RAG (Retrieval-Augmented Generation) system.
It includes endpoints for:
- Querying the knowledge base with natural language questions
- Retrieving relevant context for a query
- Generating answers based on retrieved context
- Health check and system information
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException, Query, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import chromadb
from dotenv import load_dotenv

# Add parent directory to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import local modules
from src.rag.embeddings import (
    create_chroma_client,
    get_openai_embedding_function,
    get_or_create_collection,
    query_collection,
    EmbeddingError
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amazon_insights.rag.api")

# Default settings
DEFAULT_VECTOR_DB_DIR = os.environ.get("CHROMA_DB_DIR", "data/vector_db")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 1024))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.2))
DEFAULT_TOP_K_RESULTS = int(os.environ.get("TOP_K_RESULTS", 5))
DEFAULT_COLLECTION_NAME = "amazon_insights"

# Create FastAPI app
app = FastAPI(
    title="Amazon Deep Insights RAG API",
    description="API for querying Amazon rainforest data using RAG",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    collection_name: str = Field(DEFAULT_COLLECTION_NAME, description="Vector store collection name")
    top_k: int = Field(DEFAULT_TOP_K_RESULTS, description="Number of results to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_context: bool = Field(True, description="Include retrieved context in response")

class GenerateRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    collection_name: str = Field(DEFAULT_COLLECTION_NAME, description="Vector store collection name")
    top_k: int = Field(DEFAULT_TOP_K_RESULTS, description="Number of results to retrieve")
    model: str = Field(DEFAULT_COMPLETION_MODEL, description="OpenAI model to use")
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, description="Maximum tokens in response")
    temperature: float = Field(DEFAULT_TEMPERATURE, description="Temperature for generation")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    include_context: bool = Field(True, description="Include retrieved context in response")

class QueryResult(BaseModel):
    query: str = Field(..., description="Original query")
    contexts: List[Dict[str, Any]] = Field(..., description="Retrieved contexts")
    collection_name: str = Field(..., description="Collection name")

class GenerateResult(BaseModel):
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    contexts: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved contexts")
    collection_name: str = Field(..., description="Collection name")
    model: str = Field(..., description="Model used for generation")
    tokens_used: int = Field(..., description="Tokens used for generation")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    collections: List[str] = Field(..., description="Available collections")
    embedding_model: str = Field(..., description="Embedding model")
    completion_model: str = Field(..., description="Completion model")

# Global variables
_chroma_client = None
_openai_client = None

def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = create_chroma_client(DEFAULT_VECTOR_DB_DIR)
    return _chroma_client

def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        _openai_client = openai.OpenAI(api_key=api_key)
    return _openai_client

def get_collection(collection_name: str = DEFAULT_COLLECTION_NAME):
    """Get ChromaDB collection."""
    client = get_chroma_client()
    embedding_function = get_openai_embedding_function(model=DEFAULT_EMBEDDING_MODEL)
    
    try:
        return get_or_create_collection(client, collection_name, embedding_function)
    except Exception as e:
        logger.error(f"Error getting collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error accessing vector database: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        client = get_chroma_client()
        collections = [col.name for col in client.list_collections()]
        
        return {
            "status": "healthy",
            "version": "0.1.0",
            "collections": collections,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "completion_model": DEFAULT_COMPLETION_MODEL
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResult)
async def query_endpoint(request: QueryRequest):
    """Query the vector database."""
    try:
        # Get collection
        collection = get_collection(request.collection_name)
        
        # Query collection
        results = query_collection(
            collection,
            request.query,
            n_results=request.top_k,
            where=request.filters
        )
        
        # Format results
        contexts = []
        for i, (document, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            contexts.append({
                "text": document,
                "metadata": metadata,
                "score": float(1.0 - distance)  # Convert distance to similarity score
            })
        
        return {
            "query": request.query,
            "contexts": contexts,
            "collection_name": request.collection_name
        }
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying vector database: {str(e)}"
        )

@app.post("/generate", response_model=GenerateResult)
async def generate_endpoint(request: GenerateRequest):
    """Generate answer using RAG."""
    try:
        # Get collection
        collection = get_collection(request.collection_name)
        
        # Query collection
        results = query_collection(
            collection,
            request.query,
            n_results=request.top_k,
            where=request.filters
        )
        
        # Format contexts
        contexts = []
        context_text = ""
        
        for i, (document, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            context = {
                "text": document,
                "metadata": metadata,
                "score": float(1.0 - distance)
            }
            contexts.append(context)
            
            # Add to context text
            context_text += f"\nContext {i+1}:\n{document}\n"
        
        # Create prompt
        prompt = f"""Answer the following question based on the provided context. If the answer cannot be determined from the context, say so.

Question: {request.query}

{context_text}

Answer:"""
        
        # Generate answer
        openai_client = get_openai_client()
        response = openai_client.chat.completions.create(
            model=request.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Extract answer
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        return {
            "query": request.query,
            "answer": answer,
            "contexts": contexts if request.include_context else None,
            "collection_name": request.collection_name,
            "model": request.model,
            "tokens_used": tokens_used
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}"
        )

@app.post("/search")
async def search_endpoint(
    query: str = Query(..., description="Search query"),
    collection_name: str = Query(DEFAULT_COLLECTION_NAME, description="Collection name"),
    top_k: int = Query(DEFAULT_TOP_K_RESULTS, description="Number of results")
):
    """Simple search endpoint with query parameters."""
    try:
        # Get collection
        collection = get_collection(collection_name)
        
        # Query collection
        results = query_collection(
            collection,
            query,
            n_results=top_k
        )
        
        # Format results
        search_results = []
        for i, (document, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            search_results.append({
                "text": document,
                "metadata": metadata,
                "score": float(1.0 - distance),
                "rank": i + 1
            })
        
        return {
            "query": query,
            "results": search_results,
            "collection_name": collection_name
        }
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching vector database: {str(e)}"
        )

@app.get("/collections")
async def list_collections():
    """List available collections."""
    try:
        client = get_chroma_client()
        collections = []
        
        for collection in client.list_collections():
            try:
                count = collection.count()
                collections.append({
                    "name": collection.name,
                    "count": count,
                    "metadata": collection.metadata
                })
            except Exception as e:
                collections.append({
                    "name": collection.name,
                    "count": 0,
                    "metadata": {},
                    "error": str(e)
                })
        
        return {"collections": collections}
    
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing collections: {str(e)}"
        )

# Exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Request/response middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests and responses."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} ({process_time:.4f}s)")
    
    return response

def main():
    """Run the API server."""
    import uvicorn
    
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8080))
    workers = int(os.environ.get("API_WORKERS", 1))
    log_level = os.environ.get("API_LOG_LEVEL", "info")
    
    logger.info(f"Starting RAG API server on {host}:{port}")
    
    uvicorn.run(
        "src.rag.api:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=True  # Enable auto-reload in development
    )

if __name__ == "__main__":
    main()
