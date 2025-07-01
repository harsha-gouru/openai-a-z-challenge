#!/usr/bin/env python
"""
Amazon Deep Insights - Embeddings Module

This module provides functions for generating embeddings and creating a vector database
for the RAG system. It supports:
- Text document embedding
- Geospatial feature embedding
- Vector database creation and management with ChromaDB
- Document chunking and processing
"""

import os
import sys
import json
import logging
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, Iterator
import time
import re

import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader
)
from langchain.document_transformers import Html2TextTransformer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amazon_insights.rag.embeddings")

# Load environment variables
load_dotenv()

# Default paths and settings
DEFAULT_VECTOR_DB_DIR = os.environ.get("CHROMA_DB_DIR", "data/vector_db")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
DEFAULT_BATCH_SIZE = 16  # Number of texts to embed in one API call


class EmbeddingError(Exception):
    """Exception raised for errors during embedding generation."""
    pass


def get_openai_embedding_function(
    api_key: Optional[str] = None,
    model: str = DEFAULT_EMBEDDING_MODEL
) -> embedding_functions.OpenAIEmbeddingFunction:
    """
    Get an OpenAI embedding function for ChromaDB.
    
    Args:
        api_key (Optional[str]): OpenAI API key (defaults to OPENAI_API_KEY env var)
        model (str): OpenAI embedding model to use
        
    Returns:
        embedding_functions.OpenAIEmbeddingFunction: ChromaDB-compatible embedding function
        
    Raises:
        EmbeddingError: If API key is not provided and not in environment
    """
    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise EmbeddingError("OpenAI API key not provided and OPENAI_API_KEY not set in environment")
    
    try:
        # Create and return the embedding function
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model
        )
    except Exception as e:
        raise EmbeddingError(f"Error creating OpenAI embedding function: {e}")


def create_chroma_client(
    persist_directory: str = DEFAULT_VECTOR_DB_DIR,
    client_type: str = "local"
) -> chromadb.Client:
    """
    Create a ChromaDB client.
    
    Args:
        persist_directory (str): Directory to persist the database
        client_type (str): Type of client ('local' or 'http')
        
    Returns:
        chromadb.Client: ChromaDB client
        
    Raises:
        EmbeddingError: If client creation fails
    """
    try:
        if client_type == "local":
            # Create directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            # Create a persistent client
            return chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        
        elif client_type == "http":
            # Connect to a ChromaDB server
            host = os.environ.get("CHROMA_SERVER_HOST", "localhost")
            port = os.environ.get("CHROMA_SERVER_PORT", "8000")
            
            return chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        
        else:
            raise ValueError(f"Invalid client_type: {client_type}")
    
    except Exception as e:
        raise EmbeddingError(f"Error creating ChromaDB client: {e}")


def get_or_create_collection(
    client: chromadb.Client,
    collection_name: str,
    embedding_function: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection.
    
    Args:
        client (chromadb.Client): ChromaDB client
        collection_name (str): Name of the collection
        embedding_function (Any): Embedding function to use
        metadata (Optional[Dict[str, Any]]): Metadata for the collection
        
    Returns:
        chromadb.Collection: ChromaDB collection
    """
    try:
        # Try to get existing collection
        return client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except Exception:
        # Create new collection if it doesn't exist
        return client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata=metadata or {}
        )


def load_document(file_path: str) -> List[Any]:
    """
    Load a document from a file using the appropriate loader.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        List[Any]: List of document chunks
        
    Raises:
        EmbeddingError: If document loading fails
    """
    if not os.path.exists(file_path):
        raise EmbeddingError(f"File not found: {file_path}")
    
    try:
        # Select loader based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".csv":
            loader = CSVLoader(file_path)
        elif file_ext in [".html", ".htm"]:
            loader = UnstructuredHTMLLoader(file_path)
        elif file_ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_ext in [".txt", ".text"]:
            loader = TextLoader(file_path)
        else:
            # Try generic loader for other file types
            loader = UnstructuredFileLoader(file_path)
        
        # Load the document
        return loader.load()
    
    except Exception as e:
        raise EmbeddingError(f"Error loading document {file_path}: {e}")


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum chunk size in characters
        chunk_overlap (int): Overlap between chunks in characters
        
    Returns:
        List[str]: List of text chunks
    """
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split text into chunks
    return text_splitter.split_text(text)


def chunk_document(
    document: List[Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Split a document into chunks.
    
    Args:
        document (List[Any]): Document to split
        chunk_size (int): Maximum chunk size in characters
        chunk_overlap (int): Overlap between chunks in characters
        
    Returns:
        List[Dict[str, Any]]: List of document chunks with metadata
    """
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split document into chunks
    chunks = text_splitter.split_documents(document)
    
    # Format chunks with metadata
    formatted_chunks = []
    
    for i, chunk in enumerate(chunks):
        formatted_chunks.append({
            "id": f"{chunk.metadata.get('source', 'unknown')}_{i}",
            "text": chunk.page_content,
            "metadata": {
                **chunk.metadata,
                "chunk_index": i
            }
        })
    
    return formatted_chunks


def generate_openai_embeddings(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI API.
    
    Args:
        texts (List[str]): List of texts to embed
        model (str): OpenAI embedding model to use
        api_key (Optional[str]): OpenAI API key
        batch_size (int): Number of texts to embed in one API call
        
    Returns:
        List[List[float]]: List of embeddings
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise EmbeddingError("OpenAI API key not provided and OPENAI_API_KEY not set in environment")
    
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    embeddings = []
    
    try:
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Add delay to avoid rate limits
            if i > 0:
                time.sleep(0.5)
            
            # Generate embeddings for batch
            response = client.embeddings.create(
                model=model,
                input=batch
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return embeddings
    
    except Exception as e:
        raise EmbeddingError(f"Error generating OpenAI embeddings: {e}")


def add_documents_to_collection(
    collection: chromadb.Collection,
    documents: List[Dict[str, Any]],
    batch_size: int = 100
) -> List[str]:
    """
    Add documents to a ChromaDB collection.
    
    Args:
        collection (chromadb.Collection): ChromaDB collection
        documents (List[Dict[str, Any]]): List of documents to add
        batch_size (int): Number of documents to add in one batch
        
    Returns:
        List[str]: List of document IDs
        
    Raises:
        EmbeddingError: If adding documents fails
    """
    document_ids = []
    
    try:
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract document components
            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            
            # Add documents to collection
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            document_ids.extend(ids)
            
            logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} to collection")
        
        return document_ids
    
    except Exception as e:
        raise EmbeddingError(f"Error adding documents to collection: {e}")


def process_directory(
    directory_path: str,
    collection: chromadb.Collection,
    file_extensions: List[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    recursive: bool = True
) -> Dict[str, Any]:
    """
    Process all documents in a directory and add them to a collection.
    
    Args:
        directory_path (str): Path to the directory
        collection (chromadb.Collection): ChromaDB collection
        file_extensions (List[str]): List of file extensions to process
        chunk_size (int): Maximum chunk size in characters
        chunk_overlap (int): Overlap between chunks in characters
        recursive (bool): Whether to process subdirectories recursively
        
    Returns:
        Dict[str, Any]: Processing statistics
        
    Raises:
        EmbeddingError: If processing fails
    """
    if not os.path.isdir(directory_path):
        raise EmbeddingError(f"Directory not found: {directory_path}")
    
    # Default file extensions if not provided
    if file_extensions is None:
        file_extensions = [".pdf", ".txt", ".html", ".htm", ".md", ".csv"]
    
    # Normalize file extensions
    file_extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in file_extensions]
    
    # Find all matching files
    files = []
    
    if recursive:
        for root, _, filenames in os.walk(directory_path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in file_extensions):
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in file_extensions):
                files.append(os.path.join(directory_path, filename))
    
    if not files:
        logger.warning(f"No matching files found in {directory_path}")
        return {"processed_files": 0, "added_chunks": 0, "skipped_files": 0}
    
    # Process each file
    processed_files = 0
    added_chunks = 0
    skipped_files = 0
    
    for file_path in files:
        try:
            # Load and chunk document
            document = load_document(file_path)
            chunks = chunk_document(document, chunk_size, chunk_overlap)
            
            # Add chunks to collection
            add_documents_to_collection(collection, chunks)
            
            processed_files += 1
            added_chunks += len(chunks)
            
            logger.info(f"Processed {file_path}: {len(chunks)} chunks")
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            skipped_files += 1
    
    stats = {
        "processed_files": processed_files,
        "added_chunks": added_chunks,
        "skipped_files": skipped_files
    }
    
    logger.info(f"Directory processing complete: {stats}")
    return stats


def embed_geospatial_features(
    gdf: Any,
    collection_name: str,
    client: chromadb.Client,
    embedding_function: Any,
    text_columns: List[str] = None,
    include_geometry: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> chromadb.Collection:
    """
    Embed geospatial features from a GeoDataFrame.
    
    Args:
        gdf (Any): GeoDataFrame with features
        collection_name (str): Name of the collection
        client (chromadb.Client): ChromaDB client
        embedding_function (Any): Embedding function
        text_columns (List[str]): Columns to include in text representation
        include_geometry (bool): Whether to include geometry in text representation
        chunk_size (int): Maximum chunk size for text representation
        
    Returns:
        chromadb.Collection: ChromaDB collection with embedded features
        
    Raises:
        EmbeddingError: If embedding fails
    """
    try:
        # Import geopandas here to avoid dependency issues
        import geopandas as gpd
        
        # Ensure input is a GeoDataFrame
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a GeoDataFrame")
        
        # Create or get collection
        collection = get_or_create_collection(
            client,
            collection_name,
            embedding_function,
            metadata={"type": "geospatial_features"}
        )
        
        # Default text columns if not provided
        if text_columns is None:
            text_columns = [col for col in gdf.columns if col != 'geometry']
        
        # Process each feature
        documents = []
        
        for idx, row in gdf.iterrows():
            # Create text representation of feature
            feature_text = []
            
            # Add attributes
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    feature_text.append(f"{col}: {row[col]}")
            
            # Add geometry if requested
            if include_geometry and 'geometry' in row and row.geometry is not None:
                geom_type = row.geometry.geom_type
                coords = row.geometry.centroid.coords[0]
                feature_text.append(f"geometry_type: {geom_type}")
                feature_text.append(f"centroid: {coords[0]:.6f}, {coords[1]:.6f}")
                
                # Add bounding box for polygons
                if geom_type in ['Polygon', 'MultiPolygon']:
                    bounds = row.geometry.bounds
                    feature_text.append(f"bounds: {bounds}")
            
            # Join text parts
            text = "\n".join(feature_text)
            
            # Chunk if necessary
            if len(text) > chunk_size:
                chunks = chunk_text(text, chunk_size, chunk_size // 4)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "id": f"feature_{idx}_{i}",
                        "text": chunk,
                        "metadata": {
                            "feature_id": idx,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        }
                    })
            else:
                documents.append({
                    "id": f"feature_{idx}",
                    "text": text,
                    "metadata": {"feature_id": idx}
                })
        
        # Add documents to collection
        add_documents_to_collection(collection, documents)
        
        logger.info(f"Embedded {len(gdf)} geospatial features into {len(documents)} documents")
        return collection
    
    except ImportError:
        raise EmbeddingError("GeoPandas not installed. Install with 'pip install geopandas'")
    
    except Exception as e:
        raise EmbeddingError(f"Error embedding geospatial features: {e}")


def embed_raster_metadata(
    raster_paths: List[str],
    collection_name: str,
    client: chromadb.Client,
    embedding_function: Any
) -> chromadb.Collection:
    """
    Embed metadata from raster files.
    
    Args:
        raster_paths (List[str]): Paths to raster files
        collection_name (str): Name of the collection
        client (chromadb.Client): ChromaDB client
        embedding_function (Any): Embedding function
        
    Returns:
        chromadb.Collection: ChromaDB collection with embedded metadata
        
    Raises:
        EmbeddingError: If embedding fails
    """
    try:
        # Import rasterio here to avoid dependency issues
        import rasterio
        
        # Create or get collection
        collection = get_or_create_collection(
            client,
            collection_name,
            embedding_function,
            metadata={"type": "raster_metadata"}
        )
        
        # Process each raster
        documents = []
        
        for raster_path in raster_paths:
            try:
                with rasterio.open(raster_path) as src:
                    # Extract metadata
                    metadata = {
                        "file_path": raster_path,
                        "filename": os.path.basename(raster_path),
                        "crs": str(src.crs),
                        "width": src.width,
                        "height": src.height,
                        "count": src.count,
                        "dtype": str(src.dtypes[0]),
                        "bounds": {
                            "left": src.bounds.left,
                            "bottom": src.bounds.bottom,
                            "right": src.bounds.right,
                            "top": src.bounds.top
                        },
                        "transform": [float(v) for v in src.transform],
                        "nodata": src.nodata
                    }
                    
                    # Get tags
                    if src.tags():
                        metadata["tags"] = src.tags()
                    
                    # Create text representation
                    text_parts = [
                        f"Raster: {os.path.basename(raster_path)}",
                        f"CRS: {metadata['crs']}",
                        f"Dimensions: {metadata['width']} x {metadata['height']} pixels",
                        f"Bands: {metadata['count']}",
                        f"Data Type: {metadata['dtype']}",
                        f"Bounds: {metadata['bounds']['left']:.6f}, {metadata['bounds']['bottom']:.6f}, "
                        f"{metadata['bounds']['right']:.6f}, {metadata['bounds']['top']:.6f}"
                    ]
                    
                    # Add tags if available
                    if "tags" in metadata:
                        tag_parts = [f"{k}: {v}" for k, v in metadata["tags"].items()]
                        text_parts.append("Tags: " + ", ".join(tag_parts))
                    
                    # Join text parts
                    text = "\n".join(text_parts)
                    
                    # Create document
                    document_id = hashlib.md5(raster_path.encode()).hexdigest()
                    
                    documents.append({
                        "id": document_id,
                        "text": text,
                        "metadata": metadata
                    })
            
            except Exception as e:
                logger.error(f"Error processing raster {raster_path}: {e}")
        
        # Add documents to collection
        if documents:
            add_documents_to_collection(collection, documents)
            
            logger.info(f"Embedded metadata from {len(documents)} raster files")
        else:
            logger.warning("No valid raster files to embed")
        
        return collection
    
    except ImportError:
        raise EmbeddingError("Rasterio not installed. Install with 'pip install rasterio'")
    
    except Exception as e:
        raise EmbeddingError(f"Error embedding raster metadata: {e}")


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Query a ChromaDB collection.
    
    Args:
        collection (chromadb.Collection): ChromaDB collection
        query_text (str): Query text
        n_results (int): Number of results to return
        where (Optional[Dict[str, Any]]): Filter on metadata
        where_document (Optional[Dict[str, Any]]): Filter on document content
        
    Returns:
        Dict[str, Any]: Query results
        
    Raises:
        EmbeddingError: If query fails
    """
    try:
        # Execute query
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return results
    
    except Exception as e:
        raise EmbeddingError(f"Error querying collection: {e}")


def build_knowledge_base(
    corpus_dir: str,
    persist_dir: str = DEFAULT_VECTOR_DB_DIR,
    collection_name: str = "amazon_insights",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    file_extensions: List[str] = None,
    force_rebuild: bool = False
) -> chromadb.Collection:
    """
    Build a knowledge base from a corpus directory.
    
    Args:
        corpus_dir (str): Directory containing corpus documents
        persist_dir (str): Directory to persist the vector database
        collection_name (str): Name of the collection
        embedding_model (str): OpenAI embedding model to use
        chunk_size (int): Maximum chunk size in characters
        chunk_overlap (int): Overlap between chunks in characters
        file_extensions (List[str]): List of file extensions to process
        force_rebuild (bool): Whether to force rebuilding the collection
        
    Returns:
        chromadb.Collection: ChromaDB collection
        
    Raises:
        EmbeddingError: If knowledge base building fails
    """
    try:
        # Create ChromaDB client
        client = create_chroma_client(persist_dir)
        
        # Check if collection exists
        collection_exists = False
        try:
            existing_collections = client.list_collections()
            collection_exists = any(c.name == collection_name for c in existing_collections)
        except Exception:
            pass
        
        # Delete collection if force rebuild
        if collection_exists and force_rebuild:
            logger.info(f"Forcing rebuild of collection '{collection_name}'")
            client.delete_collection(collection_name)
            collection_exists = False
        
        # Create embedding function
        embedding_function = get_openai_embedding_function(model=embedding_model)
        
        # Get or create collection
        collection = get_or_create_collection(
            client,
            collection_name,
            embedding_function,
            metadata={
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )
        
        # If collection exists and not forcing rebuild, return it
        if collection_exists and not force_rebuild:
            logger.info(f"Using existing collection '{collection_name}'")
            return collection
        
        # Process corpus directory
        logger.info(f"Building knowledge base from {corpus_dir}")
        stats = process_directory(
            corpus_dir,
            collection,
            file_extensions,
            chunk_size,
            chunk_overlap,
            recursive=True
        )
        
        logger.info(f"Knowledge base built: {stats}")
        return collection
    
    except Exception as e:
        raise EmbeddingError(f"Error building knowledge base: {e}")


def main():
    """Command-line interface for embeddings module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings and create vector database")
    
    # Command options
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build knowledge base
    build_parser = subparsers.add_parser("build", help="Build knowledge base from corpus")
    build_parser.add_argument("--corpus", "-c", required=True, help="Corpus directory")
    build_parser.add_argument("--persist-dir", "-p", default=DEFAULT_VECTOR_DB_DIR, help="Persist directory")
    build_parser.add_argument("--collection", "-n", default="amazon_insights", help="Collection name")
    build_parser.add_argument("--model", "-m", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model")
    build_parser.add_argument("--chunk-size", "-s", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size")
    build_parser.add_argument("--chunk-overlap", "-o", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap")
    build_parser.add_argument("--force", "-f", action="store_true", help="Force rebuild")
    
    # Query collection
    query_parser = subparsers.add_parser("query", help="Query collection")
    query_parser.add_argument("--persist-dir", "-p", default=DEFAULT_VECTOR_DB_DIR, help="Persist directory")
    query_parser.add_argument("--collection", "-n", default="amazon_insights", help="Collection name")
    query_parser.add_argument("--model", "-m", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model")
    query_parser.add_argument("--query", "-q", required=True, help="Query text")
    query_parser.add_argument("--results", "-r", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    if args.command == "build":
        try:
            collection = build_knowledge_base(
                corpus_dir=args.corpus,
                persist_dir=args.persist_dir,
                collection_name=args.collection,
                embedding_model=args.model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                force_rebuild=args.force
            )
            
            print(f"Knowledge base built successfully: {collection.name}")
            print(f"Collection count: {collection.count()}")
        
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "query":
        try:
            # Create client and get collection
            client = create_chroma_client(args.persist_dir)
            embedding_function = get_openai_embedding_function(model=args.model)
            
            collection = get_or_create_collection(
                client,
                args.collection,
                embedding_function
            )
            
            # Query collection
            results = query_collection(
                collection,
                args.query,
                n_results=args.results
            )
            
            # Print results
            print(f"\nQuery: {args.query}")
            print(f"Results: {len(results['documents'][0])}")
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                print(f"\n--- Result {i+1} (Distance: {distance:.4f}) ---")
                print(f"Metadata: {metadata}")
                print(f"Document: {doc[:200]}...")
        
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
