#!/usr/bin/env python
"""
Amazon Deep Insights - Embedding Module

This module contains functions for creating text and spatial embeddings using
OpenAI's embedding models. It provides core functionality for:

1. Generating embeddings for text documents
2. Processing and embedding geospatial features
3. Batch processing for efficient API usage

This module is designed to work with the RAG system and can be used
independently or as part of the larger embedding pipeline.
"""

from __future__ import annotations

import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("amazon_insights.rag.embedding")

# Default settings
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 1  # seconds

# Optional imports
try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon, LineString
except ImportError:  # pragma: no cover
    gpd = None  # type: ignore
    Point = Polygon = MultiPolygon = LineString = None  # type: ignore


class EmbeddingError(Exception):
    """Exception raised for errors during embedding generation."""
    pass


def get_openai_client(api_key: Optional[str] = None) -> "openai.OpenAI":
    """
    Get an OpenAI client instance.
    
    Args:
        api_key (Optional[str]): OpenAI API key (defaults to OPENAI_API_KEY env var)
        
    Returns:
        openai.OpenAI: OpenAI client
        
    Raises:
        EmbeddingError: If API key is not provided and not in environment
    """
    if openai is None:
        raise ImportError("openai package is not installed. Install with 'pip install openai'")
    
    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise EmbeddingError("OpenAI API key not provided and OPENAI_API_KEY not set in environment")
    
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception as e:
        raise EmbeddingError(f"Error creating OpenAI client: {e}")


def embed_texts(
    texts: List[str],
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retry_count: int = DEFAULT_RETRY_COUNT,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI API.
    
    Args:
        texts (List[str]): List of texts to embed
        model (str): OpenAI embedding model to use
        api_key (Optional[str]): OpenAI API key
        batch_size (int): Number of texts to embed in one API call
        retry_count (int): Number of retries on API failure
        retry_delay (float): Delay between retries in seconds
        
    Returns:
        List[List[float]]: List of embeddings
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    if not texts:
        return []
    
    # Get OpenAI client
    client = get_openai_client(api_key)
    
    embeddings = []
    
    # Process in batches to avoid rate limits
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Add delay to avoid rate limits (except for first batch)
        if i > 0:
            time.sleep(0.5)
        
        # Try with retries
        for attempt in range(retry_count):
            try:
                # Generate embeddings for batch
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                break
            
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"Embedding attempt {attempt+1} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise EmbeddingError(f"Failed to generate embeddings after {retry_count} attempts: {e}")
    
    return embeddings


def embed_documents(
    documents: List[Dict[str, Any]],
    text_key: str = "text",
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a list of document dictionaries.
    
    Args:
        documents (List[Dict[str, Any]]): List of document dictionaries
        text_key (str): Key for the text field in each document
        model (str): OpenAI embedding model to use
        api_key (Optional[str]): OpenAI API key
        batch_size (int): Number of texts to embed in one API call
        
    Returns:
        List[Dict[str, Any]]: Documents with embeddings added
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    if not documents:
        return []
    
    # Extract texts from documents
    texts = [doc.get(text_key, "") for doc in documents]
    
    # Generate embeddings
    embeddings = embed_texts(
        texts=texts,
        model=model,
        api_key=api_key,
        batch_size=batch_size
    )
    
    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc["embedding"] = embedding
    
    return documents


def spatial_feature_to_text(
    feature: Any,
    properties: Optional[List[str]] = None,
    include_geometry: bool = True,
) -> str:
    """
    Convert a geospatial feature to text representation.
    
    Args:
        feature: GeoPandas feature or dictionary with geometry
        properties (Optional[List[str]]): Properties to include
        include_geometry (bool): Whether to include geometry description
        
    Returns:
        str: Text representation of the feature
    """
    if gpd is None:
        raise ImportError("GeoPandas not installed. Install with 'pip install geopandas'")
    
    text_parts = []
    
    # Handle different input types
    if isinstance(feature, gpd.GeoSeries):
        geom = feature.geometry
        attrs = feature.to_dict()
    elif isinstance(feature, dict):
        geom = feature.get("geometry")
        attrs = {k: v for k, v in feature.items() if k != "geometry"}
    else:
        raise ValueError(f"Unsupported feature type: {type(feature)}")
    
    # Add properties
    if properties:
        for prop in properties:
            if prop in attrs:
                text_parts.append(f"{prop}: {attrs[prop]}")
    else:
        for key, value in attrs.items():
            if key != "geometry":
                text_parts.append(f"{key}: {value}")
    
    # Add geometry
    if include_geometry and geom is not None:
        if isinstance(geom, Point):
            text_parts.append(f"geometry_type: Point")
            text_parts.append(f"coordinates: {geom.x:.6f}, {geom.y:.6f}")
        elif isinstance(geom, Polygon):
            text_parts.append(f"geometry_type: Polygon")
            centroid = geom.centroid
            text_parts.append(f"centroid: {centroid.x:.6f}, {centroid.y:.6f}")
            text_parts.append(f"area: {geom.area:.2f}")
            text_parts.append(f"bounds: {geom.bounds}")
        elif isinstance(geom, MultiPolygon):
            text_parts.append(f"geometry_type: MultiPolygon")
            centroid = geom.centroid
            text_parts.append(f"centroid: {centroid.x:.6f}, {centroid.y:.6f}")
            text_parts.append(f"area: {geom.area:.2f}")
            text_parts.append(f"num_parts: {len(geom.geoms)}")
        elif isinstance(geom, LineString):
            text_parts.append(f"geometry_type: LineString")
            text_parts.append(f"length: {geom.length:.2f}")
            text_parts.append(f"bounds: {geom.bounds}")
    
    return "\n".join(text_parts)


def embed_geospatial_features(
    features: Union[List[Dict[str, Any]], "gpd.GeoDataFrame"],
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: Optional[str] = None,
    properties: Optional[List[str]] = None,
    include_geometry: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for geospatial features.
    
    Args:
        features: List of features or GeoDataFrame
        model (str): OpenAI embedding model to use
        api_key (Optional[str]): OpenAI API key
        properties (Optional[List[str]]): Properties to include in text representation
        include_geometry (bool): Whether to include geometry in text representation
        batch_size (int): Number of texts to embed in one API call
        
    Returns:
        List[Dict[str, Any]]: Features with embeddings added
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    if gpd is None:
        raise ImportError("GeoPandas not installed. Install with 'pip install geopandas'")
    
    # Convert GeoDataFrame to list of dictionaries
    if isinstance(features, gpd.GeoDataFrame):
        feature_list = features.to_dict('records')
        for i, feature in enumerate(feature_list):
            feature['geometry'] = features.geometry.iloc[i]
    else:
        feature_list = features
    
    # Convert features to text
    texts = []
    for feature in feature_list:
        text = spatial_feature_to_text(feature, properties, include_geometry)
        texts.append(text)
    
    # Generate embeddings
    embeddings = embed_texts(
        texts=texts,
        model=model,
        api_key=api_key,
        batch_size=batch_size
    )
    
    # Add embeddings to features
    for feature, embedding in zip(feature_list, embeddings):
        feature["embedding"] = embedding
    
    return feature_list


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1 (List[float]): First vector
        v2 (List[float]): Second vector
        
    Returns:
        float: Cosine similarity (-1 to 1)
    """
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    
    dot_product = np.dot(v1_array, v2_array)
    norm_v1 = np.linalg.norm(v1_array)
    norm_v2 = np.linalg.norm(v2_array)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)


def find_most_similar(
    query_embedding: List[float],
    embeddings: List[List[float]],
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Find the most similar embeddings to a query embedding.
    
    Args:
        query_embedding (List[float]): Query embedding
        embeddings (List[List[float]]): List of embeddings to search
        top_k (int): Number of results to return
        
    Returns:
        List[Tuple[int, float]]: List of (index, similarity) tuples
    """
    similarities = []
    
    for i, embedding in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return similarities[:top_k]


__all__ = [
    "embed_texts",
    "embed_documents",
    "spatial_feature_to_text",
    "embed_geospatial_features",
    "cosine_similarity",
    "find_most_similar",
    "EmbeddingError",
]
