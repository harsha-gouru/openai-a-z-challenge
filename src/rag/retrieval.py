#!/usr/bin/env python
"""
Amazon Deep Insights - Retrieval Module

This module contains functions for retrieving data from the vector store.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

import os
import logging

# Import helper utilities from the sibling *embedding* module
from .embedding import (
    create_chroma_client,
    get_openai_embedding_function,
    get_or_create_collection,
)

# --------------------------------------------------------------------------- #
# Module-level config & logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_VECTOR_DB_DIR = os.environ.get("CHROMA_DB_DIR", "data/vector_db")
DEFAULT_COLLECTION_NAME = "amazon_insights"
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #


def get_collection(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    persist_dir: str = DEFAULT_VECTOR_DB_DIR,
) -> "chromadb.Collection":  # type: ignore[name-defined]
    """
    Helper to **open (or create)** a ChromaDB collection with the configured
    embedding function.
    """

    import chromadb  # local import to avoid hard dep for non-retrieval callers

    client = create_chroma_client(persist_dir)
    embed_fn = get_openai_embedding_function(model=DEFAULT_EMBEDDING_MODEL)
    return get_or_create_collection(client, collection_name, embed_fn)


def semantic_search(
    query: str,
    *,
    collection: Optional["chromadb.Collection"] = None,  # type: ignore[name-defined]
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_dir: str = DEFAULT_VECTOR_DB_DIR,
    n_results: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a semantic search against the vector store.

    Parameters
    ----------
    query
        Natural-language search string.
    collection / collection_name
        Provide an already-opened collection or a name to be opened.
    persist_dir
        Directory for on-disk ChromaDB.  Ignored if *collection* passed.
    n_results
        Number of hits to return.
    filters
        Metadata filters (`where`) passed directly to ChromaDB.
    where_document
        Document-level filters.

    Returns
    -------
    List[Dict[str, Any]]
        Each dict contains: `id`, `document`, `metadata`, `score`
    """

    if collection is None:
        collection = get_collection(collection_name, persist_dir=persist_dir)

    res = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filters,
        where_document=where_document,
    )

    hits: List[Dict[str, Any]] = []
    for idx, (doc_id, doc, meta, dist) in enumerate(
        zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )
    ):
        hits.append(
            {
                "rank": idx + 1,
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                # Convert distance (0 = identical) to a similarity 0-1
                "score": float(1.0 - dist),
            }
        )

    return hits


__all__ = [
    "get_collection",
    "semantic_search",
]

