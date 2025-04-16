"""
Script to return top k chunks in the vector db related to a query.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import ScoredPoint
from sentence_transformers.SentenceTransformer import SentenceTransformer

from src.rag_main.constants import SENTENCE_TRANSFORMER


def top_k_matches(
    query: str, k: int, filters: dict[str, str] | None = None
) -> list[ScoredPoint]:
    """
    Returns top k chunks in the vector db related to a query.

    Parameters
    ----------
    query : User query.
    k     : Number of extracted documents.
    filters

    Returns
    -------
    Top k related documents to the query.
    """

    qdrant_client = QdrantClient(
        url="https://b7fce096-1c85-492d-b757-1724657c30f2.eu-west-2-0.aws.cloud.qdrant."
        "io:6333",
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    model = SentenceTransformer(SENTENCE_TRANSFORMER)
    query_embedding = model.encode(query)

    if filters is None:
        return qdrant_client.search(
            collection_name="llms",
            query_vector=query_embedding,  # type: ignore
            limit=k,
        )

    result = qdrant_client.search(
        collection_name="llms",
        query_vector=query_embedding,  # type: ignore
        limit=k,
        query_filter=Filter(must=_generate_filters(filters)),  # type: ignore
    )
    if len(result) > 0:
        return result
    # If len was 0 that means probably the metadata was not correct
    return qdrant_client.search(
        collection_name="llms",
        query_vector=query_embedding,  # type: ignore
        limit=k,
    )


def _generate_filters(filters: dict[str, str]) -> list[FieldCondition]:
    """
    Generates the correct format for the filters.

    Parameters
    ----------
    filters : Dictionary with the filters.

    Returns
    -------
    List with 'FieldCondition' objects.
    """

    return [
        FieldCondition(key=restriction_type, match=MatchValue(value=restriction_value))
        for restriction_type, restriction_value in zip(filters.keys(), filters.values())
    ]
