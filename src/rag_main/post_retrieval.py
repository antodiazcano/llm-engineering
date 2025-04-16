"""
Script to perform the rerank algorithm.
"""

from sentence_transformers.cross_encoder import CrossEncoder

from src.rag_main.constants import CROSS_ENCODER


def rerank(original_query: str, rag_documents: list[str], k: int) -> list[str]:
    """
    Returns the top k related documents with the original query.

    Parameters
    ----------
    original_query : Original user query.
    rag_documents  : Documents extracted from the vector db.
    k              : Number of documents to keep.

    Returns
    -------
    Top k related documents.
    """

    model = CrossEncoder(CROSS_ENCODER)

    top_k_matches = sorted(
        [
            (chunk, float(model.predict([(chunk, original_query)])[0]))  # type: ignore
            for chunk in rag_documents
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    return [match[0] for match in top_k_matches][:k]
