"""
RAG Pipeline with five steps: Load the documents from MongoDB (1), clean the documents
(2), chunk and embed cleaned documents (3-4) and save cleaned documents (for training)
and embeddings (for inference) into the vector db (5).
"""

from zenml import pipeline

from src.rag_populate_db.steps import (
    load_raw_documents,
    clean_documents,
    chunk_documents,
    embed_chunks,
)


@pipeline
def feature_engineering(authors: list[str]) -> None:
    """
    Executes the RAG pipeline.

    Parameters
    ----------
    authors : Name of the authors of the documents to load.
    """

    raw_documents = load_raw_documents(authors)
    cleaned_documents = clean_documents(raw_documents)
    chunks = chunk_documents(cleaned_documents, authors, min_length=100, max_length=500)
    embed_chunks(chunks)


if __name__ == "__main__":
    AUTHORS = ["Paul Iusztin"]
    feature_engineering(AUTHORS)
