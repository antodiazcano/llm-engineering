"""
Steps of the RAG pipeline.
"""

import os
import re
from loguru import logger
from dotenv import load_dotenv
from zenml import get_step_context, step
from pymongo import MongoClient
from sentence_transformers.SentenceTransformer import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient


MONGO_CLIENT: MongoClient = MongoClient("localhost", 27017)
MONGO_DB = MONGO_CLIENT.antonio
DOCUMENTS_DB = MONGO_DB.documents


@step
def load_raw_documents(authors: list[str]) -> list[dict]:
    """
    Loads the raw documents from MongoDB.

    Parameters
    ----------
    authors : Name of authors of the documents.
    """

    logger.info("Loading documents...")
    docs = []

    for author in authors:
        docs += list(DOCUMENTS_DB.find({"author": author}))

    step_context = get_step_context()
    step_context.add_output_metadata(metadata={})

    return docs


@step
def clean_documents(documents: list[dict]) -> list[dict]:
    """
    Cleans the documents.

    Parameters
    ----------
    documents : Documents to clean.

    Returns
    -------
    Cleaned documents.
    """

    logger.info("Cleaning documents...")

    for doc in documents:
        for part in ["Title", "Subtitle", "Content"]:
            doc["content"][part] = remove_emojis_and_non_ascii(doc["content"][part])

    step_context = get_step_context()
    step_context.add_output_metadata(metadata={})

    save_to_vector_db([doc["content"]["Content"] for doc in documents])
    logger.info("Clean documents saved!")

    return documents


def remove_emojis_and_non_ascii(text: str) -> str:
    """
    Removes emojis and non-ASCII characters.

    Parameters
    ----------
    text : Raw text.

    Returns
    -------
    Cleaned text.
    """

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove emojis based on Unicode categories (symbols and pictograms)
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f"  # Emoticons
        "\U0001f300-\U0001f5ff"  # Symbols and pictograms
        "\U0001f680-\U0001f6ff"  # Transport and map symbols
        "\U0001f700-\U0001f77f"  # Alchemy
        "\U0001f780-\U0001f7ff"  # Additional geometry
        "\U0001f800-\U0001f8ff"  # Additional symbols
        "\U0001f900-\U0001f9ff"  # Supplements
        "\U0001fa00-\U0001fa6f"  # Additional objects
        "\U0001fa70-\U0001faff"  # More symbols
        "\U00002702-\U000027b0"  # Other dingbats
        "\U000024c2-\U0001f251]+",  # Other characters
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)

    return text


@step
def chunk_documents(
    documents: list[dict], min_length: int = 100, max_length: int = 500
) -> list[str]:
    """
    Creates the chunks of the documents.

    Parameters
    ----------
    documents  : Cleaned documents.
    min_length : Minimum length of each chunk.
    max_length : Maximum length of each chunk.

    Returns
    -------
    Chunks of the documents.
    """

    logger.info("Chunking documents...")
    chunks = []

    for doc in documents:
        sentences = re.split(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", doc["content"]["Content"]
        )  # separate text in sentences by points and ?
        current_chunk = ""
        for sentence in sentences:
            if len(sentence) > max_length:
                continue
            sentence = sentence.strip()
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if len(current_chunk) >= min_length:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
        if len(current_chunk) >= min_length:
            chunks.append(current_chunk.strip())

    step_context = get_step_context()
    step_context.add_output_metadata(metadata={})

    return chunks


@step
def embed_chunks(chunks: list[str], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Obtains the embeddings for the chunks.

    Parameters
    ----------
    chunks : Chunks to convert.
    model  : Model used for the Sentence Transformer.

    Returns
    -------
    Embeddings for the chunks. Dimensions: [len(chunks), embedding_dim].
    """

    logger.info("Creating embeddings...")
    model = SentenceTransformer(model)  # type:ignore
    embeddings = model.encode(chunks)  # type:ignore

    step_context = get_step_context()
    step_context.add_output_metadata(metadata={})

    save_to_vector_db(chunks, embeddings)  # type:ignore
    logger.info("Embeddings saved!")

    return embeddings  # type:ignore


def save_to_vector_db(chunks: list[str], embeddings: np.ndarray | None = None) -> bool:
    """
    Saves the chunks and embeddings to Qdrant. If no embeddings are provided, then the
    text is saved for fine-tuning. Otherwise, it is saved for inference.

    Parameters
    ----------
    chunks     : Text chunks.
    embeddings : Embeddings of the chunks.

    Returns
    -------
    Dummy return.
    """

    load_dotenv()

    logger.info("Saving to Qdrant...")
    qdrant_client = QdrantClient(
        url="https://b7fce096-1c85-492d-b757-1724657c30f2.eu-west-2-0.aws.cloud.qdrant."
        "io:6333",
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # qdrant_client.recreate_collection(
    #    collection_name="llms", vectors_config={"size": 384, "distance": "Cosine"}
    # )
    # print(qdrant_client.get_collections())

    qdrant_client.upsert(
        collection_name="llms",
        points=[
            {
                "id": i,
                "vector": (
                    embeddings[i]
                    if embeddings is not None
                    else np.zeros(
                        384,
                    )
                ),
                "payload": {
                    "mode": "inference" if embeddings is not None else "train",
                    "chunk": chunks[i],
                },
            }
            for i in range(len(chunks))
        ],
    )

    step_context = get_step_context()
    step_context.add_output_metadata(metadata={})

    return True
