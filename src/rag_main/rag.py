"""
Script that joins the pre-retrieval, retrieval and post-retrieval steps.
"""

import os
import re
from huggingface_hub import InferenceClient

from src.rag_main.pre_retrieval import query_expansion, self_query
from src.rag_main.retrieval import top_k_matches
from src.rag_main.post_retrieval import rerank


def _augment_prompt(original_query: str, n: int, k: int) -> str:
    """
    Joins all the RAG pipeline steps.

    Parameters
    ----------
    original_query : Original query of the user.
    n              : Number of expanded queries.
    k              : To select the number documents from the vector db.

    Returns
    -------
    Top k documents related to the user query.
    """

    # Pre-retrieval
    queries = query_expansion(original_query, n)
    metadata = self_query(original_query)

    # Retrieval
    nk_matches = []
    ids = []
    filters = {"author": metadata}  # optional
    for query in queries:
        top_k_query = top_k_matches(query, k, filters=filters)
        for result in top_k_query:
            if result.id not in ids:  # to avoid repetition
                nk_matches.append(result.payload["chunk"])  # type: ignore
                ids.append(result.id)  # type: ignore

    # Post-retrieval
    top_k_documents = rerank(original_query, nk_matches, k)

    # Prompt
    prompt = f"""You are a content creator. Write what the user asked you to while \
    using the provided context as the primary source of information for the content.

    User query: {original_query}

    Context: {top_k_documents}
    """

    return re.sub(r"[^\S\n]+", " ", prompt)


def main(original_query: str, n: int, k: int) -> str:
    """
    Augments the user prompt with the retrieved documents and returns the LLM response.

    Parameters
    ----------
    original_query : Original query of the user.
    n              : Number of expanded queries.
    k              : To select the number documents from the vector db.

    Returns
    -------
    Response of the LLM having the retrieved documents as context.
    """

    augmented_prompt = _augment_prompt(original_query, n, k)

    client = InferenceClient(
        provider="nebius",
        api_key=os.environ["HUGGINGFACE_KEY"],
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": augmented_prompt}],
    )
    text = completion.choices[0].message.content
    end_tag = "</think>"
    think_pos = text.find(end_tag)
    response = text[think_pos + len(end_tag) :].strip()

    return response
