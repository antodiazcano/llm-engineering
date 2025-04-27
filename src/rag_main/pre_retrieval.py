"""
Script to perform the pre-retrieval step of the RAG pipeline (query expansion and self
query).
"""

import os
import re
from huggingface_hub import InferenceClient


def query_expansion(
    question: str, expand_to_n: int, separator: str = "#next-question#"
) -> list[str]:
    """
    Expands the user query to 'expand_to_n' queries to give more diversity.

    Parameters
    ----------
    question : User query.

    """

    # Prompt
    expand_to_n -= 1
    prompt = f"""You are an AI language model assistant. Your task is to generate \
    {expand_to_n} different versions of the given user question to retrieve relevant \
    documents from a vector database. By generating multiple perspectives on the user \
    question, your goal is to help the user overcome some of the limitations of the \
    distance-based similarity search.

    Provide these alternative questions separated by '{separator}' and do not use \
    nothing at the start. For example, '1: ...' or 'Version 1: ...' is not correct. \
    Just write the alternative questions separated.

    Original question: {question}.
    """
    prompt = re.sub(r"[^\S\n]+", " ", prompt)

    # Expand with the LLM
    client = InferenceClient(
        provider="nebius",
        api_key=os.environ["HUGGINGFACE_KEY"],
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
    )
    text = completion.choices[0].message.content
    end_tag = "</think>"
    think_pos = text.find(end_tag)
    response = text[think_pos + len(end_tag) :].strip()

    # Give correct format. The regular expression is just eliminating numbers at the
    # beginning and '\n' at the end of each query.
    response = [question] + [
        re.sub(r"^\s*\d+\.\s*", "", query).replace("\n", "").strip()
        for query in response.split(separator)
    ]

    return response


def self_query(question: str) -> str:
    """
    Obtains metadata associated with the question.

    Parameters
    ----------
    question : Query of te user.

    Returns
    -------
    Extracted metadata.
    """

    prompt = f"""You are an AI language model assistant. Your task is to extract \
    information from a user question.

    The required information that needs to be extracted is the user name or user id. \
    Your response should consist of only the extracted user name (e.g., John Doe) or \
    id (e.g., 1345256), nothing else. If the user question does not contain any user \
    name or id, you should return the following token: none.

    For example:

    QUESTION 1:
    My name is Paul Iusztin and I want a post about...
    RESPONSE 1:
    Paul Iusztin

    QUESTION 2:
    I want to write a post about...
    RESPONSE 2:
    none

    QUESTION 3:
    My user id is 1345256 and I want to write a post about...
    RESPONSE 3:
    1345256

    User question: {question}
    """
    prompt = re.sub(r"[^\S\n]+", " ", prompt)

    # Search with the LLM
    client = InferenceClient(
        provider="nebius",
        api_key=os.environ["HUGGINGFACE_KEY"],
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
    )
    text = completion.choices[0].message.content
    end_tag = "</think>"
    think_pos = text.find(end_tag)
    response = text[think_pos + len(end_tag) :].strip()

    return response
