"""
Script to generate train and test datasets in the format is instruction-answer pairs.
"""

import re
import os
import json
import pandas as pd
from pymongo import MongoClient
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


def _get_documents() -> list[str]:
    """
    Obtains all documents from MongoDB.

    Returns
    -------
    List with all stored documents.
    """

    client: MongoClient = MongoClient("localhost", 27017)
    db = client.antonio
    docs = db.documents

    documents = []
    for doc in docs.find():
        documents.append(doc["content"]["Content"])

    return documents


def _clean_text(text: str) -> str:
    """
    Removes non-alphanumeric characters except for apostrophes, periods, commas,
    exclamation marks, and question marks. replace multiple consecutive whitespace
    characters with a single space.

    Parameters
    ----------
    text : Text to clean.

    Returns
    -------
    Cleaned text.
    """

    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_chunks(
    documents: list[str], min_length: int = 1000, max_length: int = 2000
) -> list[str]:
    """
    Divides the documents into multiple chunks.

    Parameters
    ----------
    documents  : List with all the available documents.
    min_length : Minimum length of the chunk.
    max_length : Maximum length of the chunk.

    Returns
    -------
    Chunks obtained.
    """

    answers = []
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

    for document in documents:
        cleaned_article = _clean_text(document)
        sentences = re.split(sentence_pattern, cleaned_article)
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if len(current_chunk) >= min_length:
                    answers.append(current_chunk.strip())
                current_chunk = sentence + " "

        if len(current_chunk) >= min_length:
            answers.append(current_chunk.strip())

    return answers


def _generate_instruction_answer_pairs(
    answer: str, temperature: float = 0.7
) -> list[tuple[str, str]]:
    """
    Generates pairs of instruction and rephrased answer given the answer. Higher
    temperatures will give more diverse outputs.

    Parameters
    ----------
    answer      : Chunk of the document.
    temperature : Temperature.

    Returns
    -------
    pairs of instructions and rephrased answers.
    """

    prompt = f"""Based on the following extract, generate five instruction-answer \
    pairs. Each instruction must ask to write about a specific topic contained in the \
    context. Each answer must provide a relevant paragraph based on the information \
    found in the context. Only use concepts from the context to generate the \
    instructions. Instructions must never explicitly mention a context, a system, a \
    course, or an extract. Instructions must be self-contained and general. Answers \
    must imitate the writing style of the context.

    Example instruction: Explain the concept of an LLM Twin.
    Example answer: An LLM Twin is essentially an AI character that mimics your \
    writing style, personality, and voice. It's designed to write just like you by \
    incorporating these elements into a language model. The idea is to create a \
    digital replica of your writing habits using advanced AI techniques.

    Provide your response in JSON format with the following structure:
    {{
        "instruction_answer_pairs": [
            {{"instruction": "...", "answer": "..."}},
            ...
        ]
    }}

    Extract:
    {answer}
    """

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
    try:
        response = model.generate_content(
            prompt, generation_config={"temperature": temperature}
        )
    except ResourceExhausted:
        print("WARNING: Requests per minute exceeded!")
        return []

    json_str = response.text.strip("`json\n").strip("`").strip()
    json_data = json.loads(json_str)
    pairs = [
        (pair["instruction"], pair["answer"])
        for pair in json_data["instruction_answer_pairs"]
    ]

    return pairs


def generate_dataset(out_path: str = "data/datasets") -> None:
    """
    Generates the train and test datasets. In the book they say that as the dataset is
    small, no exploration is needed. Also, they push the dfs to Hugging Face.

    Parameters
    ----------
    out_path : Path to the folder where dataframes will be saved.
    """

    # Extract documents, generate chunks and obtain pairs
    documents = _get_documents()
    chunks = _extract_chunks(documents)
    instruction_answer_pairs = []
    # The free plan limit is 5 requests per minute, so may be there are some errors
    for chunk in chunks:
        instruction_answer_pairs += _generate_instruction_answer_pairs(chunk)
    instructions, answers = zip(*instruction_answer_pairs)

    # Save dfs
    df = pd.DataFrame({"instructions": instructions, "answers": answers})
    df.to_csv(f"{out_path}/df_sft.csv", index=False)


if __name__ == "__main__":
    generate_dataset()
