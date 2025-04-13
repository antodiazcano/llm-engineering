"""
Script to generate train and test datasets in the format is instruction-answer pairs.
"""

import re
import os
from typing import Literal
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


def _generate_sft_dataset(
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
    Pairs of instructions and rephrased answers.
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
    prompt = re.sub(r"[^\S\n]+", " ", prompt)  # delete more than one blanck space

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


def _generate_dpo_dataset(
    extract: str, temperature: float = 0.7
) -> list[tuple[str, str, str]]:
    """
    Generates triplets of instruction and accepted and rejected answers given the
    extract. Higher temperatures will give more diverse outputs.

    Parameters
    ----------
    extract     : Chunk of the document.
    temperature : Temperature.

    Returns
    -------
    Triplets of extract, accepted and rejected answer.
    """

    prompt = f"""Based on the following extract, generate five instruction-answer \
    triples. Each triple should consist of:
    1. An instruction asking about a specific topic in the context.
    2. A generated answer that attempts to answer the instruction based on the context.
    3. An extracted answer that is a relevant excerpt directly from the given context.
    Instructions must be self-contained and general, without explicitly mentioning a \
    context, system, course, or extract.

    Important:
    - Ensure that the extracted answer is a verbatim copy from the context, including \
    all punctuation and apostrophes.
    - Do not add any ellipsis (...) or [...] to indicate skipped text in the extracted \
    answer.
    - If the relevant text is not continuous, use two separate sentences from the \
    context instead of skipping text.

    Provide your response in JSON format with the following structure:
    {{
        "preference_triples": [
            {{
                "instruction": "...",
                "generated_answer": "...",
                "extracted_answer":"...",
                "..."
            }},
            ...
        ]
    }}

    Extract:
    {extract}
    """
    prompt = re.sub(r"[^\S\n]+", " ", prompt)  # delete more than one blanck space

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
    triplets = [
        (pair["instruction"], pair["generated_answer"], pair["extracted_answer"])
        for pair in json_data["preference_triples"]
    ]

    return triplets


def generate_dataset(
    mode: Literal["sft", "dpo"], out_path: str = "data/datasets"
) -> None:
    """
    Generates the train and test datasets. In the book they say that as the dataset is
    small, no exploration is needed. Also, they push the dfs to Hugging Face.

    Parameters
    ----------
    mode     : To create the dataset for SFT or DPO.
    out_path : Path to the folder where dataframes will be saved.
    """

    # Extract documents, generate chunks and obtain pairs
    documents = _get_documents()
    chunks = _extract_chunks(documents)
    output = []
    # The free plan limit is 5 requests per minute, so may be there are some errors
    for chunk in chunks:
        if mode == "sft":
            output += _generate_sft_dataset(chunk)
        else:
            output += _generate_dpo_dataset(chunk)  # type: ignore
    # Save dfs
    if mode == "sft":
        instructions, answers = zip(*output)
        df = pd.DataFrame({"instructions": instructions, "answers": answers})
    else:
        instructions, rejected_answers, accepted_answers = zip(*output)
        df = pd.DataFrame(
            {
                "instructions": instructions,
                "rejected": rejected_answers,
                "chosen": accepted_answers,
            }
        )
    df.to_csv(f"{out_path}/df_{mode}.csv", index=False)


if __name__ == "__main__":
    MODE: Literal["sft", "dpo"] = "dpo"
    generate_dataset(MODE)
