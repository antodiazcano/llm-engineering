"""
This script generates the evaluations of the fine-tuned models.
"""

import os
import re
import json
import pandas as pd
from huggingface_hub import InferenceClient


def evaluate_answer(instruction: str, answer: str) -> str:
    """
    Generates an evaluation of the answer given to the instruction following a certain
    guidelines described in the variable 'prompt'.

    Parameters
    ----------
    instruction : Instruction given to the LLM.
    answer      : Response of the LLM.

    Returns
    -------
    Evaluation of the external LLM judge.
    """

    prompt = f"""You are an expert judge. Please evaluate the quality of a given \
    answer to an instruction based on two criteria:
    1. Accuracy: How factually correct is the information presented in the \
    answer? You are a technical expert in this topic.
    2. Style: Is the tone and writing style appropriate for a blog post or social \
    media content? It should use simple but technical words and avoid formal or \
    academic language.

    Accuracy scale:
    1 (Poor): Contains factual errors or misleading information
    2 (Good): Mostly accurate with minor errors or omissions
    3 (Excellent): Highly accurate and comprehensive

    Style scale:
    1 (Poor): Too formal, uses some overly complex words
    2 (Good): Good balance of technical content and accessibility, but still uses \
    formal words and expressions
    3 (Excellent): Perfectly accessible language for blog/social media, uses simple \
    but precise technical terms when necessary

    Example of bad style: The Llama2 7B model constitutes a noteworthy progression in \
    the field of artificial intelligence, serving as the successor to its predecessor, \
    the original Llama architecture.

    Example of excellent style: Llama2 7B outperforms the original Llama model across \
    multiple benchmarks.

    Instruction: {instruction}

    Answer: {answer}

    Provide your evaluation in JSON format with the following structure:
    {{
        "accuracy": {{
            "analysis": "...",
            "score": 0
        }},
        "style": {{
            "analysis": "...",
            "score": 0
        }}
    }}
    """
    prompt = re.sub(r"[^\S\n]+", " ", prompt)

    client = InferenceClient(
        provider="nebius",
        api_key=os.environ["HUGGINGFACE_KEY"],
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
    )

    # Delete the "think" part
    text = completion.choices[0].message.content
    end_tag = "</think>"
    think_pos = text.find(end_tag)
    text_after_second_think = text[think_pos + len(end_tag) :]

    return text_after_second_think.strip()


def main() -> None:
    """
    Generates a df with all the evaluations.
    """

    df = pd.read_csv("data/evaluation/answers.csv")
    evaluations = [
        evaluate_answer(instruction, answer)
        for instruction, answer in zip(df["prompt"], df["answer"])
    ]

    accuracies = []
    accs_explanations = []
    styles = []
    styles_explanations = []
    for evaluation in evaluations:
        clean_eval = evaluation.strip("`").strip()
        eval_json = json.loads(clean_eval[4:].strip())  # to delete "json"
        accuracies.append(eval_json["accuracy"]["score"])
        accs_explanations.append(eval_json["accuracy"]["analysis"])
        styles.append(eval_json["style"]["score"])
        styles_explanations.append(eval_json["style"]["analysis"])

    df["accuracy"] = accuracies
    df["style"] = styles
    df["accuracy_explanation"] = accs_explanations
    df["style_explanation"] = styles_explanations

    df.to_csv("data/evaluation/evaluation.csv", index=False)


if __name__ == "__main__":
    main()
