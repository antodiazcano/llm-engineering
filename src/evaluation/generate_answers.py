"""
This script contains the functions to generate responses from the fine-tuned models to
evaluate them after the generation process.
"""

import os
import re
from typing import Callable
import pandas as pd
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from huggingface_hub import InferenceClient


def generate_answers(
    model_function: Callable, dataset_path: str = "data/datasets/df_sft.csv"
) -> list[tuple[str, str]]:
    """
    Generates answers that will be evaluated (using another function).

    Parameters
    ----------
    model_function : Function to generate the model response given a prompt.
    dataset_path   : Path where the dataset is allocated.

    Returns
    -------
    List with the prompts and answers of the test.
    """

    def format_sample(sample: LazyRow) -> str:
        """
        Formats an instruction.

        Parameters
        ----------
        sample : HF dataset row.

        Returns
        -------
        Formatted row.
        """

        text = f"""Below is an instruction that describes a task. Write a response \
        that appropriately completes the request.

        ### Instruction:
        {sample["instructions"]}

        ### Response:
        """

        return re.sub(r"[^\S\n]+", " ", text)  # delete more than one blank space

    dataset = load_dataset("csv", data_files=dataset_path, split="train")
    dataset = dataset.map(lambda sample: {"prompt": format_sample(sample)})
    dataset = dataset.train_test_split(test_size=0.01)
    dataset = dataset["test"]

    return [(prompt, model_function(prompt)) for prompt in dataset["prompt"]]


def generate_deepseek_response(prompt: str) -> str:
    """
    Generates a response of DeepSeek.

    Parameters
    ----------
    prompt : Prompt.

    Returns
    -------
    Response of the model.
    """

    client = InferenceClient(
        provider="nebius",
        api_key=os.environ["HUGGINGFACE_KEY"],
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=[{"role": "user", "content": prompt}],
    )
    text = completion.choices[0].message.content

    # Delete the "think" part
    end_tag = "</think>"
    think_pos = text.find(end_tag)
    text_after_second_think = text[think_pos + len(end_tag) :]

    return text_after_second_think.strip()


def main() -> None:
    """
    Creates the answers for the evaluation of all the models.
    """

    model_functions = [generate_deepseek_response]
    model_names = ["deepseek"]

    df = pd.DataFrame()
    for model_function, model_name in zip(model_functions, model_names):
        prompts, answers = zip(*generate_answers(model_function))
        temp_df = pd.DataFrame(
            {"model": [model_name] * len(answers), "prompt": prompts, "answer": answers}
        )
        df = pd.concat([df, temp_df])

    df.to_csv("data/evaluation/answers.csv", index=False)


if __name__ == "__main__":
    main()
