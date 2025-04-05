"""
Script to perform fine-tuning in the LLM.
"""

from typing import Any
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from peft import PeftModel

from src.sft.constants_sft import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LOAD_IN_4_BIT,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    ALPACA_TEMPLATE,
    OUTPUT_DIR,
    CHAT_TEMPLATE,
    LEARNING_RATE,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    TEST_SIZE,
    OPTIM,
    LR_SCHEDULER_TYPE,
    WARMUP_STEPS,
)


def _load_model() -> tuple[Any, Any]:
    """
    Loads the model and tokenizer we will fine-tune.

    Returns
    -------
    Model and tokenizer to fine-tune.
    """

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4_BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    return model, tokenizer


def finetune() -> tuple[Any, Any]:
    """
    Fine-tunes the model.

    Returns
    -------
    Fine-tuned model and tokenizer.
    """

    model, tokenizer = _load_model()
    eos_token = tokenizer.eos_token
    print(f"Setting EOS_TOKEN to {eos_token}")

    def format_samples_sft(example: LazyRow) -> dict[str, list[str]]:
        """
        Function to include EOS token at the end of the instruction and output.

        Parameters
        ----------
        example : HF dataset row.

        Returns
        -------
        Modified text.
        """

        text = []

        for instruction, output in zip(
            example["instructions"], example["answers"], strict=False
        ):
            message = ALPACA_TEMPLATE.format(instruction, output) + eos_token
            text.append(message)

        return {"text": text}

    # dataset = load_dataset("mlabonne/FineTome-Alpaca-100k")
    dataset = load_dataset("csv", data_files="data/datasets/df_sft.csv", split="train")
    print(f"Loaded dataset with {len(dataset)} samples.")

    dataset = dataset.map(
        format_samples_sft, batched=True, remove_columns=dataset.column_names
    )
    dataset = dataset.train_test_split(test_size=TEST_SIZE)

    print("Training dataset example:")
    print(dataset["train"][0])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=LEARNING_RATE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=OPTIM,
            weight_decay=0.01,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            warmup_steps=WARMUP_STEPS,
            output_dir=OUTPUT_DIR,
            report_to="comet_ml",
            seed=0,
        ),
    )

    trainer.train()

    return model, tokenizer


def inference(
    model: Any,
    tokenizer: Any,
    prompt: str = "Write a paragraph to introduce supervised fine-tuning.",
    max_new_tokens: int = 256,
) -> None:
    """
    Generates a response of the model.

    Parameters
    ----------
    model          : LLM.
    tokenizer      : Tokenizer.
    prompt         : Prompt.
    max_new_tokens : Maximum number of tokens generated.
    """

    model = FastLanguageModel.for_inference(model)
    message = ALPACA_TEMPLATE.format(prompt, "")
    inputs = tokenizer([message], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs, streamer=text_streamer, max_new_tokens=max_new_tokens, use_cache=True
    )


def save_model(
    model: Any,
    tokenizer: Any,
    output_dir: str = "data/models/sft",
) -> None:
    """
    Saves the model.

    Parameters
    ----------
    model      : Model.
    tokenizer  : Tokenizer.
    output_dir : Directory where the model weights will be saved.
    """

    model.save_pretrained_merged(
        output_dir, tokenizer, save_method="lora"
    )  # "merged_16bit")


def load_model(model_path: str = "data/models/sft") -> tuple[Any, Any]:
    """
    Loads the fine-tuned model.

    Parameters
    ----------
    model_path : Path where the weights of the model are saved.

    Returns
    -------
    Fine-tuned model and tokenizer.
    """

    base_model, trained_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4_BIT,
    )
    trained_model = PeftModel.from_pretrained(base_model, model_path)

    return trained_model, trained_tokenizer


if __name__ == "__main__":
    ft_model, ft_tokenizer = finetune()
    inference(ft_model, ft_tokenizer)
    save_model(ft_model, ft_tokenizer)
    saved_model, saved_tokenizer = load_model()
