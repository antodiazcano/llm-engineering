"""
Script to perform DPO in the LLM.
"""

from typing import Any
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from transformers import TextStreamer
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from src.sft.constants_dpo import (
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
    PER_DEVICE_EVAL_BATCH_SIZE,
    WEIGHT_DECAY,
    BETA,
    EVAL_STRATEGY,
    EVAL_STEPS,
    LOGGING_STEPS,
    SEED,
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

    def format_samples_dpo(example: LazyRow) -> dict[str, list[str]]:
        """
        Function to format triplets to be available with the DPO training.

        Parameters
        ----------
        examples : HF dataset row.

        Returns
        -------
        Modified text.
        """

        example["prompt"] = ALPACA_TEMPLATE.format(example["prompt"])
        example["rejected"] = example["rejected"] + eos_token
        example["chosen"] = example["chosen"] + eos_token

        return {
            "prompt": example["prompt"],
            "rejected": example["rejected"],
            "chosen": example["chosen"],
        }

    # dataset = load_dataset("mlabonne/FineTome-Alpaca-100k")
    dataset = load_dataset("csv", data_files="data/datasets/df_dpo.csv", split="train")
    print(f"Loaded dataset with {len(dataset)} samples.")

    dataset = dataset.map(format_samples_dpo)
    dataset = dataset.train_test_split(test_size=TEST_SIZE)

    print("Training dataset example:")
    print(dataset["train"][0])

    PatchDPOTrainer()
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        beta=BETA,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        max_length=MAX_SEQ_LENGTH // 2,
        max_prompt_length=MAX_SEQ_LENGTH // 2,
        args=DPOConfig(
            learning_rate=LEARNING_RATE,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim=OPTIM,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=WARMUP_STEPS,
            output_dir=OUTPUT_DIR,
            eval_strategy=EVAL_STRATEGY,
            eval_steps=EVAL_STEPS,
            logging_steps=LOGGING_STEPS,
            report_to="comet_ml",
            seed=SEED,
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
    output_dir: str = "data/models/dpo",
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


def load_model(model_path: str = "data/models/dpo") -> tuple[Any, Any]:
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
