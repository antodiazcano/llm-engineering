"""
Script to define the hyperparameters used in DPO.
"""

import re


# LoRA/QLoRA
MODEL_NAME = "mlabonne/TwinLlama-3.1-8B"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4_BIT = True  # True QLoRA, False LoRA
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "up_proj",
    "down_proj",
    "o_proj",
    "gate_proj",
]
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a \
response that appropriately completes the request.

###
Instruction:
{}

###
Response:
"""
ALPACA_TEMPLATE = re.sub(r"[^\S\n]+", " ", ALPACA_TEMPLATE)

# Tokenizer
CHAT_TEMPLATE = "chatml"

# Training
LEARNING_RATE = 2e-6
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
TEST_SIZE = 0.1
OPTIM = "adamw_8bit"
LR_SCHEDULER_TYPE = "linear"
WARMUP_STEPS = 10
OUTPUT_DIR = "models"
BETA = 0.5
WEIGHT_DECAY = 0.01
EVAL_STRATEGY = "steps"
OUTPUT_DIR = "models"
EVAL_STEPS = 0.2
LOGGING_STEPS = 1
SEED = 0
