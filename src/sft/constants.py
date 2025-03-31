"""
Script to define the hyperparameters used in fine-tuning.
"""

# LoRA/QLoRA
MODEL_NAME = "meta-llama/Llama-3.1-8B"
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
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response \
that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

# Tokenizer
CHAT_TEMPLATE = "chatml"

# Training
LEARNING_RATE = 3e-4
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
TEST_SIZE = 0.2
OPTIM = "adamw_8bit"
LR_SCHEDULER_TYPE = "linear"
WARMUP_STEPS = 10
OUTPUT_DIR = "data/models"
