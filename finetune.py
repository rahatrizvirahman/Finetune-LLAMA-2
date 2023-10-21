import os
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

import random
import pandas as pd


from utils import *

# Reproducibility
seed = 42
set_seed(seed)

dataset = load_dataset("csebuetnlp/squad_bn")
train_dataset = dataset['train']

print(f"Number of prompts: {len(train_dataset)}")
print(f"Column names are: {train_dataset.column_names}")

# Load model from HF with user's token and with bitsandbytes config
model_name = "meta-llama/Llama-2-7b-hf"
bnb_config = create_bnb_config()
model, tokenizer = load_model(model_name, bnb_config)

## Preprocess dataset
max_length = get_max_length(model)
dataset = preprocess_dataset(tokenizer, max_length, seed, train_dataset)

output_dir = "./results/llama2/final_checkpoint"
train(model, tokenizer, dataset, output_dir)

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()

output_merged_dir = "./results/llama2/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)

# Specify input
text = "আকাশে পাখি উড়ে।"

# Specify device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt").to(device)

# Get answer
# (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
outputs = model.generate(
    input_ids=inputs["input_ids"].to(device),
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode output & print it
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# Save and push model to hugging face
# model.push_to_hub("llama2-fine-tuned-dolly-15k")


# tokenizer.push_to_hub("llama2-fine-tuned-dolly-15k")

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("<username>/llama2-fine-tuned-dolly-15k")
# model = AutoModelForCausalLM.from_pretrained("<username>/llama2-fine-tuned-dolly-15k")
