# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 4
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import json
import re
from typing import Dict, List
import uuid
from prompt_background import retrosynthesis_background

# Model configuration
max_seq_length = 2500
train_epochs = 3
per_device_train_batch_size = 24
lora_rank = 32
lora_alpha = 64

qwen_models = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "unsloth/Qwen2.5-Coder-14B-Instruct",
    "unsloth/Qwen2.5-Coder-7B",
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Qwen2.5-32B-Instruct",
    "unsloth/Qwen2.5-72B-Instruct",
]
model_name = qwen_models[1]  # Select Qwen2.5-7B-Instruct
load_in_4bit = '4bit' in model_name
dtype = torch.bfloat16
dataset_name = '_no'
dataset_name = ''
# Load and split the dataset
dataset = load_dataset("json",
                       data_files={"train": f"data/train{dataset_name}_raw.jsonl", "test": f"data/validation{dataset_name}_raw.jsonl"})
# train_val_dataset = dataset["train"].train_test_split(test_size=0.0, seed=42)  # 90% train, 10% val
train_dataset = dataset["train"]
val_dataset = dataset["test"]
# test_dataset = dataset["test"]
length = len(train_dataset)
max_steps = length // per_device_train_batch_size * train_epochs

# Background information for retrosynthesis prompts (adapted from evaluation code)
retrosynthesis_background = retrosynthesis_background

# System prompt for reasoning and format (aligned with evaluation code)
SYSTEM_PROMPT = """
Respond in the following format:
<plan>
Provide step-by-step plan.
</plan>
<answer>
Provide the final answer in JSON format as specified in the instruction.
</answer>
"""
SYSTEM_PROMPT = """
Respond in the following format:
<answer>
Provide the final answer in JSON format as specified in the instruction.
</answer>
"""
# Prompt formatting function for retrosynthesis
response_prompt = """<plan>
To predict the reactants for the product SMILES:
1. Identify key functional groups and structural features in the product.
2. Propose retrosynthetic disconnections based on common reaction types (e.g., esterification, amide formation, sulfonamide formation, heterocycle synthesis).
3. Validate that the proposed reactants are chemically feasible and can form the product under standard conditions.
</plan>
"""

def format_retrosynthesis_prompt(example: Dict) -> Dict:
    product = example["product"]
    reactants = example["reactants"]

    #prompt = f"""{retrosynthesis_background}
    prompt = f"""

Given the product SMILES: "{product}"

Predict the reactants required to synthesize this product.

### Instruction:
- Think step-by-step to identify the reactants based on the product SMILES.
- Consider common retrosynthetic disconnections and reaction types (e.g., amide formation, esterification, nucleophilic substitution).
- Ensure the SMILES string is valid, includes atom mapping if present in the product, and uses '.' to separate multiple reactants.
- Return the predicted reactants in SMILES format as a JSON object:
  {{"reactants": "SMILES_string"}}.
"""

    response_dict = {"reactants": reactants}
    response = f"""<answer>
{json.dumps(response_dict, ensure_ascii=False)}
</answer>"""
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"conversations": convo, "task": "retrosynthesis"}

# Prepare dataset
train_retrosynthesis_dataset = train_dataset.map(format_retrosynthesis_prompt, batched=False)
val_retrosynthesis_dataset = val_dataset.map(format_retrosynthesis_prompt, batched=False)

# Load the Qwen2.5-7B-Instruct model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    fast_inference=True,
    local_files_only=True,
)

# Apply LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_alpha,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Set up chat template for Qwen-2.5
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Apply chat template to datasets and extract response part for training
def extract_response(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text  # Fallback to full text if no <answer> tag

train_retrosynthesis_dataset = train_retrosynthesis_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=False),
        "response": extract_response(x["conversations"][2]["content"])  # Extract only the <answer> part
    },
    batched=False,
    desc="Applying chat template to train dataset"
)
val_retrosynthesis_dataset = val_retrosynthesis_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(x["conversations"], tokenize=False, add_generation_prompt=False),
        "response": extract_response(x["conversations"][2]["content"])  # Extract only the <answer> part
    },
    batched=False,
    desc="Applying chat template to val dataset"
)

# Verify dataset structure
print("Sample from train dataset:", train_retrosynthesis_dataset[0])

# Training setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_retrosynthesis_dataset,
    eval_dataset=val_retrosynthesis_dataset,
    dataset_text_field="text",  # Full conversation for context
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=int(0.07 * max_steps),
        num_train_epochs=train_epochs,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=int(0.02 * max_steps),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=f"loras/{model_name}_retrosynthesis_sft_noplan",
        report_to="tensorboard",
        evaluation_strategy="steps",
        eval_steps=int(0.1 * max_steps),
        save_strategy="steps",
        save_steps=int(0.1 * max_steps),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
)

# Train only on assistant responses, aligning with Qwen-2.5 template
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",  # Start of user prompt
    response_part="<|im_start|>assistant\n",  # Start of assistant response
)

# Train the model
trainer_stats = trainer.train()

saved_path = f'sft/{model_name}_sft_retrosynthesis_{lora_rank}_{train_epochs}'
# Save the LoRA adapters
model.save_pretrained(f"loras/{saved_path}")
tokenizer.save_pretrained(f"loras/{saved_path}")

# Merge and save the model in 16-bit format for vLLM/Unsloth compatibility
model.save_pretrained_merged(
    f"merged_models/{saved_path}",
    tokenizer,
    save_method="merged_16bit",
)

print(f"Training completed. LoRA saved to loras/{saved_path}")
print(f"Merged 16-bit model saved to merged_models/{saved_path}")