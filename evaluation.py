# -*- coding: utf-8 -*-
import argparse
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from datasets import load_dataset
import json
import re
from typing import Dict, List, Union
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
from datetime import datetime
import sys
import pandas as pd

# Argument parser setup
parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 models with vLLM on retrosynthesis task")
parser.add_argument('--cuda_device', type=str, default="1", help="CUDA device number(s) to use (e.g., '6' or '0,1').")
parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                   help="Model name or path")
parser.add_argument('--dataset', type=str, default="test_small.jsonl",
                   help="Dataset file path (e.g., 'retrosynthesis_test.jsonl')")
parser.add_argument('--prompt_type', type=str, default='0',
                   help="System prompt type from available prompt types")
parser.add_argument('--best_of', type=int, default=10,
                   help="Number of output sequences to generate per prompt")
parser.add_argument('--n', type=int, default=10,
                   help="Number of top sequences to return (should be <= best_of)")
parser.add_argument('--top_k_metrics', type=str, default='1,3,5,10',
                   help="Comma-separated list of top-k accuracies to report (e.g., '1,3,5,10' or '1')")
parser.add_argument('--batch_size', type=int, default=5100,
                   help="Batch size for vLLM inference")
parser.add_argument('--temperature', type=float, default=1.0,
                   help="temperature")
parser.add_argument('--disable_logging_csv', action='store_true',
                   help='Disable logging to file and CSV output.')

args = parser.parse_args()

# Parse top_k_metrics
try:
    top_k_metrics = [int(k) for k in args.top_k_metrics.split(',')]
    if not all(k > 0 for k in top_k_metrics):
        raise ValueError("All top-k values must be positive integers")
    if max(top_k_metrics) > args.n:
        raise ValueError("All top-k values must be <= n")
except ValueError as e:
    print(f"Invalid top_k_metrics: {e}")
    sys.exit(1)

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

# Model configuration
max_seq_length = 2500
temperature = args.temperature

lora_path = args.model_name
dataset_file = args.dataset
if not dataset_file.endswith('.jsonl'):
    print("Dataset file must end with '.jsonl'")
    sys.exit(1)
dataset_name = dataset_file.replace('.jsonl', '')

# Logging setup
if not args.disable_logging_csv:
    log_dir = f"logs/evaluate_retrosynthesis/{lora_path.replace('/', '_')}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'eval_{dataset_name}_bestof{args.best_of}_topk{args.top_k_metrics}_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        filename=log_file,
        level

=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
else:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False

# Load model and tokenizer
print(f"Loading model: {lora_path}")
if not args.disable_logging_csv:
    logger.info(f"Loading model: {lora_path}")

model = LLM(
    model=lora_path,
    max_model_len=max_seq_length,
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)

# Load dataset
dataset_path = f"data/{dataset_file}"
dataset = load_dataset("json", data_files=dataset_path)["train"]

# System prompts
system_prompt_dict = {
    "0": """
        Respond in the following format:
        <answer>
        Provide the final answer in JSON format as specified in the instruction.
        </answer>
        """,
    "1-plan": """
        Respond in the following format:
        <plan>
        ...
        </plan>
        <answer>
        Provide the final answer in JSON format as specified in the instruction.
        </answer>
        """,
    'plan-reason':"""
        Respond in the following format:
        <plan>
        Provide step-by-step plan to solve the task based on the given instructions and product SMILES.
        </plan>
        <reason>
        Conduct your detail reasoning.
        </reason>
        <answer>
        Provide the reactants in SMILES format as a JSON object: {"reactants": "SMILES_string"}
        </answer>
        """,
    "reason-think": """
        Respond in the following format:
        <Plan>
        Provide step-by-step reasoning to predict the reactants from the given product SMILES.
        </plan>
        <think>
        Explain the key chemical transformations or patterns identified in the product that lead to the predicted reactants.
        </think>
        <answer>
        Provide the reactants in SMILES format as a JSON object: {"reactants": "SMILES_string"}
        </answer>

        Example:
        Product SMILES: CC(=O)Nc1ccccc1
        <plan>
        1. The product is an amide, suggesting an amide bond formation.
        2. A common retrosynthetic disconnection for amides involves an amine and an acyl chloride.
        3. The phenyl group (c1ccccc1) and the acetyl group (CC(=O)) suggest aniline (c1ccccc1NH2) and acetyl chloride (CC(=O)Cl) as reactants.
        </plan>
        <think>
        The amide bond is formed via nucleophilic acyl substitution, where the amine (aniline) attacks the acyl chloride (acetyl chloride), releasing HCl.
        </think>
        <answer>
        {"reactants": "c1ccccc1NH2.CC(=O)Cl"}
        </answer>
        """
}

# Validate prompt type
if args.prompt_type not in system_prompt_dict:
    print(f"Invalid prompt type. Available types: {list(system_prompt_dict.keys())}")
    sys.exit(1)

SYSTEM_PROMPT = system_prompt_dict[args.prompt_type]

# Prompt formatting function
def format_retrosynthesis_prompt(example: Dict) -> Dict:
    product = example["product"]
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
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "expected": {"reactants": example["reactants"]}
    }

# Prepare dataset
eval_dataset = dataset.map(format_retrosynthesis_prompt, batched=False)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=temperature,
    best_of=args.best_of,
    n=args.n,
    max_tokens=2000,  # Increased to handle complex SMILES and reasoning
)

# Utility functions
def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_xml_think(text: str) -> str:
    if args.prompt_type == "reason-think":
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        output = ""
        if reasoning_match:
            output += f"Reasoning: {reasoning_match.group(1).strip()}"
        if think_match:
            output += f"\nThink: {think_match.group(1).strip()}" if output else f"Think: {think_match.group(1).strip()}"
        return output
    return ""

# Basic SMILES validation (placeholder without RDKit)
def is_valid_smiles(smiles: Union[str, list]) -> bool:
    # Handle case where input is a list
    if isinstance(smiles, list):
        return any(is_valid_smiles(s) for s in smiles if isinstance(s, str))
    # Handle string input
    if not isinstance(smiles, str):
        return False
    if not smiles or smiles.strip() == "":
        return False
    return True
    # # Extended set to include atom mapping, stereochemistry, and common SMILES characters
    # allowed_chars = set("CcNnOoSsPpClBrFI=()[]:0123456789@#+-./\\{}|%")
    # return all(c in allowed_chars for c in smiles.replace(" ", ""))

# Compute top-k accuracy
def compute_top_k_accuracy(expected: str, predicted_list: List[str], k: int) -> bool:
    valid_predictions = [p for p in predicted_list if is_valid_smiles(p)]
    return expected in valid_predictions[:k]

# Evaluation function
def evaluate_dataset(dataset, dataset_name: str, prompt_type: str, batch_size: int):
    total_samples = len(dataset)
    top_k_correct = {k: 0 for k in top_k_metrics}
    top_k_accuracies = {k: [] for k in top_k_metrics}

    print(f"Evaluating {total_samples} examples with best_of={args.best_of}, n={args.n}, top_k_metrics={args.top_k_metrics} (batch size: {batch_size})")
    if not args.disable_logging_csv:
        logger.info(f"Evaluating {total_samples} examples with best_of={args.best_of}, n={args.n}, top_k_metrics={args.top_k_metrics} (batch size: {batch_size})")

    prompts = [
        tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )
        for example in dataset
    ]
    expected_outputs = [example["expected"] for example in dataset]

    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_prompts = prompts[batch_start:batch_end]
        batch_expected = expected_outputs[batch_start:batch_end]

        print(f"Processing batch {batch_start // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}...")
        if not args.disable_logging_csv:
            logger.info(f"Processing batch {batch_start // batch_size + 1}/{(total_samples + batch_size - 1) // batch_size}...")

        batch_outputs = model.generate(
            batch_prompts,
            sampling_params=sampling_params,
        )

        for i, output in enumerate(batch_outputs):
            sample_idx = batch_start + i + 1
            expected = batch_expected[i]["reactants"]
            all_generations = output.outputs
            predicted_reactants_list = []
            best_output_text = ""
            best_gen_idx = -1
            raw_outputs = []

            for gen_idx, generation_output in enumerate(all_generations):
                output_text = generation_output.text
                raw_outputs.append(output_text)
                predicted_answer = extract_xml_answer(output_text)
                predicted_reactants = ""
                try:
                    predicted_json = json.loads(predicted_answer) if predicted_answer else {}
                    predicted_reactants = predicted_json.get("reactants", "") if isinstance(predicted_json, dict) else ""
                except json.JSONDecodeError as e:
                    if not args.disable_logging_csv:
                        logger.warning(f"Sample {sample_idx}, Gen {gen_idx+1}: Error parsing JSON: {e}, Raw output: {output_text}")
                    predicted_reactants = ""  # Set to empty string on JSON error
                if is_valid_smiles(predicted_reactants):
                    predicted_reactants_list.append(predicted_reactants)
                else:
                    predicted_reactants_list.append("")
                    if not args.disable_logging_csv:
                        logger.warning(f"Sample {sample_idx}, Gen {gen_idx+1}: Invalid SMILES: {predicted_reactants}")
                # Select the best output: prefer exact match, then first valid SMILES, then first non-empty output
                if predicted_reactants == expected:
                    best_output_text = output_text
                    best_gen_idx = gen_idx
                elif not best_output_text and is_valid_smiles(predicted_reactants):
                    best_output_text = output_text
                    best_gen_idx = gen_idx
                elif not best_output_text and predicted_answer:
                    best_output_text = output_text
                    best_gen_idx = gen_idx

            # If no valid output was selected, pick the first output as fallback
            if not best_output_text and raw_outputs:
                best_output_text = raw_outputs[0]
                best_gen_idx = 0

            # Compute top-k accuracies for specified k values
            for k in top_k_metrics:
                is_correct = compute_top_k_accuracy(expected, predicted_reactants_list, k)
                top_k_correct[k] += int(is_correct)
                top_k_accuracies[k].append(1.0 if is_correct else 0.0)

            # Log sample details
            if not args.disable_logging_csv:
                sentence = "N/A"
                content = dataset[batch_start + i]['prompt'][1]['content']
                sentence_match = re.search(r"Given the product SMILES: \"(.*?)\"\s*Predict", content, re.DOTALL)
                if sentence_match:
                    sentence = sentence_match.group(1).strip()
                logger.info(f"{dataset_name} Sample {sample_idx}/{total_samples} (best_of={args.best_of}, n={args.n}, top_k_metrics={args.top_k_metrics})")
                logger.info(f"Product SMILES: {sentence}")
                logger.info(f"Best Generation (Index {best_gen_idx+1}) Think/Reasoning:\n---\n{extract_xml_think(best_output_text)}\n---")
                logger.info(f"Raw Outputs:\n---\n{json.dumps(raw_outputs, indent=2)}\n---")
                logger.info(f"Predicted Reactants: {json.dumps(predicted_reactants_list)}")
                logger.info(f"Expected Reactants: {expected}")
                for k in top_k_metrics:
                    logger.info(f"Top-{k} Correct: {compute_top_k_accuracy(expected, predicted_reactants_list, k)}")
                logger.info("---")

    # Compute aggregate metrics
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'prompt_type': prompt_type,
        'dataset': dataset_name,
        'model': lora_path,
        'best_of': args.best_of,
        'n': args.n,
        'top_k_metrics': args.top_k_metrics,
    }

    print(f"{dataset_name} Evaluation Results (best_of={args.best_of}, n={args.n}, top_k_metrics={args.top_k_metrics}):")
    for k in top_k_metrics:
        accuracy = top_k_correct[k] / total_samples if total_samples > 0 else 0.0
        avg_accuracy = sum(top_k_accuracies[k]) / len(top_k_accuracies[k]) if top_k_accuracies[k] else 0.0
        print(f"Top-{k} Accuracy: {accuracy:.4f}")
        results[f'top{k}_accuracy'] = accuracy
        if not args.disable_logging_csv:
            logger.info(f"Top-{k} Accuracy: {accuracy:.4f}")

    if not args.disable_logging_csv:
        logger.info(f"{dataset_name} Aggregate Results (best_of={args.best_of}, n={args.n}, top_k_metrics={args.top_k_metrics}):")
        for k in top_k_metrics:
            logger.info(f"Top-{k} Accuracy: {results[f'top{k}_accuracy']:.4f}")

        results_dir = f"/export/data/rli/Project/llms/retro/logs/results/results_temp_{temperature}"
        os.makedirs(results_dir, exist_ok=True)
        csv_file = os.path.join(results_dir, f"{dataset_name}_bestof{args.best_of}_topk{args.top_k_metrics}_evaluation_results.csv")

        df = pd.DataFrame([results])
        if os.path.exists(csv_file):
            try:
                existing_df = pd.read_csv(csv_file)
                for col in df.columns:
                    if col not in existing_df.columns:
                        existing_df[col] = None
                df = df[existing_df.columns]
                df.to_csv(csv_file, mode='a', header=False, index=False)
                print(f"Results appended to {csv_file}")
                logger.info(f"Results appended to {csv_file}")
            except Exception as e:
                logger.error(f"Error appending to CSV {csv_file}: {e}")
                print(f"Error appending to CSV {csv_file}: {e}")
                csv_file_fallback = os.path.join(results_dir, f"{dataset_name}_bestof{args.best_of}_topk{args.top_k_metrics}_evaluation_results_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(csv_file_fallback, index=False)
                print(f"Saved results to fallback CSV: {csv_file_fallback}")
                logger.info(f"Saved results to fallback CSV: {csv_file_fallback}")
        else:
            df.to_csv(csv_file, index=False)
            print(f"Results saved to {csv_file}")
            logger.info(f"Results saved to {csv_file}")

        print(f"Full log saved to {log_file}")
        logger.info(f"Full log saved to {log_file}")
    else:
        print("Logging and CSV saving are disabled.")

# Note: For production, consider using RDKit for SMILES validation and canonicalization:
# from rdkit import Chem
# def canonicalize_smiles(smiles):
#     try:
#         mol = Chem.MolFromSmiles(smiles)
#         return Chem.MolToSmiles(mol, canonical=True) if mol else ""
#     except:
#         return ""

# Run evaluation
evaluate_dataset(
    eval_dataset,
    dataset_name.capitalize(),
    prompt_type=args.prompt_type,
    batch_size=args.batch_size
)