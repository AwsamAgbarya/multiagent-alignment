import os
import torch
import json
import numpy as np
import random
import transformers
import hydra
import logging
import time
from omegaconf import DictConfig
from datasets import load_from_disk, load_dataset, disable_caching
from utils import trim_content, remove_duplicates, add_index
from processing import DataProcessingDebate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed, pipeline, AutoConfig
from tqdm import tqdm
from multiprocess import set_start_method
from vllm import LLM, SamplingParams
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
set_start_method("spawn", force=True)

def generate_answer(answer_context, model = None, tokenizer = None, device = None, temperature = 1, top_p = 0.9):
    """
    Generate a model response from a chat context.

    Args:
        answer_context (list): List of chat messages (dicts with 'role' and 'content').
        model: LLM for text generation.
        tokenizer: Tokenizer for the given LLM.
        temperature (float, optional): Sampling temperature; controls randomness (default=1).
        top_p (float, optional): Top-p (nucleus) sampling threshold (default=0.9).

    Returns:
        dict: response with a single assistant message under 'choices' representing an answer
    """
    input_text = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        max_length=len(input_ids[0]) + 2048,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True, 
        output_scores=True, 
        do_sample=True, 
        top_p=top_p, 
        temperature=temperature
    )
    generated_ids = output.sequences[:, len(input_ids[0]):].squeeze().to("cpu")
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion = {"choices": [{"message": {"role": "assistant", "content": completion}}]}
    return completion


@hydra.main(version_base="1.1", config_path="configs/debate", config_name="config")
def main(cfg: DictConfig):
    # Initialize logging
    logging.basicConfig(level=cfg.logging.level)
    logger = logging.getLogger(__name__)
    logger.info(f"Operating in the following directory:\n {os.getcwd()}")
    logger.info(f"Loading config:\n {cfg.experiment.name}\n{cfg.experiment.description}\n Agents: {cfg.debate.agents}, Rounds: {cfg.debate.rounds}, Summarize:{cfg.debate.summarize}")

    # Configurations
    name = cfg.experiment.name
    agents = cfg.debate.agents
    rounds = cfg.debate.rounds
    summarize = cfg.debate.summarize
    temperature = cfg.model.temperature
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{0 % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    top_p = cfg.model.top_p
    random.seed(0)
    disable_caching()

    # Import dataset
    logger.info(f"Loading dataset from {os.path.join(cfg.dataset.path, cfg.dataset.name)} using only {cfg.dataset.split}")
    dataset = load_from_disk(os.path.join(cfg.dataset.path, cfg.dataset.name))[cfg.dataset.split]
    dataset = dataset.shuffle()
    dataset = dataset.map(add_index, with_indices=True)

    if dataset.num_rows > 50:
        logger.info(f"Dataset has too many rows {dataset.num_rows}, will shrink down to {50}")
        dataset = dataset.filter(lambda item: item["index"] < 50)
    
    logger.info(f"Dataset keys:\n {dataset.column_names}")
    # Transform the dataset into a proper chat template
    data_processor = DataProcessingDebate(config=cfg, cot=False)
    dataset = data_processor.transform(dataset, with_golden_label=False)

    logger.info(f"Dataset keys:\n {dataset.column_names}")

    # Import model 1
    logger.info(f"Loading Model and Tokenizer from {cfg.model.path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    
    # Temporary
    modelConfig = AutoConfig.from_pretrained(cfg.model.path)
    if not hasattr(modelConfig, "parallelization_style") or modelConfig.parallelization_style is None:
        setattr(modelConfig, "parallelization_style", "none")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        torch_dtype=torch.bfloat16,
        config=modelConfig
    ).to(device)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # Import model 2  
    llm = LLM(
        cfg.model.path,
        gpu_memory_utilization=0.3,
        trust_remote_code=True,
    )

    # Iterate for every input
    start_time = time.time()
    for data in tqdm(dataset):
        agent_context = [{"role": "user", "content": data["text"]}]
        completion = generate_answer(agent_context,
                                    model = model,
                                    device=device,
                                    tokenizer = tokenizer,
                                    temperature = temperature,
                                    top_p = top_p)

    print(f"Finished in: {time.time() - start_time}")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    start_time = time.time()
    for data in tqdm(dataset):
        # agent_context = {"role": "user", "content": data["text"]}
        resp = llm.generate(data["text"], sampling_params=SamplingParams(temperature=0.0))
        print(resp[0].outputs[0].text)
    print(f"Finished in: {time.time() - start_time}")

if __name__ == "__main__":
    main()