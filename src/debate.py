import os
import torch
import torch.distributed as dist
import json
import numpy as np
import random
import transformers
import hydra
import logging
from omegaconf import DictConfig
from datasets import load_from_disk, disable_caching
from torch.utils.data import DataLoader
from utils import add_index
from processing import DataProcessingDebate
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from multiprocess import set_start_method
from vllm import LLM, SamplingParams
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_CONFIGURE_LOGGING"]="0"
set_start_method("spawn", force=True)


class Debate():

    def __init__(self, cfg, logger):
        self.agents = cfg.debate.agents
        self.rounds = cfg.debate.rounds
        self.summarize = cfg.debate.summarize
        self.temperature = cfg.model.temperature
        self.enable_vllm = cfg.training.vllm
        self.model_path = cfg.model.path
        self.logger = logger
        self.max_len = cfg.training.max_seq_length
        self.top_p = cfg.model.top_p
        self.summarize = cfg.debate.summarize
        if self.enable_vllm:
            self.pre_alloc_mem=cfg.training.vllm_mem
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{0 % torch.cuda.device_count()}")
        else:
            self.device = torch.device("cpu")
        if self.device == '':
            self.device = torch.device("cpu")
        
        print(f"Device is:\n{self.device}")
        self.load_models()

    def load_models(self):
        # Import tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.tokenizer.padding_side = "right"

        # Import model from VLLM or original HF
        if self.enable_vllm:
            self.logger.info("==============================================================================")
            self.logger.info(f"Loading Model with vLLM from {self.model_path}")
            self.model = LLM(
                model=self.model_path,
                gpu_memory_utilization=self.pre_alloc_mem,
                trust_remote_code=True,
                dtype="bfloat16"
            )
        else:
            # Temporary
            self.logger.info("==============================================================================")
            self.logger.info(f"Loading Original Model and Tokenizer from {self.model_path}")
            modelConfig = AutoConfig.from_pretrained(self.model_path)
            if not hasattr(modelConfig, "parallelization_style") or modelConfig.parallelization_style is None:
                setattr(modelConfig, "parallelization_style", "none")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                config=modelConfig
            ).to(self.device)
            self.model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=8)

    def generate(self, dataset, name, summarize=False):
        '''
        Takes in a preprocessed dataset that has text attribute which holds the prompt, and token_count which holds the token count of the text.
        Generates a json file with the generated description for each input.
        '''
        generated_description = {}
        # Iterate for every input
        for data in tqdm(dataset):
            agents_contexts = [[{"role": "user", "content": data["text"]}] for __ in range(self.agents)]
            for round in range(self.rounds):
                for i, agent_contexts in enumerate(agents_contexts):
                    if round!=0:
                        agent_contexts_other = agents_contexts[:i] + agents_contexts[i+1:]
                        random.shuffle(agent_contexts_other)
                        if summarize:
                            # Summarize 5 random agent contexts as input
                            summary = self.summarize_message(agent_contexts_other[:5])
                            message = construct_message_summary(summary, data["question"])
                        else:
                            # List 5 random agent contexts as input
                            message = construct_message(agent_contexts_other[:5], data["question"], 2 * round - 1)
                        agent_contexts.append(message)
                    
                    completion = self.generate_answer(agent_contexts)
                    assistant_message = construct_assistant_message(completion)
                    agent_contexts.append(assistant_message)
            generated_description[data['index']] = (agents_contexts,  data["answer"])
        json.dump(generated_description, open(f"{name}.json", "w"))
        self.logger.info(f"Dataset saved to {os.getcwd()}")
    
    def generate_answer(self, answer_context):
        input_text = self.tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        if self.enable_vllm:
            sampling_params = SamplingParams(
                max_tokens = len(encoded['input_ids'][0])+self.max_len,
                temperature = self.temperature,
                top_p = self.top_p,
                stop=["</s>"]
            )
            output = self.model.generate(input_text, sampling_params=sampling_params)[0]
            completion = {"choices": [{"messages": {"role": "assistant", "content": output.outputs[0].text}}]}
        else:
            encoded.to(self.device)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            output = self.model.generate(
                input_ids, 
                attention_mask=attention_mask,
                max_length=input_ids.shape[1]+self.max_len,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True, 
                output_scores=True, 
                do_sample=True, 
                top_p=self.top_p, 
                temperature=self.temperature
            )
            generated_ids = output.sequences[:, len(input_ids[0]):].squeeze().to("cpu")
            completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion = {"choices": [{"message": {"role": "assistant", "content": completion}}]}
        return completion

    def generate_batched(self, input_dataset, batch_size, name):
        '''
        Takes in a preprocessed dataset that has text attribute which holds the prompt, and token_count which holds the token count of the text.
        Unlike generate, this function takes in a DataLoader object with batch_size, and tries to pass them in batches to the model.
        Each pass will hold a list of contexts in the shape N_agents x batch_size x N_messages, each entry holds user, content and len.
        Generates a json file with the generated description for each input.
        '''
        generated_description = {}
        dataset = DataLoader(input_dataset, batch_size=batch_size)
        # Iterate for every input
        for data in tqdm(dataset):
            # N_agents x batch_size x N_messages
            agents_contexts = [[[{"role": "user", 'content':data['text'][i], 'len':data['token_count'][i].item()}] for i in range(len(data['index']))] for __ in range(self.agents)]
            # print(f"Agent contexts: \n{agents_contexts}")
            for round in range(self.rounds):
                for i, agent_contexts in enumerate(agents_contexts):
                    if round!=0:
                        agent_contexts_other = agents_contexts[:i] + agents_contexts[i+1:]
                        random.shuffle(agent_contexts_other)
                        if self.summarize:
                            # Summarize 5 random agent contexts as input
                            # print(f"Trying to summarize \n{agent_contexts_other[:5]}")
                            summary = self.summarize_message(agent_contexts_other[:5],round,in_batches=True) # Array of string summaries
                            # print(f"Recieved summary: \n{summary}")
                            messages = construct_message_summary(summary, data["question"], in_batches=True) # Array of dictionary summaries
                            # print(f"Constructed message: \n{messages}")
                        else:
                            # List 5 random agent contexts as input
                            messages = construct_message(agent_contexts_other[:5], data["question"], 2 * round - 1, in_batches=True) # Array of dictonary messages
                        for j,agent in enumerate(agent_contexts):
                            agent.append(messages[j]) # Append each entry in the batch's summary message to that batch's context
                            # print(f"Current agent context: \n{agent}")
                    completion = self.generate_answer_batched(agent_contexts)
                    assistant_message = construct_assistant_message(completion, in_batches=True)
                    for j, batch_context in enumerate(agent_contexts):
                        batch_context.append(assistant_message[j])
                        # print(f"Added \n{assistant_message[j]}\n of {j}th place to agent context {j}\n{agent_contexts}")
            for j, batch_context in enumerate(agents_contexts):
                generated_description[data['index'][j].item()] = (batch_context,  data["answer"][j])
                # print(f"Added to generated description for index {data['index'][j].item()}: \n{batch_context} with answer {data['answer'][j]} of place {j}")
        json.dump(generated_description, open(f"{name}.json", "w"))
        self.logger.info(f"Dataset saved to {os.getcwd()}")

    def generate_answer_batched(self, answer_context):
        '''
        Takes in a list of contexts in the shape batch_size x N_messages, each entry holds user, content and len.
        Passes it to a model to run in one forward pass using vllm.
        Returns a list of completions in the shape batch_size x 1, which holds an answer to each prompt.
        '''
        input_text = self.tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        if self.enable_vllm:
            # if len is an existing attribute in the context
            if 'len' in answer_context[0][0]:
                input_length = max([context[0]['len']+self.max_len for context in answer_context])
            else:
                encoded = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                # print(encoded['input_ids'].shape)
                input_length = encoded['input_ids'].shape[1] + self.max_len
            sampling_params = SamplingParams(
                max_tokens = input_length,
                temperature = self.temperature,
                top_p = self.top_p,
                stop=["</s>"]
            )
            outputs = self.model.generate(input_text, sampling_params=sampling_params)
            completion = [{"choices": [{"messages": {"role": "assistant", "content": output.outputs[0].text}}]} for output in outputs]
        else:
            self.logger.info("Cannot batch this input without VLLM enabled")
        return completion
        
    def sample(self, input_dataset, batch_size, name):
        # print(input_dataset)
        input_dataset = [{"index": b['index'], "text":b['text'], "token_count":b['token_count']} for b in input_dataset for __ in range(batch_size)]
        self.generate_batched(input_dataset, batch_size, name)

    def summarize_message(self, agent_contexts, round=0, in_batches=False):
        prefix_string = "Here are a list of opinions from different agents: "

        if in_batches:
            batched_summary = []
            for i in range(len(agent_contexts[0])):
                for agent in agent_contexts:
                    # print(f"For rouund {round} we are taking the {(2*round)-1}th message")
                    agent_response = agent[i][(2*round)-1]["content"]
                    response = "\n\n One agent response: ```{}```".format(agent_response)
                    prefix_string = prefix_string + response
                summary = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent and explain the reasoning in each solution."
                batched_summary.append([{"role": "user", "content": summary}])
            # print("From summary")
            completion = self.generate_answer_batched(batched_summary)
            content = []
            for complet in completion:
                content.append(complet["choices"][0]["messages"]["content"])
        else:
            for agent in agent_contexts:
                agent_response = agent[-1]["content"]
                response = "\n\n One agent response: ```{}```".format(agent_response)
                prefix_string = prefix_string + response

            prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent and explain the reasoning in each solution."
            agent_context = [{"role": "user", "content": prefix_string}]
            completion = self.generate_answer(agent_context)
            content = completion["choices"][0]["message"]["content"]

        return content

def construct_message(agents, prefix, idx, in_batches=False):
    '''
    Takes in a list of agents each with a list of batches with a list of messages, and a list of questions.
    Tries to create one string that contains all the messages from all agents per batch, and then appends a question to the end of the string.
    '''
    if in_batches:
        batched_summary = []
        for i in range(len(agents[0])):
            prefix_string = "Here is are solution from other agents: "
            for agent in agents:
                agent_response = agent[i][idx]["content"]
                response = "\n\n One agent response: {}".format(agent_response)

                prefix_string = prefix_string + response
            prefix_string = prefix_string + "\n\n Using each response as additional advice, can you give an updated bullet by bullet answer to {}? Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix[i])
            batched_summary.append({"role": "user", "content": prefix_string})
        return batched_summary
    else:
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct? Please reiterate your answer, with your final answer a single answer of the form \\boxed{{answer}} at the end of your response.".format(prefix)}

        prefix_string = "Here is are solution from other agents: "

        for agent in agents:
            agent_response = agent[idx]["content"]
            response = "\n\n One agent response: {}".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + "\n\n Using each response as additional advice, can you give an updated bullet by bullet answer to {}? Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix)
        return {"role": "user", "content": prefix_string}

def construct_message_summary(summary, prefix, in_batches=False):
    '''
    Takes either a single summary or a list of summaries, alongside a question or a list of questions and formats them into a message for the user.
    '''
    if in_batches:
        summary_content = []
        for i,item in enumerate(summary):
            prefix_string = "Here is a summary of solutions from several other agents: {}".format(item)
            prefix_string = prefix_string + "\n\n Examine each these solutions as additional advice, can solve {} and give your updated answer? Explain your reasoning. \n Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix[i])
            summary_content.append({"role": "user", "content": prefix_string})
        return summary_content
    else:
        prefix_string = "Here is a summary of solutions from several other agents: {}".format(summary)

        prefix_string = prefix_string + "\n\n Examine each these solutions as additional advice, can solve {} and give your updated answer? Explain your reasoning. \n Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix)
        return {"role": "user", "content": prefix_string}

def construct_assistant_message(completion, in_batches=False):
    '''
    Takes in answers or lists of answers and formats it into a message for the assistant.
    '''
    if in_batches:
        content_list = []
        for item in completion:
            content = item["choices"][0]["messages"]["content"]
            content_list.append({"role": "assistant", "content": content})
        return content_list
    else:
        content = completion["choices"][0]["messages"]["content"]
        return {"role": "assistant", "content": content}


@hydra.main(version_base="1.1", config_path="configs/debate", config_name="config")
def main(cfg: DictConfig):
    # Initialize logging
    logging.basicConfig(level=cfg.logging.level)
    logger = logging.getLogger(__name__)
    logger.info(f"Operating in the following directory:\n {os.getcwd()}")
    logger.info(f"Loading config:\n {cfg.experiment.name}\n{cfg.experiment.description}\n Agents: {cfg.debate.agents}, Rounds: {cfg.debate.rounds}, Summarize:{cfg.debate.summarize}")

    # Configurations
    name = cfg.experiment.name
    batched = cfg.training.batching_scheme
    batch_size = cfg.training.batch
    random.seed(0)
    disable_caching()

    # Import dataset
    logger.info(f"Loading dataset from {os.path.join(cfg.dataset.path, cfg.dataset.name)} using only {cfg.dataset.split}")
    # IF train exists
    dataset = load_from_disk(os.path.join(cfg.dataset.path, cfg.dataset.name))[cfg.dataset.split]
    # dataset = load_from_disk(os.path.join(cfg.dataset.path, cfg.dataset.name))
    dataset = dataset.shuffle()
    dataset = dataset.map(add_index, with_indices=True)

    # Resize dataset
    if dataset.num_rows > cfg.dataset.max_rows:
        logger.info(f"Dataset has too many rows {dataset.num_rows}, will shrink down to {cfg.dataset.max_rows}")
        dataset = dataset.filter(lambda item: item["index"] < cfg.dataset.max_rows)
    logger.info(f"Dataset keys:\n {dataset.column_names}")
    
    # Transform the dataset into a proper chat template
    data_processor = DataProcessingDebate(config=cfg, cot=False)
    dataset = data_processor.transform(dataset, with_golden_label=False)
    logger.info(f"Dataset keys:\n {dataset.column_names}")

    # Construct debate model
    debate = Debate(cfg, logger)
    # Process data according to the batched strategy
    if batched=="batch":
        debate.generate_batched(dataset, batch_size, name)
    elif batched=="duplicate":
        debate.sample(dataset, batch_size, name)
    else:
        debate.generate(dataset, name)

    if dist.is_initialized():
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
