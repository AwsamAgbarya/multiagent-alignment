from glob import glob
import openai
import json
import numpy as np
import time
import random
import os
import transformers
import torch
from tqdm import tqdm
import argparse

def generate_answer_summary(answer_context, model = "mistral", tokenizer = None, hf_model = None, device = None):
    if model not in ["mistral", "phi3", "llama3"]:
        try:
            completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0125",
                    seed=0,
                    messages=answer_context,
                    n=1)
        except:
            print("retrying due to an error......")
            time.sleep(20)
            return generate_answer_summary(answer_context)
    else:
        hf_model = hf_model.to(device)
        input_text = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        output = hf_model.generate(input_ids, max_length=len(input_ids[0]) + 2048, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True, top_p = 0.9, temperature = 1)
        generated_ids = output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion = {"choices": [{"message": {"role": "assistant", "content": completion}}]}
        cpu_device = torch.device("cpu")
        hf_model = hf_model.to(cpu_device)
    return completion

def generate_answer(answer_context, i, model, models, device = None, tokenizer = None):
    if model not in ["mistral", "phi3", "llama3"]:
        try:
            completion = openai.ChatCompletion.create(
                    model=models[i%3],
                    messages=answer_context,
                    seed=i,
                    n=1)
        except:
            print("retrying due to an error......")
            time.sleep(20)
            return generate_answer(answer_context, i, model, models)
    else:
        hf_model = models[i%3]
        hf_model = hf_model.to(device)
        input_text = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        output = hf_model.generate(input_ids, max_length=len(input_ids[0]) + 2048, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True, top_p = 0.9, temperature = 1)
        generated_ids = output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion = {"choices": [{"message": {"role": "assistant", "content": completion}}]}
        cpu_device = torch.device("cpu")
        hf_model = hf_model.to(cpu_device)
    return completion

def load_hf_model(model_path):
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True)
    except:
        raise OSError(f"{model_path} does not exist or there was an error during finetuning...")
    return model

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def summarize_message(agent_contexts, hf_model = None, tokenizer = None, device = None):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent and explain the reasoning in each solution."
    agent_context = [{"role": "user", "content": prefix_string}]
    completion = generate_answer_summary(agent_context, hf_model = hf_model, tokenizer = tokenizer, device = device)
    content = completion["choices"][0]["message"]["content"]

    return content

def construct_message(agents, prefix, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct? Please reiterate your answer, with your final answer a single answer of the form \\boxed{{answer}} at the end of your response.".format(prefix)}

    prefix_string = "Here is are solution from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: {}".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Using each response as additional advice, can you give an updated bullet by bullet answer to {}? Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix)
    return {"role": "user", "content": prefix_string}

def construct_message_summary(summary, prefix, idx):
    prefix_string = "Here is a summary of solutions from several other agents: {}".format(summary)

    prefix_string = prefix_string + "\n\n Examine each these solutions as additional advice, can solve {} and give your updated answer? Explain your reasoning. \n Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix)
    return {"role": "user", "content": prefix_string}


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def parse_answer(input_str):
	return remove_boxed(last_boxed_only_string(input_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generators", action = "store", nargs = "*", dest = "generators", required = True)
    parser.add_argument("--critics", action = "store", nargs = "*", dest = "critics", required = True)
    parser.add_argument("--save_str", action = "store", type = str, dest = "save_str", required = True)
    parser.add_argument("--model", action = "store", default = "gpt3.5", type = str, choices = ["gpt3.5", "mistral", "llama3", "phi3"])
    parser.add_argument("--summarize", action = "store_true", dest = "summarize")
    parser.add_argument("--temperature", action = "store", default = 1, type = float, dest = "temperature")
    parser.add_argument("--top_p", action = "store", default = 0.9, type = float, dest = "top_p")
    parser.add_argument("--device", action = "store", type = int, dest = "device", default = 0)
    args = parser.parse_args()
    jsons = sorted(glob("MATH/test/*/*.json"))
    random.seed(0)
    random.shuffle(jsons)
    hard_problems = []

    for json_file in jsons:
        data = json.load(open(json_file, "r"))
        if ('1' in data['level']) or ('2' in data['level']) or ('3' in data['level']):
            hard_problems.append(data)

    agents = len(args.generators)
    rounds = 2
    np.random.seed(0)

    model = args.model
    device = args.device
    tokenizer = None
    hf_model = None
    if model in ["llama3", "phi3", "mistral"]:
        generator_models = []
        for path in args.generators:
            mistral_model = load_hf_model(path)
            generator_models.append(mistral_model)
        critic_models = []
        for path in args.critics:
            mistral_model = load_hf_model(path)
            critic_models.append(mistral_model)
        if args.model == "mistral":
            model_str = "mistralai/Mistral-7B-Instruct-v0.2"
        elif args.model == "llama3":
            model_str = "meta-llama/Meta-Llama-3-8B"
        elif args.model == "phi3":
            model_str = "microsoft/Phi-3-mini-128k-instruct" 
        else:
            raise NotImplementedError()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code = True)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, trust_remote_code = True).to(device)
    else:
        generator_models = args.generators
        critic_models = args.critics

    random.seed(0)
    random.shuffle(hard_problems)

    generated_description = {}
    summarize = args.summarize

    for problem, data in tqdm(enumerate(hard_problems[500:1000]), desc = "Fine-tuned Generation on MATH"):
        question = data["problem"]
        answer = data["solution"]

        print("problem: ", problem)

        answer_parse = parse_answer(answer)

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Provide a bullet point summary of your reasoning. Your final answer should be a single answer, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    if summarize:
                        summary = summarize_message(agent_contexts_other, hf_model, tokenizer, device)
                        message = construct_message_summary(summary, question, 2 * round - 1)
                    else:
                        message = construct_message(agent_contexts_other, question, 2 * round - 1)
                    agent_context.append(message)

                if round == 0:
                    completion = generate_answer(agent_context, i=i, model = model, models = generator_models, tokenizer = tokenizer, device = device)
                else:
                    completion = generate_answer(agent_context, i=i, model = model, models = critic_models, tokenizer = tokenizer, device = device)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(completion)
                print("{} gt_answer: ".format(problem), answer_parse)

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("{}.json".format(args.save_str), "w"))
    import pdb
    pdb.set_trace()
    print(jsons)