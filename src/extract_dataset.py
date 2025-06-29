import json
import os
import numpy as np
from tqdm import tqdm
import random
import hydra
import logging
from omegaconf import DictConfig
from datasets import load_from_disk, load_dataset, disable_caching
from utils import trim_content, remove_duplicates, add_index
from processing import DataParser
import argparse
    
@hydra.main(version_base="1.1", config_path="configs/datasets/build", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=cfg.logging.level)
    logger = logging.getLogger(__name__)
    logger.info(f"Operating in the following directory:\n {os.getcwd()}")
    logger.info(f"Loading config:\n {cfg.experiment.name}\n{cfg.experiment.description}\n")

    # Configurations
    input_path = cfg.dataset.input.path
    input_file_name = cfg.dataset.input.name+".json"
    inpute_file = os.path.join(input_path, input_file_name)
    output_path = cfg.dataset.output.path
    model_ids = cfg.model.ids
    nagent = cfg.dataset.input.num_of_generators
    answer_type = cfg.dataset.input.format
    dataset_type = cfg.dataset.input.type
    iteration = cfg.experiment.iteration

    # Load the data
    input_dataset = json.load(open(inpute_file, "r"))
    keys_to_keep = list(input_dataset.keys())

    if len(input_dataset) > cfg.dataset.input.max_rows:
        logger.info(f"Dataset has too many rows {len(input_dataset)}, will shrink down to {cfg.dataset.input.max_rows}")
        keys_to_keep = keys_to_keep[:cfg.dataset.input.max_rows]
        print(f"{keys_to_keep}")
    input_dataset = {k: input_dataset[k] for k in keys_to_keep}

    # Initializations for generator
    parser = DataParser(answer_type, dataset_type)
    answers_dicts_gen = [{} for i in range(nagent)]
    counters_gen = [0 for i in range(nagent)]
    # Initializations for critic
    counters_crit = [0 for i in range(nagent)]
    answers_dicts_crit = [{} for i in range(nagent)]
    corrected_counters_crit = [0 for i in range(nagent)]
    corrected_answers_dicts_crit = [{} for i in range(nagent)]

    # for every agent context datapoint
    for k, v in tqdm(input_dataset.items()):
        agents_answers, gt_answer = v
        answers = []
        # Get final round answers
        for agent_answers in agents_answers:
            answer = parser.MC_parse(agent_answers[-1]['content'])
            if answer is not None:
                answers.append(answer)
        if len(answers) == 0:
            continue

        # Get the most frequent answer throughout the diff final round answers
        consensus_anwer = parser.most_frequent(answers)

        # =========================================== GENERATOR ==================================================
        # Get first round answers of every generator agent
        for i, agent_answers in enumerate(agents_answers):
            answers_dict = answers_dicts_gen[i] # agent i's answer dictionary
            counter = counters_gen[i] # agent i's counter
            # Get the generators answer (first through the chain)
            first_answer = parser.MC_parse(agent_answers[1]['content'])
            # If the generator was initially correct
            if parser.grade(first_answer, consensus_anwer):
                # Add the QA to its dictionary to create a dataset of all the correct answers it has generated for a given question.
                answers_dict[counter] = agent_answers[:2]
                # Increase and update counter
                counter = counter + 1
                counters_gen[i] = counter
        # =========================================== CRITIC ========================================================
        # Get first and last round answers of every agent
        for i, agent_answers in enumerate(agents_answers):
            first_answer = parser.MC_parse(agent_answers[1]['content'])
            last_answer = parser.MC_parse(agent_answers[-1]['content'])
            if last_answer is None:
                continue
            # If we reached the correct solution
            if parser.grade(last_answer, consensus_anwer):
                # Initially correct and stayed correct
                if not parser.grade(first_answer, last_answer):
                    answers_dict = answers_dicts_crit[i]
                    counter = counters_crit[i]

                    answers_dict[counter] = agent_answers
                    counter = counter + 1
                    counters_crit[i] = counter
                # Initially wrong but got corrected
                else:
                    answers_dict = corrected_answers_dicts_crit[i]
                    counter = corrected_counters_crit[i]

                    answers_dict[counter] = agent_answers
                    counter = counter + 1
                    corrected_counters_crit[i] = counter
            
    ft_json = "{}_IT{}".format(cfg.experiment.name, iteration) + "_GEN{}.json"
    # For every generating agent create a dataset of initially correct QA
    for n in range(nagent):
        with open(ft_json.format(n), "w") as f:
            json.dump(answers_dicts_gen[n], f)

    ft_json = "{}_IT{}".format(cfg.experiment.name, iteration) + "_CRIT{}.json"
    for i in range(nagent):
        answer_json = []
        incorrect_data = answers_dicts_crit[i]
        correct_data = corrected_answers_dicts_crit[i]
        data = list(correct_data.values()) + list(incorrect_data.values())
        random.shuffle(data)
        answer_json = {"id": f"identity_{i}"}
        answer_json["conversations"] = data
        with open(ft_json.format(i), "w") as f:
            json.dump(answer_json, f)
    logger.info(f"Dataset saved to {os.getcwd()}")

if __name__ == "__main__":
    main()