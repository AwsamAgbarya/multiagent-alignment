# Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains
### [Project Page](https://llm-multiagent-ft.github.io/) | [Paper](https://arxiv.org/abs/2501.05707)

[Vighnesh Subramaniam](https://vsubramaniam851.github.io/),
[Yilun Du](https://yilundu.github.io/),
[Joshua B Tenenbaum](https://scholar.google.com/citations?user=rRJ9wTJMUB8C&hl=en),
[Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/),
[Shuang Li](https://people.csail.mit.edu/lishuang/),
[Igor Mordatch](https://scholar.google.com/citations?user=Vzr1RukAAAAJ&hl=en)

This is the implementation of our paper "Multiagent Finetuning of Language Models". We design this implementation for the [MATH](https://arxiv.org/pdf/2103.03874) dataset for simplicity. Using other datasets requires little modification of the code.

## Installation and Setup
We include a `requirements.txt` to provide the basic requirements for set-up. To do finetuning with open-source language models, we also include a more details set up in the path `multiagent-ft/lm_ft` with a more detailed list of required packages.

Set your OpenAI API Key using `export OPENAI_API_KEY=your_api_key_here`.

Create a conda/pip environment and install [Pytorch](https://pytorch.org/).
Then run
```
pip install -r requirements.txt
```

### Data
To download the MATH dataset, follow the steps [here](https://github.com/hendrycks/math/).

### Hardware
We ran all experiments either with GPT-3.5 with the OpenAI API or with open-source language models on four 40GB A100s or 8 H100s. 

## Creating Finetuning Data
To create finetuning data, change directory to the `multiagent-ft` directory and run
```
python original_gen.py --agents [AGENTS] --rounds [ROUNDS] --model [MODEL] --save_str [SAVE_STR] --summarize
```
where `agents` refers to the number of multiagent debate agents, `rounds` refers to the number of rounds of debate, `model` refers to the model to use debate on, and `save_str` refers to the save log. `summarize` is an argument that makes the model summarize responses from other agents. We also include parameters for `top_p` and `temperature` that can be used with the open-source models.

## Generation Model Finetuning
To create data for finetuning generation agents, run 
```
python ft_generator.py --file_path [FILE_PATH] --save_path [SAVE_PATH] [--gpt] --iteration [ITERATION]
```

This takes in a path for a JSON file and creates the JSON/JSONL files for finetuning. If you include the `--gpt` flag, it will use the OpenAI API for finetuning assuming the goal is to finetune GPT-3.5. You can track which iteration of finetuning you're applying using the `--iteration` flag. You also can pass in the GPT model IDs to finetuning using the `model_ids` flag.

## Critic Model Finetuning
Similar to generator finetuning. Run
```
python ft_critic.py --file_path [FILE_PATH] --save_path [SAVE_PATH] [--gpt] --iteration [ITERATION]
```

## Open-Source Language Model Finetuning
To finetune open-source models, refer to the `lm_ft` directory.

## Running Finetuning Models
To run finetune models, pass either the model ID strings for OpenAI models or the model paths for the finetuned models as arguments when running
```
python ft_gen.py --generators [GENERATORS] --critics [CRITICS] --model [MODEL] --save_str [SAVE_STR]
```

To account for hardware/memory limitations, this will put all open-source models from HuggingFace on the same GPU. 

## Evaluation
To evaluate the performance, set the correct path for the JSON file and run
```
python eval_math.py
```