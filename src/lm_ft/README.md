# Open-Source Language Model Finetuning
We borrow most of our setup from [FastChat](https://github.com/lm-sys/FastChat). If FastChat does not work, we also used the [TRL trainer](https://huggingface.co/docs/trl/main/en/trainer). The setup here can be difficult and fragile. so we recommend you use a new environemnt (conda or pip). 

For all finetuning, we use either four 40GB A100s for Mistral and Phi-3 or four H100s for LLaMA-3. 

## Installation
We include a file called `lm_ft.txt` with all package requirements we used for finetuning. Please install this carefully, we have found that mismatched packages will lead to odd behavior such as OOM. After creating a new environment, install [Pytorch](https://pytorch.org) with the correct CUDA setting. Then run
```
pip install -r lm_ft.txt
```
## Running Finetuning
To run finetuning, just run
```
./ft.sh
```

You can view `ft.sh` to see that it will finetune Mistral with an example JSON file in the `data/` directory. You can set the path to the relevant finetuning file in this case. When you change the model, keep in mind that you must change the layer wrapper (officially `fsdp_transformer_layer_cls_to_wrap`) to use the correct layer name. We set this correctly for Mistral but this must be reset if you use a new model. Some models may not integrate with FastChat.

## Data Preprocessing
The current code will structure the data automatically to allow for finetuning. 