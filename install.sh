#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # put your install commands here (remove lines you don't need):
  apt update; apt install -y [...] ; apt clean
  conda install -y [...]
  
   # install DL
  pip uninstall -y transformers datasets
  pip install glob
  pip install openai
  pip install "pandas>=2.2.2"
  pip install "pyarrow>=18.0.0"
  pip install transformers==4.39.3
  pip install datasets
  pip install joblib
  pip install tqdm
  pip install sympy
  pip install pylatexenc
  pip install typing
  pip install vllm
  pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/
  
  # install config managers
  pip install hydra-core --upgrade
  
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi