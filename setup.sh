#!/bin/bash

set -e

# setup conda
source ~/miniconda3/etc/profile.d/conda.sh

## create conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (must be >=3.8): " python_version
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# install torch
read -rp "Enter torch version (check it is compatible with all your requirements): " torch_version
read -rp "Enter cuda version (10.2, 11.3 or none to avoid installing cuda support. Not sure? Check out https://stackoverflow.com/a/68499241/1908499): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch=$torch_version torchvision cpuonly -c pytorch
else
    conda install -y pytorch=$torch_version torchvision cudatoolkit=$cuda_version -c pytorch
fi

# install faiss
read -rp "Enter faiss version: " faiss_version
conda install -y -c pytorch faiss-gpu==$faiss_version cudatoolkit=$cuda_version -c pytorch

# install python requirements
pip install -r requirements.txt
classy --install-autocomplete

# download bart
python -c 'from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained("facebook/bart-large"); AutoModel.from_pretrained("facebook/bart-large")'

# download stanza en
python -c 'import stanza; stanza.download("en")'

# install java (needed to run raganato eval)
sudo apt-get install -y openjdk-11-jdk

# install xmllint (needed for exmaker wsd generation)
sudo apt-get install libxml2-utils

# download nltk wordnet
python -c "import nltk; nltk.download('wordnet')"
