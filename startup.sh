#!/bin/bash

./download_dataset.sh

pip install -r requirements.txt

python3 src/prepare_training_dataset.py

python3 src/main.py

python3 src/inference.py

