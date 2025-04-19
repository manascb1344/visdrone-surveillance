#! /bin/bash

pip install -r requirements.txt
python main.py --mode prepare
python main.py --mode train --model-size m --epochs 100 --num-gpus 1 --batch-size 64