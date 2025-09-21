#!/bin/bash
curl -L -o data/sentiment_dataset.zip https://www.kaggle.com/api/v1/datasets/download/mar1mba/russian-sentiment-dataset
unzip data/sentiment_dataset.zip -d data/