#!/bin/bash

base_path="model-artifacts"
config_path="$base_path/model"
model_path="$base_path/optimized-model"
tokenizer_path=$base_path"/tokenizer"

torch-model-archiver \
    --model-name SentimentAnalysis \
    --version 1.0 \
    --model-file "$model_path/distillbert_quant_sentiment.pt" \
    --handler handler.py \
    --extra-files "$config_path/config.json,$tokenizer_path/special_tokens_map.json,$tokenizer_path/tokenizer_config.json,$tokenizer_path/tokenizer.json,$tokenizer_path/vocab.txt,index_to_classes.json" \
    --export-path model-store


    