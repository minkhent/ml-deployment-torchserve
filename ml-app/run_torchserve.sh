#!/bin/bash

torchserve \
   --start \
   --ncs \
   --ts-config config.properties \
   --model-store model-store \
   --models sentiment_model=SentimentAnalysis.mar