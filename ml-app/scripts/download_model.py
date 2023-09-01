import os 
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


dir = Path().resolve().parent
model_path = os.path.join(dir,'model-artifacts/model') 
tokenizer_path = os.path.join(dir ,'model-artifacts/tokenizer')

tokenizer = AutoTokenizer.from_pretrained("sbcBI/sentiment_analysis" )
model = AutoModelForSequenceClassification.from_pretrained("sbcBI/sentiment_analysis")
tokenizer.save_pretrained(tokenizer_path)
model.save_pretrained(model_path)

print(f"Save model and tokenizer at {model_path} and {tokenizer_path}")


