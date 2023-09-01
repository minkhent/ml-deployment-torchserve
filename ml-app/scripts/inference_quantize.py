import os 
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
dir = Path().resolve().parent

class TextClassifier:

    def __init__(self , model_dir , tokenizer_dir) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.quant_sentiment = torch.jit.load(f"{model_dir}/distillbert_quant_sentiment.pt")
        self.quant_sentiment.eval()
        self.quant_sentiment.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir ,
                                                    local_files_only=True)


    def preprocess(self , texts):
        tokenized_data = self.tokenizer(texts,
                                   padding=True, 
                                   return_tensors="pt")
        return tokenized_data


    def inference(self, inputs):

        with torch.jit.optimized_execution(False):
            output = self.quant_sentiment(**inputs.to(self.device))
            
        probabilities = torch.nn.functional.softmax(output[0], dim=-1)
        predictions = torch.argmax(probabilities, axis=1)
        predictions = predictions.tolist()
            
        return predictions
    

    def postprocess(self , outputs: list):

        import json
        with open(f'{dir}/index_to_classes.json') as json_file:
            data = json.load(json_file)
        return [data[str(out)] for out in outputs]
    

    def handle(self, texts):

        model_input = self.preprocess(texts)
        model_output = self.inference(model_input)
        preds = self.postprocess(model_output)

        return preds



model_path = os.path.join(dir,'model-artifacts/optimized-model')
tokenizer_path = os.path.join(dir ,'model-artifacts/tokenizer')

classifier = TextClassifier( model_path , tokenizer_path)

def warm_up():
    texts =["I've been waiting for a HuggingFace course my whole life.",]
    with torch.jit.optimized_execution(False):
        for _ in range(5):
            classifier.handle(texts)

            

if __name__ == "__main__":

    warm_up()
    print("Warming up torch script model..")

    texts =["I've been waiting for a HuggingFace course my whole life.",]
    import time
    start = time.perf_counter()

    sample_count = 100
    for _ in range(sample_count):
        prediction = classifier.handle(texts)
    print(f"Average prediction time {((time.perf_counter()-start) * 1000)/sample_count} ms per data sample." )