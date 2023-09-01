import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import json
from ts.torch_handler.base_handler import BaseHandler
import logging


logger = logging.getLogger(__name__)

class TextClassificationHandler(BaseHandler):


    def initialize(self, context):
        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        
        # use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # load the model
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)
        print(model_path)
        if os.path.isfile(model_path):
            self.model = torch.jit.load(f"{model_dir}/distillbert_quant_sentiment.pt")
            self.model.to(self.device)
            self.model.eval()

        logger.info('Successfully loaded model')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logger.info('Successfully loaded tokenizer')

        with open('index_to_classes.json') as json_file:
            self.classes = json.load(json_file)
            print(self.classes)

        self.initialized = True


    def postprocess(self, outputs):

        return [self.classes[str(out)] for out in outputs]
    
    
    def inference(self, inputs):
        with torch.jit.optimized_execution(False):
            output = self.model(**inputs.to(self.device))
            
        probabilities = torch.nn.functional.softmax(output[0], dim=-1)
        predictions = torch.argmax(probabilities, axis=1)
        predictions = predictions.tolist()
            
        return predictions
    
    def preprocess(self,inputs):

        data = inputs[0].get('body')
        texts = data.get('data')
        # tokenize the texts
        tokenized_data = self.tokenizer(texts,
                                   padding=True,
                                   return_tensors='pt')
        return tokenized_data 
    

    def handle(self, data, context):

        if not self.initialized:
          self.initialized(context)
          
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        preds = self.postprocess(model_output)


        return [preds]
    