import os 
from pathlib import Path
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification



def quantize_model(model , inputs):

    """ Quantize fp32 precision to int8 . """

    optimized_model_path = os.path.join(dir,'model-artifacts/optimized-model')
    model_name = "distillbert_quant_sentiment.pt"

    # can use [fbgemm , x86 ] for server deployment
    backend ="x86" 
    quantized_model_int8 = torch.quantization.quantize_dynamic(model,
                                                                {torch.nn.Linear},
                                                                dtype=torch.qint8)
    
    quantization_config = torch.quantization.get_default_qconfig(backend)
    quantized_model_int8.qconfig = quantization_config
    quantized_model = torch.quantization.convert(quantized_model_int8, inplace=True)
    quantized_model.eval()

    # change depend on model
    dummy_input = (inputs["input_ids"] , inputs['attention_mask'])
    traced_model = torch.jit.trace(quantized_model , dummy_input)
    torch.jit.save(traced_model, f"{optimized_model_path}/{model_name}")
    print(f"Saved quantized model at {optimized_model_path}/{model_name}")


if __name__ == "__main__":

    dir = Path().resolve().parent
    model_path = os.path.join(dir,'model-artifacts/model') 
    tokenizer_path = os.path.join(dir ,'model-artifacts/tokenizer')

    tokenizer = AutoTokenizer.from_pretrained( tokenizer_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               local_files_only=True ,
                                                               return_dict=False)
    raw_inputs = ["I love you more than you know."]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    
    device = torch.device('cpu')
    model = model.to(device)
    inputs = inputs.to(device)
    quantize_model(model , inputs)
    