# Run scripts

```
cd ml-app/scripts

# download model from huggingface hub
python download_model.py


# optimize(quantize) model ( fp32 to int8 )
python optimize_model.py


# check optimized-model's inference time
python inference_quantize.py

```

# Create model-archive

```
# back to main dir "ml-app"
cd ..
./create_mar.sh
```

We can access three containers at:

* Prometheus: http://localhost:9090/
* Grafana: http://localhost:3000/
* Torchserve: http://localhost:6060/

# Check sever

http://localhost:6060/ping

