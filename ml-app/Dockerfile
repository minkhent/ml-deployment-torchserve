FROM pytorch/torchserve:latest-cpu  

COPY . .
 
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

#start the server with model 
CMD ["torchserve", "--start" , "--ncs", "--model-store", "model-store" ,"--models" ,"sentiment_model=SentimentAnalysis.mar"]