import requests
import base64
import json 


def test_model(payload_data):

    api_url = 'http://127.0.0.1:6060/predictions/sentiment_model'
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

    payload = json.dumps(payload_data)
    response = requests.post(api_url, data=payload, headers=headers)

    # try:
    #     data = response.json()
    #     print(data)                  
    # except requests.exceptions.RequestException:
    #     print(response.text)

if __name__ == "__main__":

    test_data = {
        "data" :["I love you more than you know."]
        }
    import time
    start = time.perf_counter()
    sample_count = 1000
    for _ in range(sample_count):
        test_model(test_data)
    print(f"Average prediction time {((time.perf_counter()-start) * 1000)/sample_count} ms per data sample." )