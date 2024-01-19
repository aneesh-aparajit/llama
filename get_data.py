import os
import requests
import dotenv
from urllib.request import urlretrieve
dotenv.load_dotenv()



url = "https://huggingface.co/api/datasets/uonlp/CulturaX/parquet/ml/train"

payload = {}
headers = {
  'Authorization': f'Bearer {os.getenv("HF_TOKEN")}'
}

response = requests.request("GET", url, headers=headers, data=payload)
urls = response.text[1:-1].split(",")
urls = [url[1:-1] for url in urls]

for url in urls:
    print(url)
