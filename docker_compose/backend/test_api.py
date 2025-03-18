import requests

url = "http://localhost:8000/generate"
data = {"query": "What is the capital of Colombia?", "max_length": 100}

response = requests.post(url, json=data)
print(response.json())


data = {"query": "How many people live there?", "max_length": 100}

response = requests.post(url, json=data)
print(response.json())
