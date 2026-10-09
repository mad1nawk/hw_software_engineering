import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "qwen2.5:7b",
    "prompt": "explain what is llm, ml, harness.",
    "stream": False
}

response = requests.post(url, json=payload)
print(response.json()["response"])
