import requests
import json

url = "https://api.fireworks.ai/inference/v1/completions"
payload = {
  "model": "accounts/fireworks/models/deepseek-v3-0324",
  "max_tokens": 4096,
  "top_p": 1,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.6,
  "prompt": "Hello, how are you?"
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": "Bearer fw_S5ELtzgjkMPjBGZLSRBA1P"
}
response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
print(response.text)