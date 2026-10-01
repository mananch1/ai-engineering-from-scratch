import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
print("API KEY:", os.getenv("ANTHROPIC_API_KEY"))
print("MODEL:", os.getenv("LLM_MODEL"))

client = anthropic.Anthropic()

MODEL = os.environ.get("LLM_MODEL")


try:
    response = client.messages.create(
        model = MODEL,
        max_tokens = 256,
        messages = [{"role":"user","content":[
            {"type":"text",
            "text":"What is a NN"}
            ]}]
    )
    print(response.content[0].text)

except:
    print("Hi")

import urllib.request
import json

url = "https://api.anthropic.com/v1/messages"

headers = {
    "Content-Type":"application/json",
    "x-api-key":os.getenv("ANTHROPIC_API_KEY"),
    "anthropic-version" : "2023-06-01"
}
body = json.dumps({
    "model" : os.getenv("LLM_MODEL"),
    "max_tokens":256,
    "messages":[{"role":"user","content":[
            {"type":"text",
            "text":"What is a NN"}
            ]}]
}).encode()

req = urllib.request.Request(url,data=body,headers=headers,method="POST")
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["content"][0]["text"])







