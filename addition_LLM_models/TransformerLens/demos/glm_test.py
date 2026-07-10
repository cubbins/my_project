import os
from openai import OpenAI

# 1. Configure client for Zenmux
client = OpenAI(
    base_url="https://zenmux.ai/api/v1",
    api_key=os.environ.get("ZENMUX_API_KEY", "YOUR_ZENMUX_API_KEY") 
)

# 2. Call GLM-5.2 free tier
response = client.chat.completions.create(
    model="z-ai/glm-5.2-free",
    messages=[
        {"role": "user", "content": "Write a quick python script to parse an API JSON response."}
    ]
)
print(response.choices[0].message.content)
