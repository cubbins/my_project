import torch
# Swap out InferenceClientModel for TransformersModel
from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel

# 1. Initialize the search tool
search_tool = DuckDuckGoSearchTool()

# 2. Setup the local model using Hugging Face Transformers pipeline
# This automatically utilizes your GPU via device_map="auto"
model = TransformersModel(
    model_id="Qwen/Qwen2.5-3B-Instruct", 
    device_map="auto",
    torch_dtype=torch.bfloat16 # Use bfloat16 for efficient VRAM usage on your 5060
)

# 3. Create the agent and equip it with the search tool
agent = CodeAgent(
    tools=[search_tool], 
    model=model,
    add_base_tools=False
)

# 4. Define your topic and run the agent locally
topic = "James Webb Space Telescope"
prompt = f"Find the official Wikipedia page URL for the topic: '{topic}'. Return only the URL."

response = agent.run(prompt)
print("\nResult (Computed Locally):")
print(response)
