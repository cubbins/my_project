# Change HfApiModel to InferenceClientModel
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

# Initialize the search tool
search_tool = DuckDuckGoSearchTool()

# Use InferenceClientModel instead of HfApiModel
model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create the agent and equip it with the search tool
agent = CodeAgent(
    tools=[search_tool], 
    model=model,
    add_base_tools=False
)

# Define your topic and run the agent
topic = "James Webb Space Telescope"
prompt = f"Find the official Wikipedia page URL for the topic: '{topic}'. Return only the URL."

response = agent.run(prompt)
print("\nResult:")
print(response)
