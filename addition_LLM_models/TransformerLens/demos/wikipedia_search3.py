import torch
from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel

search_tool = DuckDuckGoSearchTool()

model = TransformersModel(
    model_id="Qwen/Qwen2.5-3B-Instruct", 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Crucial: set max_steps to 8 or 10 so it can do multiple research iterations
agent = CodeAgent(
    tools=[search_tool], 
    model=model,
    add_base_tools=False,
    max_steps=10 
)

# Your complex research query
research_question = "Are liberal democracies portrayed as more vulnerable to AI competition?"

# Formulate the multi-step research prompt
prompt = f"""
You are an advanced academic research assistant. Investigate the following question: 
"{research_question}"

Execute your research using these steps:
1. Break this question down into at least 3 distinct search angles (e.g., "AI and authoritarianism", "democratic vulnerability to AI disinformation", "geopolitics of artificial intelligence").
2. Perform multiple searches to find relevant concepts, political theories, and specific controversies.
3. Locate the Wikipedia pages that cover these sub-topics or the overall debate.
4. Provide a synthesis answer to the question based on what you find, and format your output with a markdown list of the Wikipedia articles and their URLs that back up your points.
"""

response = agent.run(prompt)
print("\nResearch Results:")
print(response)
