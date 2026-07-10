# Import the new, correct modular components from autogen
from autogen import ConversableAgent
from autogen.openai import OpenAIChatClient

# 1. Define the offline model configuration using the new OpenAIChatClient standard
# This routes explicitly to your local, offline Ollama instance
client = OpenAIChatClient(
    model="llama3",                           # Must match your local Ollama model name
    base_url="http://localhost:11434/v1",     # Local Ollama endpoint
    api_key="ollama",                         # Required placeholder for local servers
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False
    }
)

# 2. Create the AI Thinker Agent (replaces AssistantAgent)
assistant = ConversableAgent(
    name="Assistant_Agent",
    client=client,
    system_message="You are a helpful AI assistant. Provide concise answers.",
    human_input_mode="NEVER"
)

# 3. Create the User Agent (replaces UserProxyAgent)
user_proxy = ConversableAgent(
    name="User_Proxy",
    client=client,
    system_message="You are a human user testing an AI agent.",
    human_input_mode="NEVER"  # Operates autonomously without pausing for typing
)

# 4. Initiate the offline conversation using the modern client syntax
print("--- Starting Offline Multi-Agent Conversation (New Architecture) ---")
result = user_proxy.initiate_chat(
    recipient=assistant,
    message="Write a 2-sentence poem about a robot learning to paint.",
    max_turns=2  # Prevents infinite loops
)
