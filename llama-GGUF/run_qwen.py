from llama_cpp import Llama

# Load the first shard; llama.cpp automatically loads shard 2
llm = Llama(
    model_path=r"C:\LLM\models\qwen2.5-7b\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    n_ctx=4096,
    n_gpu_layers=0,   # CPU-only for now
)

response = llm(
    "Hello! Please describe your capabilities.",
    max_tokens=200,
)

print(response["choices"][0]["text"])
