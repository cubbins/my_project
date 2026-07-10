import json
import torch
from transformers import AutoTokenizer
from smolagents import CodeAgent, TransformersModel


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "qwen_agent_low_level_result.json"

research_question = "Are liberal democracies portrayed as more vulnerable to AI competition?"


def load_agent() -> CodeAgent:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TransformersModel(
        model_id=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    agent = CodeAgent(
        tools=[],
        model=model,
        add_base_tools=False,
        max_steps=2,
    )

    return agent


def main() -> int:
    agent = load_agent()

    prompt = f"""
Answer this research question in 3 short bullet points.

Research question:
{research_question}

Do not search the web.
Do not write code.
Do not return JSON.
Only give a short academic interpretation.
"""

    print("Running low-level Qwen CodeAgent test...")

    response = agent.run(prompt)

    result = {
        "topic": research_question,
        "agent_response": str(response),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\nSuccess. Saved result to: {OUTPUT_FILE}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())