import json
import time
import re
import torch
import traceback
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from transformers import AutoTokenizer
from smolagents import CodeAgent, TransformersModel


import warnings
from transformers import logging as hf_logging

warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
hf_logging.set_verbosity_error()



MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

OUTPUT_FILE = "qwen_agent_low_level_result.json"
MONITOR_FILE = "qwen_agent_monitor_log.json"

research_question = "Are liberal democracies portrayed as more vulnerable to AI competition?"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_agent_text22(text: str) -> str:
    """
    Remove ANSI color/control codes and simplify rich terminal box characters.
    """
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    replacements = {
        "╭": "+",
        "╮": "+",
        "╰": "+",
        "╯": "+",
        "─": "-",
        "━": "-",
        "│": "|",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_agent_text(text: str) -> str:
    """
    Convert rich/ANSI terminal output into plain text suitable for log files.
    """
    if not text:
        return ""

    # Remove ANSI escape/control sequences
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    text = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b[@-Z\\-_]", "", text)

    # Replace rich box drawing characters
    replacements = {
        "╭": "+",
        "╮": "+",
        "╰": "+",
        "╯": "+",
        "─": "-",
        "━": "-",
        "│": "|",
        "┃": "|",
        "┌": "+",
        "┐": "+",
        "└": "+",
        "┘": "+",
        "├": "+",
        "┤": "+",
        "┬": "+",
        "┴": "+",
        "┼": "+",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove excessive trailing spaces on each line
    lines = [line.rstrip() for line in text.splitlines()]

    # Collapse repeated blank lines
    cleaned_lines = []
    previous_blank = False

    for line in lines:
        blank = line.strip() == ""
        if blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = blank

    return "\n".join(cleaned_lines).strip()


def load_agent22() -> CodeAgent:
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

def load_agent() -> CodeAgent:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if "<|endoftext|>" in tokenizer.get_vocab() and "<|endoftext|>" != tokenizer.eos_token:
            tokenizer.pad_token = "<|endoftext|>"
            added_pad_token = False
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            added_pad_token = True
    else:
        added_pad_token = False

    model = TransformersModel(
        model_id=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.model.config.pad_token_id = tokenizer.pad_token_id
    model.model.generation_config.pad_token_id = tokenizer.pad_token_id

    if added_pad_token:
        model.model.resize_token_embeddings(len(tokenizer))

    agent = CodeAgent(
        tools=[],
        model=model,
        add_base_tools=False,
        max_steps=2,
    )

    return agent





def run_agent_with_monitor(agent: CodeAgent, prompt: str, max_retries: int = 1) -> dict:
    monitor = {
        "task": "Low-level Qwen CodeAgent research interpretation",
        "research_question": research_question,
        "model_id": MODEL_ID,
        "prompt": prompt,
        "started_at": now_iso(),
        "attempts": [],
        "final_success": False,
        "final_response": None,
        "ended_at": None,
        "elapsed_seconds": None,
    }

    start_time = time.time()

    for attempt_number in range(1, max_retries + 2):
        attempt_start = time.time()

        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        attempt_record = {
            "attempt_number": attempt_number,
            "started_at": now_iso(),
            "ended_at": None,
            "status": "started",

"agent_console_output": None,
"agent_error_output": None,


            "response": None,
            "exception_type": None,
            "exception_message": None,
            "traceback": None,
            "elapsed_seconds": None,
            "evaluation": None,
        }

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                response = agent.run(prompt)

            attempt_record["status"] = "success"
            attempt_record["response"] = str(response)
            attempt_record["evaluation"] = "Agent completed without raising an exception."

            monitor["final_success"] = True
            monitor["final_response"] = str(response)

        except Exception as exc:
            attempt_record["status"] = "failed"
            attempt_record["exception_type"] = type(exc).__name__
            attempt_record["exception_message"] = str(exc)
            attempt_record["traceback"] = traceback.format_exc()
            attempt_record["evaluation"] = "Agent attempt failed; retry may be useful."

        finally:
            raw_stdout = stdout_buffer.getvalue()
            raw_stderr = stderr_buffer.getvalue()

            #attempt_record["agent_console_output_raw"] = raw_stdout
            #attempt_record["agent_error_output_raw"] = raw_stderr
            #attempt_record["agent_console_output_clean"] = clean_agent_text(raw_stdout)
            #attempt_record["agent_error_output_clean"] = clean_agent_text(raw_stderr)


            attempt_record["agent_console_output"] = clean_agent_text(raw_stdout)
            attempt_record["agent_error_output"] = clean_agent_text(raw_stderr)


            attempt_record["elapsed_seconds"] = round(time.time() - attempt_start, 3)
            attempt_record["ended_at"] = now_iso()

            monitor["attempts"].append(attempt_record)

            with open(MONITOR_FILE, "w", encoding="utf-8") as f:
                json.dump(monitor, f, indent=4, ensure_ascii=False)

        if monitor["final_success"]:
            break

    monitor["ended_at"] = now_iso()
    monitor["elapsed_seconds"] = round(time.time() - start_time, 3)

    with open(MONITOR_FILE, "w", encoding="utf-8") as f:
        json.dump(monitor, f, indent=4, ensure_ascii=False)

    return monitor


def main() -> int:
    agent = load_agent()

    prompt22 = f"""
       Answer this research question in 3 short bullet points.

       Research question:
    {research_question}

Do not search the web.
Do not write code.
Do not return JSON.
Only give a short academic interpretation.
"""

    prompt = f"""
Use final_answer to answer this research question in 3 short bullet points.

Research question:
{research_question}

Do not search the web.
Do not use tools.
Return your answer by calling final_answer("...").
"""




    print("Running monitored Qwen CodeAgent test...")

    monitor = run_agent_with_monitor(
        agent=agent,
        prompt=prompt,
        max_retries=1,
    )

    result = {
        "topic": research_question,
        "agent_response": monitor["final_response"],
        "success": monitor["final_success"],
        "monitor_file": MONITOR_FILE,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\nResult saved to: {OUTPUT_FILE}")
    print(f"Monitor log saved to: {MONITOR_FILE}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0 if monitor["final_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())