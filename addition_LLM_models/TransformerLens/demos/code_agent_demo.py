import json
import re
import time
import traceback
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

import torch
from transformers import AutoTokenizer
from smolagents import CodeAgent, TransformersModel



MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

MONITOR_FILE = "codeagent_programming_monitor.json"
SOLUTION_FILE = "codeagent_final_solution.py"


PROGRAMMING_TASK = """
Write a Python function named summarize_transactions(transactions).

The input is a list of dictionaries. Each dictionary has:
- "account": string
- "category": string
- "amount": number
- "date": string in YYYY-MM-DD format

The function must return a dictionary with:
{
    "total": total of all amounts,
    "by_account": totals grouped by account,
    "by_category": totals grouped by category,
    "largest_transaction": the original transaction dictionary with the largest absolute amount,
    "month_totals": totals grouped by YYYY-MM
}

Rules:
- Do not use pandas.
- Do not read or write files.
- Round numeric totals to 2 decimals.
- If transactions is empty, return:
  {
      "total": 0,
      "by_account": {},
      "by_category": {},
      "largest_transaction": None,
      "month_totals": {}
  }
"""


TEST_TRANSACTIONS = [
    {"account": "checking", "category": "salary", "amount": 2500.00, "date": "2026-01-03"},
    {"account": "checking", "category": "rent", "amount": -1200.00, "date": "2026-01-05"},
    {"account": "credit", "category": "groceries", "amount": -84.255, "date": "2026-01-08"},
    {"account": "credit", "category": "groceries", "amount": -40.10, "date": "2026-02-02"},
    {"account": "savings", "category": "interest", "amount": 3.456, "date": "2026-02-15"},
]


EXPECTED = {
    "total": 1179.1,
    "by_account": {
        "checking": 1300.0,
        "credit": -124.36,
        "savings": 3.46,
    },
    "by_category": {
        "salary": 2500.0,
        "rent": -1200.0,
        "groceries": -124.36,
        "interest": 3.46,
    },
    "largest_transaction": {"account": "checking", "category": "salary", "amount": 2500.00, "date": "2026-01-03"},
    "month_totals": {
        "2026-01": 1215.74,
        "2026-02": -36.64,
    },
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean_agent_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    text = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b[@-Z\\-_]", "", text)

    replacements = {
        "╭": "+", "╮": "+", "╰": "+", "╯": "+",
        "─": "-", "━": "-", "│": "|", "┃": "|",
        "┌": "+", "┐": "+", "└": "+", "┘": "+",
        "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    lines = [line.rstrip() for line in text.splitlines()]

    cleaned_lines = []
    previous_blank = False

    for line in lines:
        blank = line.strip() == ""
        if blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = blank

    return "\n".join(cleaned_lines).strip()


def extract_code(text: str) -> str:
    """
    Extract Python code from agent output.
    Accepts fenced code blocks or raw code.
    """
    text = str(text).strip()

    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text


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

    return CodeAgent(
        tools=[],
        model=model,
        add_base_tools=False,
        max_steps=3,
        additional_authorized_imports=["math", "datetime", "collections"],
    )




def load_agent() -> CodeAgent:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"

    # Create a real pad token, not EOS
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = TransformersModel(
        model_id=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        model_kwargs={
            "dtype": torch.bfloat16,
        },
    )

    # Because we added a token, resize embeddings
    model.model.resize_token_embeddings(len(tokenizer))

    return CodeAgent(
        tools=[],
        model=model,
        add_base_tools=False,
        max_steps=3,
        additional_authorized_imports=["math", "datetime", "collections"],
    )






def run_tests(candidate_code: str) -> tuple[bool, str]:
    namespace = {}

    try:
        exec(candidate_code, namespace)

        if "summarize_transactions" not in namespace:
            return False, "Function summarize_transactions was not defined."

        func = namespace["summarize_transactions"]

        result = func(TEST_TRANSACTIONS)

        if result != EXPECTED:
            return (
                False,
                "Function returned incorrect result.\n"
                f"Expected:\n{json.dumps(EXPECTED, indent=2)}\n\n"
                f"Got:\n{json.dumps(result, indent=2)}"
            )

        empty_result = func([])

        expected_empty = {
            "total": 0,
            "by_account": {},
            "by_category": {},
            "largest_transaction": None,
            "month_totals": {},
        }

        if empty_result != expected_empty:
            return (
                False,
                "Function failed empty-list test.\n"
                f"Expected:\n{json.dumps(expected_empty, indent=2)}\n\n"
                f"Got:\n{json.dumps(empty_result, indent=2)}"
            )

        return True, "All tests passed."

    except Exception:
        return False, traceback.format_exc()


def ask_agent_for_solution(agent: CodeAgent, prompt: str) -> tuple[str, str, str]:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        response = agent.run(prompt)

    return str(response), clean_agent_text(stdout_buffer.getvalue()), clean_agent_text(stderr_buffer.getvalue())


def main() -> int:
    agent = load_agent()

    monitor = {
        "task": PROGRAMMING_TASK,
        "model_id": MODEL_ID,
        "started_at": now_iso(),
        "ended_at": None,
        "attempts": [],
        "final_success": False,
        "final_solution_file": None,
    }

    feedback = ""
    max_attempts = 4

    for attempt_number in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt_number} of {max_attempts}")

        prompt = f"""
You are solving a Python programming task.

Return only Python code. Do not include explanation.

Task:
{PROGRAMMING_TASK}

Previous test feedback:
{feedback if feedback else "No previous feedback. This is the first attempt."}
"""

        attempt_start = time.time()

        attempt_record = {
            "attempt_number": attempt_number,
            "started_at": now_iso(),
            "prompt": prompt,
            "agent_response": None,
            "agent_console_output": None,
            "agent_error_output": None,
            "candidate_code": None,
            "test_passed": False,
            "test_feedback": None,
            "elapsed_seconds": None,
        }

        try:
            raw_response, console_output, error_output = ask_agent_for_solution(agent, prompt)
            candidate_code = extract_code(raw_response)

            passed, test_feedback = run_tests(candidate_code)

            attempt_record["agent_response"] = raw_response
            attempt_record["agent_console_output"] = console_output
            attempt_record["agent_error_output"] = error_output
            attempt_record["candidate_code"] = candidate_code
            attempt_record["test_passed"] = passed
            attempt_record["test_feedback"] = test_feedback

            print(test_feedback)

            if passed:
                with open(SOLUTION_FILE, "w", encoding="utf-8") as f:
                    f.write(candidate_code)
                    f.write("\n")

                monitor["final_success"] = True
                monitor["final_solution_file"] = SOLUTION_FILE
                break

            feedback = test_feedback

        except Exception as exc:
            error_text = traceback.format_exc()

            attempt_record["agent_response"] = None
            attempt_record["test_passed"] = False
            attempt_record["test_feedback"] = error_text

            feedback = error_text

            print(f"Attempt failed with exception: {exc}")

        finally:
            attempt_record["elapsed_seconds"] = round(time.time() - attempt_start, 3)
            attempt_record["ended_at"] = now_iso()

            monitor["attempts"].append(attempt_record)

            with open(MONITOR_FILE, "w", encoding="utf-8") as f:
                json.dump(monitor, f, indent=4, ensure_ascii=False)

    monitor["ended_at"] = now_iso()

    with open(MONITOR_FILE, "w", encoding="utf-8") as f:
        json.dump(monitor, f, indent=4, ensure_ascii=False)

    print(f"\nMonitor written to: {MONITOR_FILE}")

    if monitor["final_success"]:
        print(f"Final passing solution written to: {SOLUTION_FILE}")
        return 0

    print("No satisfactory solution found.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())