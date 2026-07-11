

"""
Example 1 — Electronics paper
python essay_generator.py --field electronics --topic "Op-amp stability"


Example 2 — Economics paper
python essay_generator.py --field economics --topic "Monetary policy transmission"

Example 3 — Outline only
python essay_generator.py --field chemistry --topic "Catalyst poisoning" --outline_only

Example 4 — Using OpenAI backend
python essay_generator.py --field political_science --topic "Alliance formation" --backend openai --model gpt-4.1

python essay_generator.py --field electronics --topic "Op-amp stability" \
    --backend local \
    --model Qwen/Qwen2.5-3B-Instruct

python essay_generator.py --field electronics --topic "Op-amp stability" \
    --output_dir opamp_stability_run \
    --output opamp_stability.md


now run this:

python essay_generator.py --field electronics --topic "Op-amp stability" --output_dir essay_output

for 7B

python essay_generator.py --field electronics --topic "Op-amp stability" --model Qwen/Qwen2.5-7B-Instruct --output_dir essay_output

Qwen2.5‑3B‑Instruct

python essay_generator.py --field electronics --topic "Op-amp stability" --model Qwen/Qwen2.5-3B-Instruct --output_dir essay_output

python essay_generator2.py \
  --field literary_fiction \
  --topic "A pilgrim reflects on youth, knowledge, exile, and inward awakening" \
  --pdfs hesse_pdfs \
  --mode literary_influence \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir hesse_influenced_run \
  --output literary_essay.md

python essay_generator2_corrected.py \
  --field literary_fiction \
  --topic "A pilgrim reflects on youth, knowledge, exile, and inward awakening" \
  --pdfs hesse_pdfs \
  --mode literary_influence \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir hesse_influenced_run \
  --output literary_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 500

  
python essay_generator2_corrected.py \
  --field literary_fiction \
  --topic "A man in debt wanders through the city while arguing with his own conscience" \
  --pdfs dostoevsky_pdfs \
  --mode literary_influence \
  --style_author dostoevsky \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir dostoevsky_influenced_run \
  --output psychological_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 700

python essay_7_7_6_24AM.py \
  --field literary_fiction \
  --topic "style profile only" \
  --pdfs hesse_pdfs \
  --build_style_profile \
  --output_dir hesse_profile  

python essay_7_7_6_24AM.py \
  --field literary_fiction \
  --topic "A man in debt wanders through the city while arguing with his own conscience" \
  --pdfs dostoevsky_pdfs \
  --mode literary_influence \
  --style_author dostoevsky \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir dostoevsky_influenced_run \
  --output psychological_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 700

  python essay_7_7_10_09_dynamic_thought_memory_audit.py ... --use_instruction_memory


  

python essay_7_7_10_09_dynamic_thought_memory_audit.py \
  --field literary_fiction \
  --topic "A man in debt wanders through the city while arguing with his own conscience" \
  --pdfs dostoevsky_pdfs \
  --mode literary_influence \
  --style_author dostoevsky \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir dostoevsky_influenced_run \
  --output psychological_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 700
  --use_instruction_memory

python essay_7_8_6_48.py \
  --field literary_fiction \
  --topic "A man in debt wanders through the city while arguing with his own conscience" \
  --pdfs dostoevsky_pdfs \
  --mode literary_influence \
  --style_author dostoevsky \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir dostoevsky_influenced_run \
  --output psychological_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 700
  --use_instruction_memory

  I want 'powershell' query the network. 


"""



import os
 
from datetime import datetime
import re
import glob
import uuid
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LLM backends ---------------- #

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4.1"):
        if OpenAI is None:
            raise ImportError("openai package not installed.")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content


class LocalTransformersBackend:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Newer transformers prefers dtype over torch_dtype.
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 max_new_tokens: int = 600,
                 temperature: float = 0.4):
        """
        Generate only the assistant continuation, not the echoed prompt.
        Also uses inference_mode() to reduce GPU memory pressure.
        """
        prompt = (
            f"<system>\n{system_prompt}\n</system>\n"
            f"<user>\n{user_prompt}\n</user>\n"
            f"<assistant>\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6000).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode only newly generated tokens so the output does not include
        # the system/user prompt again.
        generated_ids = output[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return text


class LLMRouter:
    def __init__(self, backend: str = "openai", model: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIBackend(model=model or "gpt-4.1")
        elif backend == "local":
            if model is None:
                raise ValueError("Local backend requires model path.")
            self.llm = LocalTransformersBackend(model_path=model)
        else:
            raise ValueError("Unknown backend")

    def call(self, system_prompt: str, user_prompt: str, **generate_kwargs) -> str:
        return self.llm.generate(system_prompt, user_prompt, **generate_kwargs)


# ---------------- Persistent intermediate instruction memory ---------------- #

class IntermediateInstructionMemory:
    """
    Saves intermediate LLM call data to disk and reloads compact string memory
    on later runs.

    New in this version:
    - Builds a memory-composition audit for each LLM call.
    - Records exactly which prior memory fragments were inserted.
    - Saves fragment type, timestamp, character count, thought seconds,
      thought weight, and preview text.
    - Writes per-run reports showing the degree of retained memory used.
    """

    def __init__(self,
                 memory_dir: str = "instruction_memory",
                 memory_file: str = "intermediate_memory.jsonl",
                 max_loaded_items: int = 6,
                 max_memory_chars: int = 5000,
                 enabled: bool = True):
        self.memory_dir = memory_dir
        self.memory_file = os.path.join(memory_dir, memory_file)
        self.max_loaded_items = max_loaded_items
        self.max_memory_chars = max_memory_chars
        self.enabled = enabled

        # A run_id groups all neural-network calls made during one execution
        # of the primary program. This lets us create a per-run report.
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_run_records: List[Dict[str, Any]] = []
        self.current_run_compositions: List[Dict[str, Any]] = []

        if self.enabled:
            os.makedirs(self.memory_dir, exist_ok=True)

    def load_items(self) -> List[Dict[str, Any]]:
        if not self.enabled or not os.path.exists(self.memory_file):
            return []

        items: List[Dict[str, Any]] = []
        with open(self.memory_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return items[-self.max_loaded_items:]

    def build_memory_bundle(self) -> Dict[str, Any]:
        """
        Return both:
        - memory_text: the actual string inserted into the LLM prompt
        - pieces: structured audit data describing every inserted fragment

        This is the key diagnostic function for memory retention.
        """
        items = self.load_items()
        if not items:
            return {
                "memory_text": "",
                "pieces": [],
                "raw_memory_chars_before_cap": 0,
                "inserted_memory_chars": 0,
                "truncated_by_char_cap": False,
            }

        blocks = []
        pieces = []

        for i, item in enumerate(items, start=1):
            call_type = item.get("call_type", "llm_call")
            timestamp = item.get("timestamp", "")
            source_run_id = item.get("run_id", "")
            thought_seconds = float(item.get("thought_seconds", 0.0) or 0.0)
            thought_weight = float(item.get("thought_weight", 1.0) or 1.0)
            output = item.get("output", "").strip()
            output = re.sub(r"\s+", " ", output)

            original_output_chars = len(output)

            # Keep each remembered output compact.
            retained_output = output[:900]
            retained_output_chars = len(retained_output)

            header = (
                f"[Memory item {i}; type={call_type}; time={timestamp}; "
                f"source_run_id={source_run_id}; "
                f"thought_seconds={thought_seconds:.3f}; "
                f"weight={thought_weight:.3f}; "
                f"retained_chars={retained_output_chars}; "
                f"original_output_chars={original_output_chars}]\n"
            )
            block = f"{header}{retained_output}\n"
            blocks.append(block)

            pieces.append({
                "memory_item_number": i,
                "source_run_id": source_run_id,
                "source_timestamp": timestamp,
                "source_call_type": call_type,
                "thought_seconds": round(thought_seconds, 6),
                "thought_weight": round(thought_weight, 6),
                "original_output_chars": original_output_chars,
                "retained_output_chars": retained_output_chars,
                "block_chars_before_global_cap": len(block),
                "retention_ratio": round(retained_output_chars / max(original_output_chars, 1), 6),
                "preview": retained_output[:300],
            })

        raw_memory_text = "\n".join(blocks).strip()
        raw_memory_chars = len(raw_memory_text)
        truncated = False
        memory_text = raw_memory_text

        if len(memory_text) > self.max_memory_chars:
            truncated = True
            # Keep the newest material at the end, consistent with the older behavior.
            memory_text = memory_text[-self.max_memory_chars:]

        wrapped_memory_text = f"""
PERSISTENT INTERMEDIATE MEMORY FROM PRIOR RUNS

The following items are compressed intermediate results saved from earlier
executions of this program. Treat them as guidance, not as source truth.
Use them to preserve continuity of goals, style, constraints, and prior
generation behavior.

{memory_text}

END PERSISTENT INTERMEDIATE MEMORY
"""

        return {
            "memory_text": wrapped_memory_text,
            "pieces": pieces,
            "raw_memory_chars_before_cap": raw_memory_chars,
            "inserted_memory_chars": len(wrapped_memory_text),
            "max_memory_chars": self.max_memory_chars,
            "truncated_by_char_cap": truncated,
        }

    def build_memory_string(self) -> str:
        """
        Backward-compatible helper. Existing code can still request only
        the string form of the memory.
        """
        return self.build_memory_bundle()["memory_text"]

    def save_memory_composition(self,
                                call_type: str,
                                pieces: List[Dict[str, Any]],
                                memory_text: str,
                                raw_memory_chars_before_cap: int,
                                inserted_memory_chars: int,
                                truncated_by_char_cap: bool,
                                prompt_chars_without_memory: int,
                                prompt_chars_with_memory: int) -> None:
        """
        Save an audit record showing what memory was inserted into this
        particular neural-network call.
        """
        if not self.enabled:
            return

        total_piece_retained_chars = sum(p.get("retained_output_chars", 0) for p in pieces)
        total_piece_original_chars = sum(p.get("original_output_chars", 0) for p in pieces)
        total_piece_thought_seconds = sum(p.get("thought_seconds", 0.0) for p in pieces)
        total_piece_thought_weight = sum(p.get("thought_weight", 0.0) for p in pieces)

        record = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "call_type": call_type,
            "inserted_piece_count": len(pieces),
            "raw_memory_chars_before_cap": raw_memory_chars_before_cap,
            "inserted_memory_chars": inserted_memory_chars,
            "max_memory_chars": self.max_memory_chars,
            "truncated_by_char_cap": truncated_by_char_cap,
            "prompt_chars_without_memory": prompt_chars_without_memory,
            "prompt_chars_with_memory": prompt_chars_with_memory,
            "memory_fraction_of_augmented_prompt": round(
                inserted_memory_chars / max(prompt_chars_with_memory, 1), 6
            ),
            "total_piece_original_chars": total_piece_original_chars,
            "total_piece_retained_chars": total_piece_retained_chars,
            "piece_retention_ratio": round(
                total_piece_retained_chars / max(total_piece_original_chars, 1), 6
            ),
            "total_piece_thought_seconds": round(total_piece_thought_seconds, 6),
            "total_piece_thought_weight": round(total_piece_thought_weight, 6),
            "pieces": pieces,
            "memory_text_preview": memory_text[:1200],
        }

        self.current_run_compositions.append(record)

    def save_call(self,
                  call_type: str,
                  system_prompt: str,
                  user_prompt: str,
                  output: str,
                  metadata: Optional[Dict[str, Any]] = None,
                  thought_seconds: float = 0.0,
                  prompt_chars: int = 0,
                  output_chars: int = 0) -> None:
        if not self.enabled:
            return

        # Fictitious metric: "time of thought" is wall-clock time spent
        # inside the neural-network call as seen by the overriding program.
        # Weight is a simple combined measure of time and text volume.
        thought_weight = thought_seconds * max(output_chars, 1) / 1000.0

        record = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "call_type": call_type,
            "thought_seconds": round(thought_seconds, 6),
            "thought_weight": round(thought_weight, 6),
            "prompt_chars": prompt_chars,
            "output_chars": output_chars,
            "metadata": metadata or {},
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": output,
        }

        self.current_run_records.append(record)

        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summarize_current_run(self) -> Dict[str, Any]:
        records = self.current_run_records
        compositions = self.current_run_compositions
        total_calls = len(records)
        total_thought_seconds = sum(r.get("thought_seconds", 0.0) for r in records)
        total_thought_weight = sum(r.get("thought_weight", 0.0) for r in records)
        total_prompt_chars = sum(r.get("prompt_chars", 0) for r in records)
        total_output_chars = sum(r.get("output_chars", 0) for r in records)

        if total_calls:
            average_thought_seconds = total_thought_seconds / total_calls
            max_thought_seconds = max(r.get("thought_seconds", 0.0) for r in records)
            min_thought_seconds = min(r.get("thought_seconds", 0.0) for r in records)
        else:
            average_thought_seconds = 0.0
            max_thought_seconds = 0.0
            min_thought_seconds = 0.0

        total_inserted_memory_chars = sum(c.get("inserted_memory_chars", 0) for c in compositions)
        total_inserted_piece_count = sum(c.get("inserted_piece_count", 0) for c in compositions)
        total_memory_thought_seconds = sum(c.get("total_piece_thought_seconds", 0.0) for c in compositions)
        total_memory_thought_weight = sum(c.get("total_piece_thought_weight", 0.0) for c in compositions)

        if compositions:
            average_memory_fraction = sum(
                c.get("memory_fraction_of_augmented_prompt", 0.0) for c in compositions
            ) / len(compositions)
        else:
            average_memory_fraction = 0.0

        return {
            "run_id": self.run_id,
            "total_calls": total_calls,
            "total_thought_seconds": round(total_thought_seconds, 6),
            "average_thought_seconds": round(average_thought_seconds, 6),
            "min_thought_seconds": round(min_thought_seconds, 6),
            "max_thought_seconds": round(max_thought_seconds, 6),
            "total_thought_weight": round(total_thought_weight, 6),
            "total_prompt_chars": total_prompt_chars,
            "total_output_chars": total_output_chars,
            "memory_composition_summary": {
                "composition_records": len(compositions),
                "total_inserted_piece_count": total_inserted_piece_count,
                "total_inserted_memory_chars": total_inserted_memory_chars,
                "total_memory_thought_seconds_reused": round(total_memory_thought_seconds, 6),
                "total_memory_thought_weight_reused": round(total_memory_thought_weight, 6),
                "average_memory_fraction_of_augmented_prompt": round(average_memory_fraction, 6),
            },
            "calls": records,
            "memory_compositions": compositions,
        }

    def write_run_report(self, output_dir: str) -> None:
        if not self.enabled:
            return

        os.makedirs(output_dir, exist_ok=True)
        summary = self.summarize_current_run()

        json_path = os.path.join(output_dir, "memory_thought_report.json")
        md_path = os.path.join(output_dir, "memory_thought_report.md")
        composition_json_path = os.path.join(output_dir, "memory_composition_report.json")
        composition_md_path = os.path.join(output_dir, "memory_composition_report.md")
        composition_txt_path = os.path.join(output_dir, "memory_inserted_pieces.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        lines = []
        lines.append("# Memory Thought Report")
        lines.append("")
        lines.append(f"Run ID: `{summary['run_id']}`")
        lines.append(f"Total neural-network calls: {summary['total_calls']}")
        lines.append(f"Total time of thought: {summary['total_thought_seconds']:.3f} seconds")
        lines.append(f"Average time of thought: {summary['average_thought_seconds']:.3f} seconds")
        lines.append(f"Minimum time of thought: {summary['min_thought_seconds']:.3f} seconds")
        lines.append(f"Maximum time of thought: {summary['max_thought_seconds']:.3f} seconds")
        lines.append(f"Total thought weight: {summary['total_thought_weight']:.3f}")
        lines.append(f"Total prompt characters: {summary['total_prompt_chars']}")
        lines.append(f"Total output characters: {summary['total_output_chars']}")
        lines.append("")
        msum = summary["memory_composition_summary"]
        lines.append("## Memory composition summary")
        lines.append("")
        lines.append(f"Composition records: {msum['composition_records']}")
        lines.append(f"Inserted memory pieces: {msum['total_inserted_piece_count']}")
        lines.append(f"Inserted memory characters: {msum['total_inserted_memory_chars']}")
        lines.append(f"Reused memory time of thought: {msum['total_memory_thought_seconds_reused']:.3f} seconds")
        lines.append(f"Reused memory thought weight: {msum['total_memory_thought_weight_reused']:.3f}")
        lines.append(f"Average memory fraction of augmented prompt: {msum['average_memory_fraction_of_augmented_prompt']:.3f}")
        lines.append("")
        lines.append("## Calls")
        lines.append("")
        lines.append("| # | Type | Thought seconds | Weight | Prompt chars | Output chars |")
        lines.append("|---:|---|---:|---:|---:|---:|")

        for i, r in enumerate(summary["calls"], start=1):
            lines.append(
                f"| {i} | {r.get('call_type', '')} | "
                f"{r.get('thought_seconds', 0.0):.3f} | "
                f"{r.get('thought_weight', 0.0):.3f} | "
                f"{r.get('prompt_chars', 0)} | "
                f"{r.get('output_chars', 0)} |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Separate, more detailed report: which exact memory pieces were combined
        # into each LLM prompt.
        composition_report = {
            "run_id": summary["run_id"],
            "memory_composition_summary": summary["memory_composition_summary"],
            "memory_compositions": summary["memory_compositions"],
        }

        with open(composition_json_path, "w", encoding="utf-8") as f:
            json.dump(composition_report, f, indent=2, ensure_ascii=False)

        comp_lines = []
        comp_lines.append("# Memory Composition Report")
        comp_lines.append("")
        comp_lines.append(f"Run ID: `{summary['run_id']}`")
        comp_lines.append("")
        comp_lines.append("This report shows the exact prior memory fragments inserted into each LLM call.")
        comp_lines.append("")
        comp_lines.append("## Summary")
        comp_lines.append("")
        comp_lines.append(f"Composition records: {msum['composition_records']}")
        comp_lines.append(f"Inserted memory pieces: {msum['total_inserted_piece_count']}")
        comp_lines.append(f"Inserted memory characters: {msum['total_inserted_memory_chars']}")
        comp_lines.append(f"Average memory fraction of augmented prompt: {msum['average_memory_fraction_of_augmented_prompt']:.3f}")
        comp_lines.append("")

        txt_lines = []
        txt_lines.append("MEMORY INSERTED PIECES")
        txt_lines.append(f"Run ID: {summary['run_id']}")
        txt_lines.append("")

        for ci, comp in enumerate(summary["memory_compositions"], start=1):
            comp_lines.append(f"## LLM call {ci}: `{comp.get('call_type', '')}`")
            comp_lines.append("")
            comp_lines.append(f"Inserted piece count: {comp.get('inserted_piece_count', 0)}")
            comp_lines.append(f"Inserted memory chars: {comp.get('inserted_memory_chars', 0)}")
            comp_lines.append(f"Memory fraction of augmented prompt: {comp.get('memory_fraction_of_augmented_prompt', 0.0):.3f}")
            comp_lines.append(f"Truncated by char cap: {comp.get('truncated_by_char_cap', False)}")
            comp_lines.append("")
            comp_lines.append("| Piece | Source call | Source run | Thought sec | Weight | Retained chars | Retention ratio |")
            comp_lines.append("|---:|---|---|---:|---:|---:|---:|")

            txt_lines.append("=" * 80)
            txt_lines.append(f"LLM call {ci}: {comp.get('call_type', '')}")
            txt_lines.append(f"Inserted piece count: {comp.get('inserted_piece_count', 0)}")
            txt_lines.append(f"Inserted memory chars: {comp.get('inserted_memory_chars', 0)}")
            txt_lines.append(f"Memory fraction: {comp.get('memory_fraction_of_augmented_prompt', 0.0):.6f}")
            txt_lines.append("")

            for piece in comp.get("pieces", []):
                comp_lines.append(
                    f"| {piece.get('memory_item_number', '')} | "
                    f"{piece.get('source_call_type', '')} | "
                    f"{piece.get('source_run_id', '')} | "
                    f"{piece.get('thought_seconds', 0.0):.3f} | "
                    f"{piece.get('thought_weight', 0.0):.3f} | "
                    f"{piece.get('retained_output_chars', 0)} | "
                    f"{piece.get('retention_ratio', 0.0):.3f} |"
                )

                txt_lines.append(f"--- Memory item {piece.get('memory_item_number', '')} ---")
                txt_lines.append(f"Source call type: {piece.get('source_call_type', '')}")
                txt_lines.append(f"Source run id: {piece.get('source_run_id', '')}")
                txt_lines.append(f"Source timestamp: {piece.get('source_timestamp', '')}")
                txt_lines.append(f"Thought seconds: {piece.get('thought_seconds', 0.0):.6f}")
                txt_lines.append(f"Thought weight: {piece.get('thought_weight', 0.0):.6f}")
                txt_lines.append(f"Original output chars: {piece.get('original_output_chars', 0)}")
                txt_lines.append(f"Retained output chars: {piece.get('retained_output_chars', 0)}")
                txt_lines.append(f"Retention ratio: {piece.get('retention_ratio', 0.0):.6f}")
                txt_lines.append("Preview:")
                txt_lines.append(piece.get("preview", ""))
                txt_lines.append("")

            comp_lines.append("")

        with open(composition_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(comp_lines))

        with open(composition_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

        print(f"Memory thought report saved: {md_path}")
        print(f"Memory thought JSON saved: {json_path}")
        print(f"Memory composition report saved: {composition_md_path}")
        print(f"Memory composition JSON saved: {composition_json_path}")
        print(f"Inserted memory pieces saved: {composition_txt_path}")


class MemoryAugmentedLLMRouter:
    """
    Wraps LLMRouter.

    Before each call:
        loads prior saved intermediate results
        appends them to the instruction stream
        records exactly which memory pieces were appended

    After each call:
        saves the new intermediate result to disk
        updates the per-run thought report
    """

    def __init__(self,
                 base_router: LLMRouter,
                 memory: IntermediateInstructionMemory):
        self.base_router = base_router
        self.memory = memory
        self.call_count = 0

    def call(self, system_prompt: str, user_prompt: str, **generate_kwargs) -> str:
        self.call_count += 1
        call_type = f"llm_call_{self.call_count}"

        memory_bundle = self.memory.build_memory_bundle()
        memory_text = memory_bundle["memory_text"]

        prompt_chars_without_memory = len(system_prompt) + len(user_prompt)

        if memory_text:
            augmented_user_prompt = f"""
{memory_text}

CURRENT REQUEST

{user_prompt}
"""
        else:
            augmented_user_prompt = user_prompt

        prompt_chars_with_memory = len(system_prompt) + len(augmented_user_prompt)

        # This audit records the exact memory pieces combined with the current prompt,
        # before the neural network is invoked.
        self.memory.save_memory_composition(
            call_type=call_type,
            pieces=memory_bundle["pieces"],
            memory_text=memory_text,
            raw_memory_chars_before_cap=memory_bundle["raw_memory_chars_before_cap"],
            inserted_memory_chars=memory_bundle["inserted_memory_chars"] if memory_text else 0,
            truncated_by_char_cap=memory_bundle["truncated_by_char_cap"],
            prompt_chars_without_memory=prompt_chars_without_memory,
            prompt_chars_with_memory=prompt_chars_with_memory,
        )

        start_time = time.perf_counter()
        output = self.base_router.call(
            system_prompt,
            augmented_user_prompt,
            **generate_kwargs
        )
        end_time = time.perf_counter()
        thought_seconds = end_time - start_time

        self.memory.save_call(
            call_type=call_type,
            system_prompt=system_prompt,
            user_prompt=augmented_user_prompt,
            output=output,
            metadata={
                "max_new_tokens": generate_kwargs.get("max_new_tokens"),
                "temperature": generate_kwargs.get("temperature"),
                "memory_piece_count_inserted": len(memory_bundle["pieces"]),
                "memory_chars_inserted": memory_bundle["inserted_memory_chars"] if memory_text else 0,
                "memory_truncated_by_char_cap": memory_bundle["truncated_by_char_cap"],
            },
            thought_seconds=thought_seconds,
            prompt_chars=prompt_chars_with_memory,
            output_chars=len(output),
        )

        print(
            f"[time_of_thought] call={self.call_count} "
            f"seconds={thought_seconds:.3f} "
            f"output_chars={len(output)} "
            f"memory_pieces={len(memory_bundle['pieces'])} "
            f"memory_chars={memory_bundle['inserted_memory_chars'] if memory_text else 0}"
        )

        return output


# ---------------- Data structures ---------------- #

@dataclass
class Chunk:
    id: str
    doc_id: str
    source_path: str
    page_num: int
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass
class SectionPlan:
    title: str
    goal: str
    user_notes: str


@dataclass
class SectionDraft:
    title: str
    text: str
    citations: List[str]


# ---------------- PDF ingestion ---------------- #

def load_pdfs(pdf_folder: str) -> List[str]:
    """Return PDF paths, including uppercase .PDF and PDFs in subdirectories."""
    patterns = [
        os.path.join(pdf_folder, "*.pdf"),
        os.path.join(pdf_folder, "*.PDF"),
        os.path.join(pdf_folder, "**", "*.pdf"),
        os.path.join(pdf_folder, "**", "*.PDF"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    return sorted(set(paths))


def extract_chunks_from_pdf(path: str,
                            doc_id: Optional[str] = None,
                            max_chars_per_chunk: int = 1200) -> List[Chunk]:
    doc = fitz.open(path)
    chunks: List[Chunk] = []
    doc_id = doc_id or str(uuid.uuid4())

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + max_chars_per_chunk, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        source_path=path,
                        page_num=page_num + 1,
                        text=chunk_text,
                    )
                )
            start = end

    return chunks


def build_corpus(pdf_folder: str) -> List[Chunk]:
    pdf_paths = load_pdfs(pdf_folder)
    corpus: List[Chunk] = []
    for path in pdf_paths:
        corpus.extend(extract_chunks_from_pdf(path))
    return corpus


# ---------------- Semantic index ---------------- #

class SemanticIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.emb_matrix: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[Chunk]):
        self.chunks.extend(chunks)

    def build(self):
        texts = [c.text for c in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        for c, e in zip(self.chunks, embeddings):
            c.embedding = e
        self.emb_matrix = embeddings

    def search(self, query: str, top_k: int = 8) -> List[RetrievalResult]:
        if self.emb_matrix is None:
            raise RuntimeError("Index not built yet.")
        q_emb = self.model.encode([query], convert_to_numpy=True)[0].reshape(1, -1)
        sims = cosine_similarity(q_emb, self.emb_matrix)[0]
        idxs = np.argsort(-sims)[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(sims[i]))
            for i in idxs
        ]


# ---------------- Essay generator core ---------------- #

class LegacyAcademicEssayGenerator:
    def __init__(self,
                 llm: LLMRouter,
                 index: SemanticIndex,
                 field: str,
                 citation_style: str = "APA"):
        self.llm = llm
        self.index = index
        self.field = field
        self.citation_style = citation_style

    # ---- Outline ---- #

    def build_outline(self, topic: str, constraints: str = "") -> str:
        system_prompt = "You are an expert academic writing assistant."
        user_prompt = f"""
You are helping a student write a {self.field} paper.

Topic: {topic}

Constraints:
{constraints}

Task:
1. Propose a clear, logical outline (sections + subsections).
2. For each section, briefly state its purpose.
3. Keep the outline suitable for a 10–15 page term paper or research paper.
"""
        return self.llm.call(system_prompt, user_prompt)

    # ---- Section drafting ---- #

    def draft_section(self,
                      section_plan: SectionPlan,
                      retrieval_queries: List[str]) -> SectionDraft:
        # Aggregate retrieval results
        retrieved_chunks: List[RetrievalResult] = []
        for q in retrieval_queries:
            retrieved_chunks.extend(self.index.search(q, top_k=5))

        # Deduplicate by chunk id
        seen = set()
        unique_chunks: List[RetrievalResult] = []
        for r in retrieved_chunks:
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                unique_chunks.append(r)

        context_blocks = []
        citation_labels = []
        for r in unique_chunks:
            c = r.chunk
            label = f"{os.path.basename(c.source_path)}, p.{c.page_num}"
            citation_labels.append(label)
            context_blocks.append(f"[{label}]\n{c.text}\n")

        context_text = "\n\n".join(context_blocks)

        system_prompt = "You are an academic writing assistant who grounds claims in provided sources."
        user_prompt = f"""
Section title: {section_plan.title}
Section goal: {section_plan.goal}

User's own notes / arguments:
{section_plan.user_notes}

Retrieved reference context (from PDFs, with page hints):
{context_text}

Instructions:
- Write a coherent draft of this section.
- Use the user's notes as the backbone of the argument.
- Integrate reference material only when it clearly supports or contrasts the argument.
- When you use a specific passage, include an inline citation marker like [Source, p.X]; do NOT invent bibliographic details.
- Maintain an academic tone appropriate for {self.citation_style} style, but do NOT output the full reference list.
- Avoid hallucinating facts; if the context does not support a claim, say that evidence is limited.
"""

        text = self.llm.call(system_prompt, user_prompt)
        return SectionDraft(
            title=section_plan.title,
            text=text,
            citations=sorted(set(citation_labels))
        )

    # ---- Final assembly ---- #

    def assemble_essay(self,
                       title: str,
                       sections: List[SectionDraft]) -> str:
        parts = [f"# {title}\n"]
        for s in sections:
            parts.append(f"\n## {s.title}\n\n{s.text}\n")

        # Simple reference list from citation labels
        parts.append("\n## References (placeholders)\n")
        for c in sorted({cit for s in sections for cit in s.citations}):
            parts.append(f"- {c}")

        return "\n".join(parts)


# ---------------- Example CLI-like usage ---------------- #


# ---------------- Essay generator core ---------------- #


# ---------------- Dynamic EssayGenerator loader ---------------- #

def load_essay_generator_class(module_path: Optional[str] = None):
    """
    Dynamically load EssayGenerator from a separate .py file on disk.

    By default, this loads essay_generator_class.py from the same directory
    as this primary program. This preserves the main pipeline while allowing
    EssayGenerator to be edited, generated, or replaced independently.
    """
    import importlib.util

    if module_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.join(here, "essay_generator_class.py")

    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f"Could not find EssayGenerator module: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "dynamic_essay_generator_class",
        module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)

    # Inject shared symbols from this primary program into the dynamically
    # loaded EssayGenerator module before execution. The extracted class
    # returns SectionDraft and refers to the same pipeline dataclasses used
    # by the main script, so the module must see those names at runtime.
    module.LLMRouter = LLMRouter
    module.SemanticIndex = SemanticIndex
    module.SectionPlan = SectionPlan
    module.SectionDraft = SectionDraft
    module.RetrievalResult = RetrievalResult

    spec.loader.exec_module(module)

    if not hasattr(module, "EssayGenerator"):
        raise AttributeError(
            f"Module {module_path} does not define class EssayGenerator"
        )

    return module.EssayGenerator


EssayGenerator = load_essay_generator_class()

# ---------------- Example CLI-like usage ---------------- #












import argparse

def cli():
    parser = argparse.ArgumentParser(
        description="Full essay generator: outline → sections → citations → final draft"
    )

    parser.add_argument(
        "--field",
        type=str,
        required=True,
        help="Academic field (e.g., electronics, chemistry, economics, political_science)"
    )

    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Essay topic (quoted string)"
    )

    parser.add_argument(
        "--pdfs",
        type=str,
        default="papers",
        help="Folder containing reference PDFs"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "local"],
        default="local",
        help="LLM backend to use"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="/models/Qwen2.5-7B-Instruct",
        help="Model name (OpenAI) or local model path (transformers)"
    )


    parser.add_argument(
        "--outline_only",
        action="store_true",
        help="Only generate outline and exit"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="final_essay.md",
        help="Filename for the final essay"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="essay_output",
        help="Directory to save essay output files"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["academic", "literary_influence"],
        default="academic",
        help="Writing mode"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of retrieved chunks per query. Lower values reduce GPU memory use."
    )

    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=9000,
        help="Maximum retrieved context characters sent to the LLM per section."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=600,
        help="Maximum generated tokens per LLM call."
    )

    parser.add_argument(
        "--style_author",
        type=str,
        choices=["generic", "hesse", "dostoevsky"],
        default="generic",
        help="High-level literary style profile to condition generation"
    )

    parser.add_argument(
        "--use_instruction_memory",
        action="store_true",
        help="Load saved intermediate LLM results and prepend them to future LLM calls."
    )

    parser.add_argument(
        "--instruction_memory_dir",
        type=str,
        default="instruction_memory",
        help="Directory where intermediate LLM results are saved and loaded."
    )

    parser.add_argument(
        "--instruction_memory_file",
        type=str,
        default="intermediate_memory.jsonl",
        help="JSONL file used to persist intermediate LLM results."
    )

    parser.add_argument(
        "--max_loaded_memory_items",
        type=int,
        default=6,
        help="Maximum number of prior intermediate results loaded into the next prompt."
    )

    parser.add_argument(
        "--max_memory_chars",
        type=int,
        default=5000,
        help="Maximum characters of persistent memory inserted into a prompt."
    )



    args = parser.parse_args()



    # ---- Load PDFs + build index ----
    print(f"Loading PDFs from: {args.pdfs}")
    corpus = build_corpus(args.pdfs)
    print(f"Loaded {len(corpus)} chunks.")

    if not corpus:
        abs_pdf_dir = os.path.abspath(args.pdfs)
        raise FileNotFoundError(
            f"No extractable PDF text was found in: {abs_pdf_dir}\n"
            "Check that the directory exists, contains .pdf or .PDF files, "
            "and that the PDFs contain selectable text rather than scanned images."
        )

    index = SemanticIndex()
    index.add_chunks(corpus)
    print("Building semantic index...")
    index.build()
    print("Index ready.")

    # ---- LLM backend ----
    llm_router = LLMRouter(
        backend=args.backend,
        model=args.model
    )

    memory = None
    if args.use_instruction_memory:
        memory = IntermediateInstructionMemory(
            memory_dir=args.instruction_memory_dir,
            memory_file=args.instruction_memory_file,
            max_loaded_items=args.max_loaded_memory_items,
            max_memory_chars=args.max_memory_chars,
            enabled=True
        )
        llm_router = MemoryAugmentedLLMRouter(
            base_router=llm_router,
            memory=memory
        )
        print(f"Instruction memory enabled: {memory.memory_file}")



    generator = EssayGenerator(
        llm=llm_router,
        index=index,
        field=args.field,
        citation_style="APA",
        mode=args.mode,
        style_author=args.style_author,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens
    )






    # ---- Outline ----
    print("\n=== OUTLINE ===\n")
    if args.mode == "literary_influence":
        outline_constraints = (
            "Create an original literary structure. Emphasize theme, tone, image-patterns, "
            "psychological development, and symbolic progression. Do not use academic term-paper framing."
        )
    else:
        outline_constraints = "Include theory, empirical evidence, and implications."

    outline = generator.build_outline(
        topic=args.topic,
        constraints=outline_constraints
    )
    print(outline)

    if args.outline_only:
        return

    # ---- Section plan generation (simple heuristic) ----
    # Later, this can be replaced with an LLM-based outline parser.
    if args.mode == "literary_influence":
        if args.style_author == "dostoevsky":
            section_titles = [
                "The First Agitation",
                "Pride and Poverty",
                "The Argument Within",
                "Humiliation",
                "Confession Without Relief",
                "Judgment and Unfinished Awakening",
            ]
            notes = f"""
My notes for this psychological movement:
- Focus on {args.topic}
- Use original prose, not academic explanation
- Emphasize guilt, pride, shame, debt, illness, social pressure, conscience, and unstable self-justification
- Prefer tense urban interiors and argumentative inward speech over serene nature imagery
- Do not imitate or copy any protected author directly
"""
        else:
            section_titles = [
                "Departure",
                "Youth and Restlessness",
                "Knowledge and Disillusionment",
                "Exile and Solitude",
                "Inward Awakening",
                "Return Without Arrival",
            ]
            notes = f"""
My notes for this literary movement:
- Focus on {args.topic}
- Use original prose, not academic explanation
- Emphasize inner conflict, symbolic landscape, memory, solitude, and gradual self-recognition
- Do not imitate or copy any protected author directly
"""

        section_plans = []
        for title in section_titles:
            section_plans.append(
                SectionPlan(
                    title=title,
                    goal=f"Develop the literary movement '{title}' within the topic: {args.topic}.",
                    user_notes=notes
                )
            )
    else:
        section_titles = [
            "Introduction",
            "Background & Theory",
            "Core Analysis",
            "Case Evidence",
            "Design / Implications",
            "Conclusion"
        ]
        section_plans = []
        for title in section_titles:
            section_plans.append(
                SectionPlan(
                    title=title,
                    goal=f"Explain the role of {args.topic} in {args.field}.",
                    user_notes=f"""
My notes for this section on {title}:
- Focus on {args.topic}
- Connect to {args.field}
- Maintain academic tone
"""
                )
            )

    # ---- Draft sections ----
    sections = []
    for plan in section_plans:
        print(f"\n--- Drafting section: {plan.title} ---")
        if args.mode == "literary_influence":
            if args.style_author == "dostoevsky":
                queries = [
                    args.topic,
                    plan.title,
                    "guilt pride shame poverty debt conscience confession fever illness city room",
                ]
            else:
                queries = [
                    args.topic,
                    plan.title,
                    "inward awakening solitude exile youth knowledge symbolic journey",
                ]
        else:
            queries = [
                args.topic,
                plan.title,
                args.field,
                "analysis",
                "technical details",
                "supply chain",
                "design",
            ]
        draft = generator.draft_section(plan, queries)
        sections.append(draft)

    # ---- Final essay ----
    final_essay = generator.assemble_essay(
        title=args.topic,
        sections=sections
    )


    print("\n=== FINAL ESSAY (Markdown) ===\n")
    print(final_essay)


    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save full essay
    with open(os.path.join(output_dir, args.output), "w", encoding="utf-8") as f:
        f.write(final_essay)

    # Save each section safely
    for s in sections:
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', s.title)
        filename = f"{safe_title}.md"
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(s.text)

    # Save per-run statistics about the fictitious "time of thought" metric.
    if memory is not None:
        memory.write_run_report(output_dir)






if __name__ == "__main__":
    cli()
