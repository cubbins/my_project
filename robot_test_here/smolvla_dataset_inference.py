#!/usr/bin/env python3
"""
Offline SmolVLA inference using one recorded LeRobot dataset frame.

This program:

1. Makes the FFmpeg shared DLLs visible to Python.
2. Loads lerobot/smolvla_base.
3. Loads episode 0 from lerobot/libero.
4. Retrieves one recorded observation.
5. Preprocesses the observation.
6. Predicts one robot action.
7. Postprocesses the predicted action.
8. Compares it with the recorded dataset action.
9. Saves a JSON report.

No physical robot is connected or controlled.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths and model settings
# ---------------------------------------------------------------------------

FFMPEG_SHARED_BIN = Path(r"C:\ffmpeg-shared\bin")
OUTPUT_DIR = Path(r"C:\robot_test_here")
OUTPUT_REPORT = OUTPUT_DIR / "smolvla_dataset_inference_report.json"

MODEL_ID = "lerobot/smolvla_base"
DATASET_ID = "lerobot/libero"

EPISODE_INDEX = 0
FRAME_OFFSET_WITHIN_EPISODE = 0


# Keep this handle alive for the entire process.
_dll_directory_handle = None


def configure_windows_ffmpeg_dlls() -> None:
    """
    Add the shared FFmpeg directory to the Windows DLL search path.

    TorchCodec dynamically loads FFmpeg shared libraries such as:

        avcodec-62.dll
        avformat-62.dll
        avutil-60.dll

    This must happen before importing LeRobot dataset modules that may invoke
    TorchCodec.
    """
    global _dll_directory_handle

    if os.name != "nt":
        return

    if not FFMPEG_SHARED_BIN.is_dir():
        raise FileNotFoundError(
            f"FFmpeg shared-library directory was not found:\n"
            f"{FFMPEG_SHARED_BIN}"
        )

    required_dll_patterns = (
        "avcodec-*.dll",
        "avformat-*.dll",
        "avutil-*.dll",
        "swscale-*.dll",
        "swresample-*.dll",
    )

    missing_patterns = [
        pattern
        for pattern in required_dll_patterns
        if not list(FFMPEG_SHARED_BIN.glob(pattern))
    ]

    if missing_patterns:
        raise RuntimeError(
            "The FFmpeg directory does not contain all required DLL families:\n"
            + "\n".join(missing_patterns)
        )

    _dll_directory_handle = os.add_dll_directory(
        str(FFMPEG_SHARED_BIN)
    )

    # Put the shared build first for child processes and secondary lookups.
    os.environ["PATH"] = (
        str(FFMPEG_SHARED_BIN)
        + os.pathsep
        + os.environ.get("PATH", "")
    )


configure_windows_ffmpeg_dlls()

# These imports deliberately occur after configure_windows_ffmpeg_dlls().
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def bytes_to_gib(value: int) -> float:
    """Convert bytes to gibibytes."""
    return value / (1024**3)


def tensor_description(value: torch.Tensor) -> dict[str, Any]:
    """Return JSON-compatible tensor information."""
    tensor = value.detach().cpu()

    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }

    if tensor.numel() > 0 and tensor.is_floating_point():
        result.update(
            {
                "minimum": float(tensor.min().item()),
                "maximum": float(tensor.max().item()),
                "mean": float(tensor.float().mean().item()),
            }
        )

    return result


def describe_value(name: str, value: Any) -> None:
    """Print a compact description of a dataset field."""
    if isinstance(value, torch.Tensor):
        print(
            f"{name:<42} "
            f"shape={tuple(value.shape)!s:<18} "
            f"dtype={str(value.dtype):<15} "
            f"device={value.device}"
        )
    elif isinstance(value, str):
        text = value if len(value) <= 100 else value[:97] + "..."
        print(f"{name:<42} {text!r}")
    else:
        print(f"{name:<42} {type(value).__name__}: {value}")


def print_cuda_memory(label: str) -> dict[str, float]:
    """Print and return current CUDA memory measurements."""
    if not torch.cuda.is_available():
        return {}

    allocated = bytes_to_gib(torch.cuda.memory_allocated())
    reserved = bytes_to_gib(torch.cuda.memory_reserved())
    peak = bytes_to_gib(torch.cuda.max_memory_allocated())

    print(f"\n{label}")
    print("-" * len(label))
    print(f"Allocated: {allocated:.3f} GiB")
    print(f"Reserved:  {reserved:.3f} GiB")
    print(f"Peak:      {peak:.3f} GiB")

    return {
        "allocated_gib": allocated,
        "reserved_gib": reserved,
        "peak_allocated_gib": peak,
    }


def load_first_episode() -> LeRobotDataset:
    """
    Load only episode 0 where supported.

    Restricting the episode limits the amount of dataset material needed for
    this first test. The fallback accommodates installations whose constructor
    does not accept an `episodes` argument.
    """
    try:
        print(
            f"Loading dataset {DATASET_ID!r}, "
            f"restricted to episode {EPISODE_INDEX}..."
        )

        return LeRobotDataset(
            DATASET_ID,
            episodes=[EPISODE_INDEX],
        )

    except TypeError as exc:
        print(
            "This LeRobotDataset constructor did not accept the "
            "'episodes' argument."
        )
        print("Falling back to the normal dataset constructor.")
        print("Original message:", exc)

        return LeRobotDataset(DATASET_ID)


def determine_frame_index(dataset: LeRobotDataset) -> tuple[int, int, int]:
    """
    Determine the absolute frame range for the requested episode.

    Returns:
        selected_frame_index, episode_start_index, episode_end_index
    """
    episodes = dataset.meta.episodes

    try:
        start_value = episodes["dataset_from_index"][EPISODE_INDEX]
        end_value = episodes["dataset_to_index"][EPISODE_INDEX]

        episode_start = int(
            start_value.item()
            if hasattr(start_value, "item")
            else start_value
        )
        episode_end = int(
            end_value.item()
            if hasattr(end_value, "item")
            else end_value
        )

    except Exception as exc:
        print(
            "Could not read episode-boundary columns from metadata; "
            "using dataset frame 0."
        )
        print("Metadata message:", exc)

        episode_start = 0
        episode_end = len(dataset)

    selected_index = episode_start + FRAME_OFFSET_WITHIN_EPISODE

    if selected_index >= episode_end:
        raise IndexError(
            f"Requested frame offset {FRAME_OFFSET_WITHIN_EPISODE} is outside "
            f"episode {EPISODE_INDEX}, whose absolute range is "
            f"[{episode_start}, {episode_end})."
        )

    if selected_index >= len(dataset):
        raise IndexError(
            f"Selected dataset index {selected_index} is outside the loaded "
            f"dataset length {len(dataset)}."
        )

    return selected_index, episode_start, episode_end


def make_processors(
    policy: SmolVLAPolicy,
    dataset: LeRobotDataset,
    device: torch.device,
):
    """
    Create the model's preprocessing and postprocessing pipelines.

    The first attempt supplies dataset normalization statistics. The fallback
    uses the processor information stored with the pretrained model.
    """
    common_arguments = {
        "preprocessor_overrides": {
            "device_processor": {
                "device": str(device),
            }
        }
    }

    try:
        return make_pre_post_processors(
            policy.config,
            MODEL_ID,
            dataset_stats=dataset.meta.stats,
            **common_arguments,
        )

    except TypeError as exc:
        print(
            "\nProcessor factory did not accept dataset_stats; "
            "using the model's saved processor configuration."
        )
        print("Original message:", exc)

        return make_pre_post_processors(
            policy.config,
            MODEL_ID,
            **common_arguments,
        )


def convert_tensor_to_list(value: Any) -> Any:
    """Convert tensors and simple values into JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable.", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    print("=" * 78)
    print("SmolVLA recorded-dataset inference test")
    print("=" * 78)
    print("Model:", MODEL_ID)
    print("Dataset:", DATASET_ID)
    print("Episode:", EPISODE_INDEX)
    print("Frame offset:", FRAME_OFFSET_WITHIN_EPISODE)
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("FFmpeg shared directory:", FFMPEG_SHARED_BIN)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # -----------------------------------------------------------------------
    # Load the model
    # -----------------------------------------------------------------------

    print("\nLoading SmolVLA policy...")
    load_start = time.perf_counter()

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy = policy.to(device)
    policy.eval()

    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start

    print(f"Model loaded in {model_load_seconds:.2f} seconds.")
    memory_after_model_load = print_cuda_memory("CUDA after model loading")

    # -----------------------------------------------------------------------
    # Load the dataset
    # -----------------------------------------------------------------------

    dataset_start = time.perf_counter()
    dataset = load_first_episode()
    dataset_load_seconds = time.perf_counter() - dataset_start

    print(f"\nDataset initialized in {dataset_load_seconds:.2f} seconds.")
    print("Loaded dataset length:", len(dataset))
    print("Dataset FPS:", getattr(dataset.meta, "fps", "unknown"))

    selected_index, episode_start, episode_end = determine_frame_index(
        dataset
    )

    print("Episode absolute start:", episode_start)
    print("Episode absolute end:", episode_end)
    print("Selected absolute frame:", selected_index)

    # Accessing the frame causes LeRobot/TorchCodec to decode the camera images.
    print("\nReading and decoding the selected dataset frame...")
    frame_start = time.perf_counter()

    frame = dict(dataset[selected_index])

    frame_load_seconds = time.perf_counter() - frame_start
    print(f"Frame loaded in {frame_load_seconds:.3f} seconds.")

    print("\nRaw dataset fields")
    print("-" * 78)

    for key in sorted(frame):
        describe_value(key, frame[key])

    # -----------------------------------------------------------------------
    # Create processors and preprocess the recorded observation
    # -----------------------------------------------------------------------

    print("\nCreating policy processors...")
    preprocess, postprocess = make_processors(
        policy=policy,
        dataset=dataset,
        device=device,
    )

    print("Preprocessing the recorded observation...")
    processed_frame = preprocess(frame)

    print("\nProcessed policy fields")
    print("-" * 78)

    for key in sorted(processed_frame):
        describe_value(key, processed_frame[key])

    # Reset clears any previously queued action chunk.
    if hasattr(policy, "reset"):
        policy.reset()

     # -----------------------------------------------------------------------
    # Run inference
    # -----------------------------------------------------------------------

    print("\nRunning SmolVLA action inference...")

    # Map the LIBERO dataset camera names to the camera names expected
    # by the pretrained SmolVLA configuration.
    if "observation.images.image" in processed_frame:
        processed_frame["observation.images.camera1"] = (
            processed_frame["observation.images.image"]
        )

    if "observation.images.image2" in processed_frame:
        processed_frame["observation.images.camera2"] = (
            processed_frame["observation.images.image2"]
        )

    # The pretrained policy expects three camera inputs, while this
    # LIBERO sample provides only two. For this diagnostic test only,
    # duplicate camera2 as camera3 so the model can proceed.
    if (
        "observation.images.camera3" not in processed_frame
        and "observation.images.camera2" in processed_frame
    ):
        processed_frame["observation.images.camera3"] = (
            processed_frame["observation.images.camera2"]
        )

    print("\nImage keys supplied to SmolVLA:")
    for key in sorted(processed_frame):
        if key.startswith("observation.images."):
            print(" ", key, tuple(processed_frame[key].shape))

    torch.cuda.synchronize()
    inference_start = time.perf_counter()

    with torch.inference_mode():
        predicted_action_normalized = policy.select_action(processed_frame)
        predicted_action = postprocess(predicted_action_normalized)

    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start

    print(f"Inference completed in {inference_seconds:.4f} seconds.")

    predicted_cpu = predicted_action.detach().float().cpu()

    print("\nPredicted action")
    print("-" * 78)
    print("Shape:", tuple(predicted_cpu.shape))
    print("Dtype:", predicted_cpu.dtype)
    print(predicted_cpu)

    # -----------------------------------------------------------------------
    # Compare with the action recorded in the dataset
    # -----------------------------------------------------------------------

    recorded_action = frame.get("action")
    comparison: dict[str, Any] | None = None

    if isinstance(recorded_action, torch.Tensor):
        recorded_cpu = recorded_action.detach().float().cpu()

        print("\nRecorded dataset action")
        print("-" * 78)
        print("Shape:", tuple(recorded_cpu.shape))
        print("Dtype:", recorded_cpu.dtype)
        print(recorded_cpu)

        predicted_flat = predicted_cpu.reshape(-1)
        recorded_flat = recorded_cpu.reshape(-1)

        comparable_length = min(
            predicted_flat.numel(),
            recorded_flat.numel(),
        )

        predicted_comparable = predicted_flat[:comparable_length]
        recorded_comparable = recorded_flat[:comparable_length]

        absolute_error = (
            predicted_comparable - recorded_comparable
        ).abs()

        comparison = {
            "comparable_elements": comparable_length,
            "mean_absolute_error": float(
                absolute_error.mean().item()
            ),
            "maximum_absolute_error": float(
                absolute_error.max().item()
            ),
        }

        print("\nSingle-frame comparison")
        print("-" * 78)
        print(
            "Comparable elements:",
            comparison["comparable_elements"],
        )
        print(
            "Mean absolute error:",
            f"{comparison['mean_absolute_error']:.6f}",
        )
        print(
            "Maximum absolute error:",
            f"{comparison['maximum_absolute_error']:.6f}",
        )
    else:
        print(
            "\nNo tensor-valued 'action' field was found in this frame, "
            "so no recorded-action comparison was made."
        )

    memory_after_inference = print_cuda_memory(
        "CUDA after dataset inference"
    )

    # -----------------------------------------------------------------------
    # Save a structured report
    # -----------------------------------------------------------------------

    report = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "episode_index": EPISODE_INDEX,
        "episode_start_index": episode_start,
        "episode_end_index": episode_end,
        "selected_frame_index": selected_index,
        "model_load_seconds": model_load_seconds,
        "dataset_load_seconds": dataset_load_seconds,
        "frame_load_seconds": frame_load_seconds,
        "inference_seconds": inference_seconds,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "predicted_action": convert_tensor_to_list(predicted_action),
        "recorded_action": convert_tensor_to_list(recorded_action),
        "comparison": comparison,
        "memory_after_model_load": memory_after_model_load,
        "memory_after_inference": memory_after_inference,
        "raw_frame_fields": {
            key: (
                tensor_description(value)
                if isinstance(value, torch.Tensor)
                else {
                    "type": type(value).__name__,
                    "value": convert_tensor_to_list(value),
                }
            )
            for key, value in frame.items()
        },
    }

    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nReport written to:")
    print(OUTPUT_REPORT)

    print("\nNo physical robot was connected or controlled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())