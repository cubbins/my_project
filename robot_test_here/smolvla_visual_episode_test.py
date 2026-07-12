#!/usr/bin/env python3
"""
Visual offline SmolVLA test using recorded LIBERO observations.

The program:

1. Registers the FFmpeg shared-library directory on Windows.
2. Loads the pretrained SmolVLA policy.
3. Loads one episode from the lerobot/libero dataset.
4. Selects evenly spaced frames from that episode.
5. Decodes the two recorded camera views.
6. Maps the LIBERO camera keys to the names expected by SmolVLA.
7. Duplicates the second camera as camera3 for diagnostic compatibility.
8. Runs one independent action prediction for each selected frame.
9. Compares the six predicted action values with the first six recorded values.
10. Saves camera images, JSON results, and a visual HTML report.

This is an offline diagnostic experiment. It does not connect to or control
a physical robot.

python smolvla_visual_episode_test.py --frames 12

python smolvla_visual_episode_test.py ^
  --frames 10 ^
  --output-dir C:\robot_test_here\visual_test_run_01

"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Windows FFmpeg shared-library configuration
# ---------------------------------------------------------------------------

DEFAULT_FFMPEG_SHARED_BIN = Path(r"C:\ffmpeg-shared\bin")

# Keep this object alive for the lifetime of the process.
_FFMPEG_DLL_HANDLE = None


def configure_ffmpeg_shared_libraries(ffmpeg_bin: Path) -> None:
    """
    Add the FFmpeg shared-library directory to Windows' DLL search path.

    This must happen before importing LeRobot dataset modules or TorchCodec.
    """
    global _FFMPEG_DLL_HANDLE

    if os.name != "nt":
        return

    if not ffmpeg_bin.is_dir():
        raise FileNotFoundError(
            f"FFmpeg shared-library directory does not exist:\n{ffmpeg_bin}"
        )

    required_patterns = (
        "avcodec-*.dll",
        "avformat-*.dll",
        "avutil-*.dll",
        "swscale-*.dll",
        "swresample-*.dll",
    )

    missing = [
        pattern
        for pattern in required_patterns
        if not list(ffmpeg_bin.glob(pattern))
    ]

    if missing:
        raise RuntimeError(
            "Required FFmpeg DLL families were not found:\n"
            + "\n".join(f"  {item}" for item in missing)
        )

    _FFMPEG_DLL_HANDLE = os.add_dll_directory(str(ffmpeg_bin))

    # Also place the directory first on PATH for secondary library lookups.
    os.environ["PATH"] = (
        str(ffmpeg_bin)
        + os.pathsep
        + os.environ.get("PATH", "")
    )


# Register DLLs before importing torchcodec-dependent modules.
configure_ffmpeg_shared_libraries(DEFAULT_FFMPEG_SHARED_BIN)

import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "lerobot/smolvla_base"
DEFAULT_DATASET_ID = "lerobot/libero"
DEFAULT_EPISODE_INDEX = 0
DEFAULT_FRAME_COUNT = 8
DEFAULT_OUTPUT_DIR = Path(r"C:\robot_test_here\smolvla_visual_report")


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def tensor_to_python(value: Any) -> Any:
    """Convert tensors and common scalar types to JSON-compatible values."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()

        if tensor.ndim == 0:
            return tensor.item()

        return tensor.float().tolist()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)

        if math.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return None


def get_task_text(frame: dict[str, Any]) -> str:
    task = frame.get("task", "")

    if isinstance(task, list):
        return str(task[0]) if task else ""

    return str(task).strip()


def gpu_memory_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gib": bytes_to_gib(torch.cuda.memory_allocated()),
        "reserved_gib": bytes_to_gib(torch.cuda.memory_reserved()),
        "peak_allocated_gib": bytes_to_gib(
            torch.cuda.max_memory_allocated()
        ),
    }


# ---------------------------------------------------------------------------
# Image conversion and saving
# ---------------------------------------------------------------------------

def image_tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert a CHW or BCHW image tensor into an RGB PIL image.

    LeRobot image tensors are normally floating point and scaled to [0, 1].
    """
    tensor = image_tensor.detach().cpu()

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Expected a one-image batch but received shape "
                f"{tuple(tensor.shape)}"
            )

        tensor = tensor[0]

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected CHW image tensor; received {tuple(tensor.shape)}"
        )

    if tensor.shape[0] not in (1, 3, 4):
        raise ValueError(
            "Expected 1, 3, or 4 channels; received "
            f"{tensor.shape[0]}"
        )

    if tensor.is_floating_point():
        tensor = tensor.float()

        minimum = float(tensor.min())
        maximum = float(tensor.max())

        # Handle [-1, 1] tensors if encountered.
        if minimum < 0.0 and maximum <= 1.0:
            tensor = (tensor + 1.0) / 2.0

        tensor = tensor.clamp(0.0, 1.0)
        tensor = (tensor * 255.0).round().to(torch.uint8)
    else:
        tensor = tensor.to(torch.uint8)

    tensor = tensor.permute(1, 2, 0).contiguous()
    array = tensor.numpy()

    if array.shape[2] == 1:
        array = array[:, :, 0]
        return Image.fromarray(array, mode="L").convert("RGB")

    if array.shape[2] == 4:
        return Image.fromarray(array, mode="RGBA").convert("RGB")

    return Image.fromarray(array, mode="RGB")


def save_frame_cameras(
    frame: dict[str, Any],
    frame_number: int,
    image_dir: Path,
) -> list[dict[str, str]]:
    """
    Save the real camera images contained in one raw dataset frame.

    Only the original LIBERO views are saved. The diagnostic duplicated
    camera3 is not saved as a separate image.
    """
    camera_fields = (
        ("observation.images.image", "Camera 1"),
        ("observation.images.image2", "Camera 2"),
    )

    saved: list[dict[str, str]] = []

    for key, label in camera_fields:
        value = frame.get(key)

        if not isinstance(value, torch.Tensor):
            continue

        filename = f"frame_{frame_number:04d}_{key.split('.')[-1]}.png"
        output_path = image_dir / filename

        image = image_tensor_to_pil(value)
        image.save(output_path)

        saved.append(
            {
                "key": key,
                "label": label,
                "filename": filename,
            }
        )

    return saved


# ---------------------------------------------------------------------------
# LeRobot and SmolVLA setup
# ---------------------------------------------------------------------------

def load_dataset_episode(
    dataset_id: str,
    episode_index: int,
) -> LeRobotDataset:
    """
    Load one episode where supported by the installed LeRobot release.
    """
    print(
        f"Loading dataset {dataset_id!r}, episode {episode_index}..."
    )

    try:
        return LeRobotDataset(
            dataset_id,
            episodes=[episode_index],
        )

    except TypeError as exc:
        print(
            "The installed LeRobotDataset constructor did not accept "
            "'episodes'. Falling back to the complete dataset."
        )
        print("Constructor message:", exc)

        return LeRobotDataset(dataset_id)


def make_processors_compatibly(
    policy: SmolVLAPolicy,
    dataset: LeRobotDataset,
    model_id: str,
    device: torch.device,
):
    """
    Create policy preprocessing and postprocessing pipelines.

    The first form supplies dataset statistics. The fallback supports
    LeRobot releases whose factory signature differs.
    """
    overrides = {
        "device_processor": {
            "device": str(device),
        }
    }

    try:
        return make_pre_post_processors(
            policy.config,
            model_id,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides=overrides,
        )

    except TypeError as exc:
        print(
            "Processor factory did not accept dataset_stats; "
            "using the saved pretrained processor configuration."
        )
        print("Processor message:", exc)

        return make_pre_post_processors(
            policy.config,
            model_id,
            preprocessor_overrides=overrides,
        )


def adapt_camera_keys(
    processed_frame: dict[str, Any],
) -> dict[str, str]:
    """
    Map LIBERO camera names to those expected by smolvla_base.

    LIBERO provides:
        observation.images.image
        observation.images.image2

    The pretrained policy expects:
        observation.images.camera1
        observation.images.camera2
        observation.images.camera3

    camera3 is duplicated from camera2 solely to let the diagnostic pipeline
    run. It is not an authentic third camera observation.
    """
    mapping_notes: dict[str, str] = {}

    image1_key = "observation.images.image"
    image2_key = "observation.images.image2"

    camera1_key = "observation.images.camera1"
    camera2_key = "observation.images.camera2"
    camera3_key = "observation.images.camera3"

    if image1_key in processed_frame:
        processed_frame[camera1_key] = processed_frame[image1_key]
        mapping_notes[camera1_key] = image1_key

    if image2_key in processed_frame:
        processed_frame[camera2_key] = processed_frame[image2_key]
        mapping_notes[camera2_key] = image2_key

    if camera3_key not in processed_frame:
        if camera2_key in processed_frame:
            processed_frame[camera3_key] = processed_frame[camera2_key]
            mapping_notes[camera3_key] = (
                camera2_key + " (diagnostic duplicate)"
            )
        elif camera1_key in processed_frame:
            processed_frame[camera3_key] = processed_frame[camera1_key]
            mapping_notes[camera3_key] = (
                camera1_key + " (diagnostic duplicate)"
            )

    expected = (camera1_key, camera2_key, camera3_key)
    missing = [key for key in expected if key not in processed_frame]

    if missing:
        raise KeyError(
            "Unable to create all camera inputs expected by SmolVLA:\n"
            + "\n".join(f"  {key}" for key in missing)
        )

    return mapping_notes


# ---------------------------------------------------------------------------
# Episode frame selection
# ---------------------------------------------------------------------------

def choose_evenly_spaced_indices(
    dataset_length: int,
    requested_count: int,
) -> list[int]:
    """
    Choose approximately evenly spaced local frame indices.

    The first and final frames are included when two or more frames are
    requested.
    """
    if dataset_length <= 0:
        raise ValueError("The loaded dataset contains no frames.")

    count = max(1, min(requested_count, dataset_length))

    if count == 1:
        return [0]

    values: list[int] = []

    for position in range(count):
        ratio = position / (count - 1)
        index = round(ratio * (dataset_length - 1))

        if index not in values:
            values.append(index)

    return values


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compare_actions(
    predicted: torch.Tensor,
    recorded: torch.Tensor | None,
) -> dict[str, Any]:
    """
    Compare the policy output with the comparable prefix of the recorded action.

    smolvla_base outputs six dimensions while LIBERO stores seven. The final
    LIBERO value is preserved separately instead of being silently discarded.
    """
    predicted_flat = predicted.detach().float().cpu().reshape(-1)

    result: dict[str, Any] = {
        "predicted": predicted_flat.tolist(),
        "predicted_dimensions": int(predicted_flat.numel()),
        "recorded": None,
        "recorded_dimensions": None,
        "recorded_extra_values": [],
        "comparable_dimensions": 0,
        "absolute_errors": [],
        "mean_absolute_error": None,
        "root_mean_squared_error": None,
        "maximum_absolute_error": None,
    }

    if not isinstance(recorded, torch.Tensor):
        return result

    recorded_flat = recorded.detach().float().cpu().reshape(-1)

    comparable_count = min(
        predicted_flat.numel(),
        recorded_flat.numel(),
    )

    predicted_comp = predicted_flat[:comparable_count]
    recorded_comp = recorded_flat[:comparable_count]

    differences = predicted_comp - recorded_comp
    absolute_errors = differences.abs()

    result.update(
        {
            "recorded": recorded_flat.tolist(),
            "recorded_dimensions": int(recorded_flat.numel()),
            "recorded_extra_values": (
                recorded_flat[comparable_count:].tolist()
            ),
            "comparable_dimensions": int(comparable_count),
            "absolute_errors": absolute_errors.tolist(),
            "mean_absolute_error": float(
                absolute_errors.mean().item()
            ),
            "root_mean_squared_error": float(
                torch.sqrt((differences**2).mean()).item()
            ),
            "maximum_absolute_error": float(
                absolute_errors.max().item()
            ),
        }
    )

    return result


# ---------------------------------------------------------------------------
# HTML action chart generation
# ---------------------------------------------------------------------------

def action_comparison_svg(
    predicted: list[float],
    recorded: list[float],
    width: int = 760,
) -> str:
    """
    Return an inline SVG comparing predicted and recorded action dimensions.

    Bars extend left or right from a central zero axis.
    """
    comparable_count = min(len(predicted), len(recorded))

    if comparable_count == 0:
        return "<p>No comparable action values.</p>"

    row_height = 56
    top_margin = 36
    bottom_margin = 24
    left_label_width = 72
    right_text_width = 180
    chart_width = width - left_label_width - right_text_width
    chart_left = left_label_width
    zero_x = chart_left + chart_width / 2
    half_chart = chart_width / 2 - 12

    all_values = (
        predicted[:comparable_count]
        + recorded[:comparable_count]
    )

    max_abs = max(abs(value) for value in all_values)
    max_abs = max(max_abs, 1e-6)

    height = top_margin + comparable_count * row_height + bottom_margin

    lines = [
        (
            f'<svg viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="Action comparison chart">'
        ),
        (
            f'<line x1="{zero_x:.1f}" y1="{top_margin - 18}" '
            f'x2="{zero_x:.1f}" y2="{height - bottom_margin}" '
            'stroke="#666" stroke-width="1"/>'
        ),
        (
            f'<text x="{zero_x:.1f}" y="18" text-anchor="middle" '
            'font-size="12">zero</text>'
        ),
    ]

    for index in range(comparable_count):
        predicted_value = float(predicted[index])
        recorded_value = float(recorded[index])

        row_y = top_margin + index * row_height
        predicted_y = row_y
        recorded_y = row_y + 22

        predicted_width = abs(predicted_value) / max_abs * half_chart
        recorded_width = abs(recorded_value) / max_abs * half_chart

        predicted_x = (
            zero_x
            if predicted_value >= 0
            else zero_x - predicted_width
        )
        recorded_x = (
            zero_x
            if recorded_value >= 0
            else zero_x - recorded_width
        )

        lines.extend(
            [
                (
                    f'<text x="8" y="{row_y + 18}" font-size="13">'
                    f'A{index + 1}</text>'
                ),
                (
                    f'<rect x="{predicted_x:.1f}" y="{predicted_y}" '
                    f'width="{predicted_width:.1f}" height="16" '
                    'fill="#4472c4"/>'
                ),
                (
                    f'<rect x="{recorded_x:.1f}" y="{recorded_y}" '
                    f'width="{recorded_width:.1f}" height="16" '
                    'fill="#ed7d31"/>'
                ),
                (
                    f'<text x="{width - right_text_width + 10}" '
                    f'y="{predicted_y + 13}" font-size="12">'
                    f'Predicted: {predicted_value:.5f}</text>'
                ),
                (
                    f'<text x="{width - right_text_width + 10}" '
                    f'y="{recorded_y + 13}" font-size="12">'
                    f'Recorded: {recorded_value:.5f}</text>'
                ),
            ]
        )

    lines.append("</svg>")
    return "\n".join(lines)


def metric_value(value: Any) -> str:
    number = safe_float(value)

    if number is None:
        return "Not available"

    return f"{number:.6f}"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html_report(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    cards: list[str] = []

    for result in results:
        comparison = result["comparison"]
        predicted = comparison["predicted"]
        recorded = comparison["recorded"] or []

        image_html = []

        for image_record in result["images"]:
            relative_path = (
                Path("images") / image_record["filename"]
            ).as_posix()

            image_html.append(
                f"""
                <figure>
                    <img
                        src="{html.escape(relative_path)}"
                        alt="{html.escape(image_record['label'])}"
                    >
                    <figcaption>
                        {html.escape(image_record['label'])}<br>
                        <code>{html.escape(image_record['key'])}</code>
                    </figcaption>
                </figure>
                """
            )

        extra_recorded = comparison["recorded_extra_values"]

        if extra_recorded:
            extra_html = (
                "<p><strong>Unmatched recorded action value(s):</strong> "
                + ", ".join(f"{value:.6f}" for value in extra_recorded)
                + "</p>"
            )
        else:
            extra_html = ""

        chart = action_comparison_svg(predicted, recorded)

        cards.append(
            f"""
            <section class="frame-card">
                <header>
                    <h2>
                        Selected frame {result['selection_number']}
                        — dataset index {result['dataset_index']}
                    </h2>
                    <p class="task">
                        <strong>Task:</strong>
                        {html.escape(result['task'])}
                    </p>
                </header>

                <div class="image-grid">
                    {''.join(image_html)}
                </div>

                <div class="camera-note">
                    The policy was supplied with the two real LIBERO camera
                    views. Its required third camera input was created by
                    duplicating camera2 for diagnostic compatibility.
                </div>

                <h3>Action comparison</h3>

                <div class="legend">
                    <span class="legend-predicted"></span> Predicted
                    <span class="legend-recorded"></span> Recorded
                </div>

                <div class="chart-container">
                    {chart}
                </div>

                {extra_html}

                <table class="metrics">
                    <tr>
                        <th>Inference time</th>
                        <td>{result['inference_seconds']:.4f} seconds</td>
                    </tr>
                    <tr>
                        <th>Comparable dimensions</th>
                        <td>{comparison['comparable_dimensions']}</td>
                    </tr>
                    <tr>
                        <th>Mean absolute error</th>
                        <td>
                            {metric_value(comparison['mean_absolute_error'])}
                        </td>
                    </tr>
                    <tr>
                        <th>Root mean squared error</th>
                        <td>
                            {
                                metric_value(
                                    comparison[
                                        'root_mean_squared_error'
                                    ]
                                )
                            }
                        </td>
                    </tr>
                    <tr>
                        <th>Maximum absolute error</th>
                        <td>
                            {
                                metric_value(
                                    comparison[
                                        'maximum_absolute_error'
                                    ]
                                )
                            }
                        </td>
                    </tr>
                </table>
            </section>
            """
        )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmolVLA Visual Episode Test</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 0;
        background: #f2f3f5;
        color: #222;
        line-height: 1.45;
    }}

    main {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 28px;
    }}

    .report-header,
    .frame-card {{
        background: white;
        border: 1px solid #d5d8dc;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.07);
    }}

    h1, h2, h3 {{
        margin-top: 0;
    }}

    .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 12px;
    }}

    .summary-item {{
        background: #f7f8fa;
        padding: 12px;
        border-radius: 6px;
    }}

    .task {{
        font-size: 1.05rem;
    }}

    .image-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 18px;
        margin: 18px 0;
    }}

    figure {{
        margin: 0;
    }}

    figure img {{
        width: 100%;
        height: auto;
        border: 1px solid #bbb;
        border-radius: 6px;
        image-rendering: auto;
    }}

    figcaption {{
        margin-top: 6px;
        text-align: center;
    }}

    .camera-note {{
        padding: 12px;
        background: #fff8dc;
        border-left: 4px solid #c9a227;
        margin: 14px 0 22px;
    }}

    .legend {{
        margin-bottom: 8px;
    }}

    .legend span {{
        display: inline-block;
        width: 16px;
        height: 12px;
        margin-left: 14px;
        margin-right: 4px;
    }}

    .legend span:first-child {{
        margin-left: 0;
    }}

    .legend-predicted {{
        background: #4472c4;
    }}

    .legend-recorded {{
        background: #ed7d31;
    }}

    .chart-container {{
        width: 100%;
        overflow-x: auto;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 8px;
        box-sizing: border-box;
    }}

    .chart-container svg {{
        width: 100%;
        min-width: 680px;
        height: auto;
    }}

    table.metrics {{
        border-collapse: collapse;
        width: 100%;
        margin-top: 18px;
    }}

    table.metrics th,
    table.metrics td {{
        padding: 9px 12px;
        border: 1px solid #d7d7d7;
        text-align: left;
    }}

    table.metrics th {{
        width: 40%;
        background: #f4f5f6;
    }}

    code {{
        overflow-wrap: anywhere;
    }}

    .warning {{
        background: #fff3cd;
        border: 1px solid #e6cf7b;
        padding: 14px;
        border-radius: 6px;
    }}
</style>
</head>
<body>
<main>
    <section class="report-header">
        <h1>SmolVLA Visual Offline Episode Test</h1>

        <div class="summary-grid">
            <div class="summary-item">
                <strong>Model</strong><br>
                {html.escape(summary['model_id'])}
            </div>

            <div class="summary-item">
                <strong>Dataset</strong><br>
                {html.escape(summary['dataset_id'])}
            </div>

            <div class="summary-item">
                <strong>Episode</strong><br>
                {summary['episode_index']}
            </div>

            <div class="summary-item">
                <strong>Frames tested</strong><br>
                {summary['frames_tested']}
            </div>

            <div class="summary-item">
                <strong>Mean inference time</strong><br>
                {summary['mean_inference_seconds']:.4f} seconds
            </div>

            <div class="summary-item">
                <strong>Mean frame MAE</strong><br>
                {metric_value(summary['mean_frame_mae'])}
            </div>

            <div class="summary-item">
                <strong>Peak CUDA allocation</strong><br>
                {summary['gpu_memory']['peak_allocated_gib']:.3f} GiB
            </div>

            <div class="summary-item">
                <strong>GPU</strong><br>
                {html.escape(summary['gpu'])}
            </div>
        </div>

        <p class="warning">
            This report is a pipeline and representation diagnostic.
            The pretrained policy expects three cameras and six action
            dimensions, while the LIBERO data supplies two cameras and seven
            action dimensions. The third policy camera is a duplicated input,
            and the seventh recorded action is not compared with the
            six-dimensional prediction.
        </p>
    </section>

    {''.join(cards)}
</main>
</body>
</html>
"""

    output_path.write_text(document, encoding="utf-8")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run visual offline SmolVLA inference over several "
            "recorded LIBERO frames."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Model identifier. Default: {DEFAULT_MODEL_ID}",
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_ID,
        help=f"Dataset identifier. Default: {DEFAULT_DATASET_ID}",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=DEFAULT_EPISODE_INDEX,
        help=f"Episode index. Default: {DEFAULT_EPISODE_INDEX}",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=DEFAULT_FRAME_COUNT,
        help=(
            "Number of evenly spaced frames to test. "
            f"Default: {DEFAULT_FRAME_COUNT}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable.", file=sys.stderr)
        return 1

    if args.frames < 1:
        print("ERROR: --frames must be at least 1.", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    report_path = output_dir / "smolvla_visual_report.html"
    json_path = output_dir / "smolvla_visual_results.json"

    image_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    print("=" * 78)
    print("SmolVLA visual offline episode test")
    print("=" * 78)
    print("Model:", args.model)
    print("Dataset:", args.dataset)
    print("Episode:", args.episode)
    print("Requested frame count:", args.frames)
    print("Output directory:", output_dir)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("FFmpeg shared DLLs:", DEFAULT_FFMPEG_SHARED_BIN)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # -----------------------------------------------------------------------
    # Load policy
    # -----------------------------------------------------------------------

    print("\nLoading SmolVLA policy...")
    model_start = time.perf_counter()

    policy = SmolVLAPolicy.from_pretrained(args.model)
    policy = policy.to(device)
    policy.eval()

    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_start

    print(f"Model loaded in {model_load_seconds:.2f} seconds.")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------

    dataset_start = time.perf_counter()

    dataset = load_dataset_episode(
        dataset_id=args.dataset,
        episode_index=args.episode,
    )

    dataset_load_seconds = time.perf_counter() - dataset_start

    print(
        f"Dataset initialized in {dataset_load_seconds:.2f} seconds."
    )
    print("Loaded frame count:", len(dataset))
    print("Dataset FPS:", getattr(dataset.meta, "fps", "unknown"))

    selected_indices = choose_evenly_spaced_indices(
        dataset_length=len(dataset),
        requested_count=args.frames,
    )

    print("Selected local frame indices:", selected_indices)

    preprocess, postprocess = make_processors_compatibly(
        policy=policy,
        dataset=dataset,
        model_id=args.model,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Process selected frames
    # -----------------------------------------------------------------------

    results: list[dict[str, Any]] = []

    for selection_number, dataset_index in enumerate(
        selected_indices,
        start=1,
    ):
        print("\n" + "=" * 78)
        print(
            f"Frame {selection_number}/{len(selected_indices)} "
            f"— dataset index {dataset_index}"
        )
        print("=" * 78)

        raw_start = time.perf_counter()
        frame = dict(dataset[dataset_index])
        frame_load_seconds = time.perf_counter() - raw_start

        task = get_task_text(frame)
        print("Task:", task)
        print(f"Frame decoded in {frame_load_seconds:.4f} seconds.")

        saved_images = save_frame_cameras(
            frame=frame,
            frame_number=dataset_index,
            image_dir=image_dir,
        )

        processed_frame = preprocess(frame)
        mapping_notes = adapt_camera_keys(processed_frame)

        print("Camera mapping:")

        for destination, source in mapping_notes.items():
            print(f"  {destination} <- {source}")

        # This is essential: otherwise select_action may return actions left
        # in the policy's previous 50-step action queue.
        if hasattr(policy, "reset"):
            policy.reset()

        torch.cuda.synchronize()
        inference_start = time.perf_counter()

        with torch.inference_mode():
            predicted_normalized = policy.select_action(
                processed_frame
            )
            predicted_action = postprocess(predicted_normalized)

        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start

        predicted_cpu = (
            predicted_action.detach().float().cpu()
        )

        recorded_action = frame.get("action")

        comparison = compare_actions(
            predicted=predicted_cpu,
            recorded=(
                recorded_action
                if isinstance(recorded_action, torch.Tensor)
                else None
            ),
        )

        print(
            f"Inference completed in {inference_seconds:.4f} seconds."
        )
        print("Predicted action:", comparison["predicted"])
        print("Recorded action:", comparison["recorded"])
        print(
            "Mean absolute error:",
            comparison["mean_absolute_error"],
        )

        results.append(
            {
                "selection_number": selection_number,
                "dataset_index": dataset_index,
                "task": task,
                "frame_load_seconds": frame_load_seconds,
                "inference_seconds": inference_seconds,
                "images": saved_images,
                "camera_mapping": mapping_notes,
                "comparison": comparison,
                "frame_index": tensor_to_python(
                    frame.get("frame_index")
                ),
                "episode_index": tensor_to_python(
                    frame.get("episode_index")
                ),
                "timestamp": tensor_to_python(
                    frame.get("timestamp")
                ),
            }
        )

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------

    inference_times = [
        result["inference_seconds"]
        for result in results
    ]

    frame_maes = [
        result["comparison"]["mean_absolute_error"]
        for result in results
        if result["comparison"]["mean_absolute_error"] is not None
    ]

    mean_inference_seconds = (
        sum(inference_times) / len(inference_times)
        if inference_times
        else 0.0
    )

    mean_frame_mae = (
        sum(frame_maes) / len(frame_maes)
        if frame_maes
        else None
    )

    gpu_memory = gpu_memory_snapshot()

    summary = {
        "model_id": args.model,
        "dataset_id": args.dataset,
        "episode_index": args.episode,
        "frames_tested": len(results),
        "selected_indices": selected_indices,
        "model_load_seconds": model_load_seconds,
        "dataset_load_seconds": dataset_load_seconds,
        "mean_inference_seconds": mean_inference_seconds,
        "mean_frame_mae": mean_frame_mae,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory": gpu_memory,
        "ffmpeg_shared_bin": str(DEFAULT_FFMPEG_SHARED_BIN),
        "output_directory": str(output_dir),
    }

    output_payload = {
        "summary": summary,
        "frames": results,
        "important_limitations": [
            (
                "The LIBERO sample has two authentic camera views, while "
                "smolvla_base expects three. camera3 duplicates camera2."
            ),
            (
                "The policy outputs six action values while LIBERO stores "
                "seven. Only the first six are compared."
            ),
            (
                "Single-step numerical agreement is not equivalent to "
                "closed-loop robot-task success."
            ),
            (
                "No action generated by this program is sent to a robot or "
                "simulator."
            ),
        ],
    }

    json_path.write_text(
        json.dumps(
            output_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    build_html_report(
        results=results,
        summary=summary,
        output_path=report_path,
    )

    print("\n" + "=" * 78)
    print("Visual test completed")
    print("=" * 78)
    print("HTML report:")
    print(report_path)
    print("\nJSON results:")
    print(json_path)
    print("\nSaved camera images:")
    print(image_dir)
    print(
        f"\nMean inference time: "
        f"{mean_inference_seconds:.4f} seconds"
    )

    if mean_frame_mae is not None:
        print(f"Mean frame MAE: {mean_frame_mae:.6f}")

    print(
        f"Peak CUDA allocation: "
        f"{gpu_memory.get('peak_allocated_gib', 0.0):.3f} GiB"
    )

    print("\nNo physical robot was connected or controlled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())