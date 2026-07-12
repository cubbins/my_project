import time
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


MODEL_ID = "lerobot/smolvla_base"


def gib(value: int) -> float:
    return value / (1024 ** 3)


def print_gpu_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return

    print(f"\n{label}")
    print("Allocated:", f"{gib(torch.cuda.memory_allocated()):.3f} GiB")
    print("Reserved: ", f"{gib(torch.cuda.memory_reserved()):.3f} GiB")
    print("Peak:     ", f"{gib(torch.cuda.max_memory_allocated()):.3f} GiB")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")

    print("=" * 70)
    print("SmolVLA model-loading test")
    print("=" * 70)
    print("Model:", MODEL_ID)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_gpu_memory("Before loading")

    start = time.perf_counter()

    policy = SmolVLAPolicy.from_pretrained(MODEL_ID)
    policy = policy.to(device)
    policy.eval()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"\nModel loaded successfully in {elapsed:.2f} seconds.")
    print("Policy class:", type(policy).__name__)
    print("Policy device:", next(policy.parameters()).device)
    print("Policy dtype:", next(policy.parameters()).dtype)

    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    trainable_count = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )

    print("Parameters:", f"{parameter_count:,}")
    print("Trainable parameters:", f"{trainable_count:,}")

    print_gpu_memory("After loading")

    print("\nConfiguration:")
    print(policy.config)

    print("\nNo robot commands were issued.")


if __name__ == "__main__":
    main()