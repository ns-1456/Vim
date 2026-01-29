from __future__ import annotations

from typing import Iterable, List

import torch


@torch.no_grad()
def benchmark_memory(
    model: torch.nn.Module,
    resolutions: Iterable[int] = (224, 512, 1024, 2048),
    device: str = "cuda",
) -> List[dict]:
    """
    Measure peak GPU memory at increasing resolutions.

    Protocol:
      1. Create dummy input at resolution
      2. torch.cuda.reset_peak_memory_stats()
      3. Forward + backward pass
      4. Record torch.cuda.max_memory_allocated()
    """
    assert torch.cuda.is_available(), "CUDA is required for memory benchmarking."

    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()

    results = []
    for res in resolutions:
        h = w = res
        x = torch.randn(1, 3, h, w, device=device_t, requires_grad=True)
        torch.cuda.reset_peak_memory_stats(device_t)

        # Forward + backward on dummy loss.
        out = model(x)
        loss = out.mean()
        loss.backward()

        peak_mem = torch.cuda.max_memory_allocated(device_t) / (1024**3)
        results.append({"resolution": res, "peak_memory_gb": peak_mem})
        print(f"Resolution {res}x{res}: peak memory {peak_mem:.2f} GB")

    return results


if __name__ == "__main__":
    from vim import vim_tiny_cifar100

    model = vim_tiny_cifar100(img_size=224)
    benchmark_memory(model)

