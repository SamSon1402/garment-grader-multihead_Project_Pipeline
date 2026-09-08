"""Optional Modal GPU benchmark example.

This is intentionally outside the real-time conveyor path. It demonstrates where
cloud GPU experiments can live while deployment decisions remain based on target
edge hardware benchmarks.

Run after `pip install -e .[modal]` and `modal setup`:
    modal run -m garment_grader.integrations.modal_benchmark
"""

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

if modal is not None:  # pragma: no cover - requires Modal account
    app = modal.App("garment-grader-benchmark")
    image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

    @app.function(image=image, gpu="L4", timeout=300)
    def smoke_benchmark(iterations: int = 100) -> dict[str, float]:
        import time
        import numpy as np

        x = np.random.rand(1024, 1024).astype("float32")
        start = time.perf_counter()
        for _ in range(iterations):
            _ = x @ x.T
        elapsed = time.perf_counter() - start
        return {"iterations": float(iterations), "elapsed_s": elapsed, "avg_ms": elapsed * 1000 / iterations}

    @app.local_entrypoint()
    def main(iterations: int = 100) -> None:
        print(smoke_benchmark.remote(iterations))
