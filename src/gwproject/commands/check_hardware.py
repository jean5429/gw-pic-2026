from __future__ import annotations

import json
import os
import platform

from gwproject.provenance import accelerator_manifest, package_versions


def main() -> int:
    result = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "packages": package_versions(("torch", "torchvision", "numpy", "scipy", "gwpy")),
        "accelerator": accelerator_manifest(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["accelerator"].get("torch_cuda_available") else 2


if __name__ == "__main__":
    raise SystemExit(main())
