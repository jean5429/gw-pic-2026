import time

import pytest


@pytest.mark.benchmark
def test_cuda_balanced_or_cpu_profile_forward_cost():
    torch = pytest.importorskip("torch")
    from gwproject.gravity_spy import build_resnet18_grayscale

    torch.set_num_threads(4)
    model = build_resnet18_grayscale(22, pretrained=False, conv1_stride=2, use_maxpool=False).eval()
    batch = torch.randn(8, 1, 140, 170)
    with torch.no_grad():
        model(batch)
    start = time.perf_counter()
    with torch.no_grad():
        model(batch)
    milliseconds_per_image = (time.perf_counter() - start) / len(batch) * 1000
    assert milliseconds_per_image < 500
