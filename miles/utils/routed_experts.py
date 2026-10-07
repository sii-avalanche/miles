from typing import Any

import numpy as np
import pybase64


def is_boxed_ray_ref(value: Any) -> bool:
    if not hasattr(value, "inner"):
        return False
    try:
        import ray
    except ImportError:
        return False
    return isinstance(value.inner, ray.ObjectRef)


def decode_routed_experts(
    routed_experts: str,
    num_tokens: int,
    num_layers: int,
    moe_router_topk: int,
) -> np.ndarray:
    x = np.frombuffer(pybase64.b64decode(routed_experts.encode("ascii")), dtype=np.int32)
    row = num_layers * moe_router_topk
    if x.size == (num_tokens + 1) * row:
        # sglang also forwarded the final token, whose routing feeds no training position
        x = x[: num_tokens * row]
    assert x.size == 0 or x.any(), (
        "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
        "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
    )
    return x.reshape(num_tokens, num_layers, moe_router_topk)


def resolve_routed_experts(
    routed_experts: Any,
    num_tokens: int,
    num_layers: int,
    moe_router_topk: int,
):
    if is_boxed_ray_ref(routed_experts):
        import ray

        routed_experts = ray.get(routed_experts.inner)
    if isinstance(routed_experts, str):
        return decode_routed_experts(routed_experts, num_tokens, num_layers, moe_router_topk)
    return routed_experts
