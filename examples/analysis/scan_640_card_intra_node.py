"""Reproduce the 640-card decode intra-node bandwidth sensitivity scan.

The fixed GEMM and VECTOR terms come from results/decode_640_cards.csv. This
script recalculates communication with independent TP-reduce, EP-dispatch, and
EP-combine widths. It prints results only and does not write report files.
"""

from dataclasses import dataclass
from math import ceil
from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from infersim.cost.collective import (
    all_reduce_cost,
    all_to_all_cost,
    payload_bytes,
)
from infersim.schema.hardware import HardwareSpec


NUM_LAYERS = 80
HIDDEN_SIZE = 8192
TOTAL_CARDS = 640
CARDS_PER_NODE = 8
REPLICAS = 16
BATCH_PER_REPLICA = 64
TOTAL_BATCH = REPLICAS * BATCH_PER_REPLICA

ATTENTION_TP = 4
ATTENTION_DP = 10
MOE_TP = 5
EXPERT_PARALLEL = 8
EXPERTS_PER_TOKEN = 4

TP_REDUCE_BITS = 32
EP_DISPATCH_BITS = 4
EP_COMBINE_BITS = 16

GEMM_MS = 20.801879
VECTOR_MS = 1.949727

LOCAL_ATTENTION_REQUESTS = ceil(BATCH_PER_REPLICA / ATTENTION_DP)
LOCAL_ROUTED_ASSIGNMENTS = ceil(
    BATCH_PER_REPLICA * EXPERTS_PER_TOKEN / EXPERT_PARALLEL
)

ATTENTION_TP_PAYLOAD_BYTES = payload_bytes(
    LOCAL_ATTENTION_REQUESTS * HIDDEN_SIZE,
    TP_REDUCE_BITS,
)
ROUTED_TP_PAYLOAD_BYTES = payload_bytes(
    LOCAL_ROUTED_ASSIGNMENTS * HIDDEN_SIZE,
    TP_REDUCE_BITS,
)
EP_DISPATCH_PAYLOAD_BYTES = payload_bytes(
    LOCAL_ATTENTION_REQUESTS * EXPERTS_PER_TOKEN * HIDDEN_SIZE,
    EP_DISPATCH_BITS,
)
EP_COMBINE_PAYLOAD_BYTES = payload_bytes(
    LOCAL_ATTENTION_REQUESTS * EXPERTS_PER_TOKEN * HIDDEN_SIZE,
    EP_COMBINE_BITS,
)


@dataclass(frozen=True)
class ScanRow:
    intra_node_gbps: int
    tp_ms: float
    ep_ms: float
    total_ms: float
    user_tokens_per_s: float
    system_tokens_per_s: float
    paths: tuple[str, str, str, str]


def _hardware(intra_node_gbps: int) -> HardwareSpec:
    # Geometry fields are schema-required but do not enter collective costs.
    return HardwareSpec.from_dict(
        {
            "name": "640-card review accelerator",
            "memory_capacity_gb": 200,
            "memory_bandwidth_gbps": 2000,
            "cards_per_node": CARDS_PER_NODE,
            "compute_tflops": {
                "gemm": {"w4a4": 1024, "w4a8": 1024},
                "vector": {
                    "fp4": 32,
                    "int8": 32,
                    "bf16": 32,
                    "fp32": 32,
                },
            },
            "gemm_tile": {"m": 128, "n": 128, "k": 64},
            "gemm_engines": 1,
            "vector_width": 1,
            "vector_units": 1,
            "kernel_launch_latency_us": {
                "gemm": 5,
                "vector": 3,
                "collective": 8,
            },
            "interconnect": {
                "intra_node_gbps": intra_node_gbps,
                "intra_node_latency_us": 1,
                "inter_node_gbps": 800,
                "inter_node_latency_us": 5,
            },
        }
    )


def calculate_row(intra_node_gbps: int) -> ScanRow:
    hardware = _hardware(intra_node_gbps)
    attention = all_reduce_cost(
        ATTENTION_TP_PAYLOAD_BYTES,
        ATTENTION_TP,
        hardware,
    )
    routed = all_reduce_cost(
        ROUTED_TP_PAYLOAD_BYTES,
        MOE_TP,
        hardware,
    )
    dispatch = all_to_all_cost(
        EP_DISPATCH_PAYLOAD_BYTES,
        EXPERT_PARALLEL,
        hardware,
    )
    combine = all_to_all_cost(
        EP_COMBINE_PAYLOAD_BYTES,
        EXPERT_PARALLEL,
        hardware,
    )

    # The synthetic target has no shared expert, so there is no third TP
    # all-reduce. Each listed collective is launched once per layer.
    tp_ms = NUM_LAYERS * (attention.seconds + routed.seconds) * 1000
    ep_ms = NUM_LAYERS * (dispatch.seconds + combine.seconds) * 1000
    total_ms = GEMM_MS + VECTOR_MS + tp_ms + ep_ms
    user_tokens_per_s = 1000 / total_ms
    system_tokens_per_s = TOTAL_BATCH * user_tokens_per_s
    return ScanRow(
        intra_node_gbps=intra_node_gbps,
        tp_ms=tp_ms,
        ep_ms=ep_ms,
        total_ms=total_ms,
        user_tokens_per_s=user_tokens_per_s,
        system_tokens_per_s=system_tokens_per_s,
        paths=(
            attention.path,
            routed.path,
            dispatch.path,
            combine.path,
        ),
    )


def scan_rows() -> tuple[ScanRow, ...]:
    return tuple(calculate_row(bandwidth) for bandwidth in range(100, 801, 100))


def main() -> None:
    print(
        "total_batch="
        f"{TOTAL_BATCH}, batch_per_replica={BATCH_PER_REPLICA}, "
        f"local_attention_requests={LOCAL_ATTENTION_REQUESTS}, "
        f"local_routed_assignments={LOCAL_ROUTED_ASSIGNMENTS}"
    )
    print(
        "payload_bytes: "
        f"attention_tp={ATTENTION_TP_PAYLOAD_BYTES:.0f}, "
        f"routed_tp={ROUTED_TP_PAYLOAD_BYTES:.0f}, "
        f"ep_dispatch={EP_DISPATCH_PAYLOAD_BYTES:.0f}, "
        f"ep_combine={EP_COMBINE_PAYLOAD_BYTES:.0f}"
    )
    print(
        "intra_GBps,tp_ms,ep_ms,total_ms,user_tokens_per_s,"
        "system_tokens_per_s,attention_tp_path,routed_tp_path,"
        "ep_dispatch_path,ep_combine_path"
    )
    for row in scan_rows():
        print(
            f"{row.intra_node_gbps},{row.tp_ms:.6f},{row.ep_ms:.6f},"
            f"{row.total_ms:.6f},{row.user_tokens_per_s:.6f},"
            f"{row.system_tokens_per_s:.3f},{','.join(row.paths)}"
        )


if __name__ == "__main__":
    main()
