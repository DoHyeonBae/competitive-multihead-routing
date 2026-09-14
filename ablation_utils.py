"""
diagnose_ablation_uniform_and_head_removal.py와 analyze_correlation_and_joint_removal.py가
공유하는 핵심 함수(모델 forward를 stage별로 다르게 제어하는 부분). 두 스크립트에
똑같은 로직을 복붙해서 나중에 한쪽만 고치고 다른 쪽을 안 고치는 사고를 막기 위해
따로 뺐다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attention_unet_multitask import (
    AttentionUNetResNet34MultiTask,
    positive_segmentation_metrics_multitask,
)

STAGE_ORDER = ["dec4", "dec3", "dec2", "dec1"]  # 실제 forward에서 호출되는 순서(coarse->fine)
STAGE_LABELS = {
    "dec4": "dec4(14x14)", "dec3": "dec3(28x28)", "dec2": "dec2(56x56)", "dec1": "dec1(112x112)",
}


def detect_gate_type(ckpt: dict) -> str:
    state_keys = ckpt["model_state_dict"].keys()
    if any("psi_heads" in k for k in state_keys):
        return "multi_split"
    elif any("psi_router" in k for k in state_keys):
        return "competitive_moe"
    return ckpt.get("gate_type", "competitive_moe")


def run_forward(
    model: AttentionUNetResNet34MultiTask,
    images: torch.Tensor,
    active_mask_map: dict[str, torch.Tensor] | None = None,
    uniform_stage_names: set[str] | None = None,
) -> torch.Tensor:
    """model.forward()를 그대로 쓰지 않고 stage별로 다른 active_mask/uniform 설정을
    줄 수 있도록 직접 재구현. 로직은 AttentionUNetResNet34MultiTask.forward()와
    완전히 동일하고, active_mask/force_uniform_router만 stage마다 다르게 건다.

    active_mask_map의 각 값은 (num_heads,) 0/1 텐서 -- 여러 head를 동시에 0으로
    줘도 됨(joint removal ablation용, 이 함수 입장에서는 head가 몇 개 꺼지는지
    신경 안 씀)."""
    active_mask_map = active_mask_map or {}
    uniform_stage_names = uniform_stage_names or set()
    stages = {"dec4": model.dec4, "dec3": model.dec3, "dec2": model.dec2, "dec1": model.dec1}

    for name, stage in stages.items():
        stage.gate.force_uniform_router = name in uniform_stage_names

    input_size = images.shape[-2:]
    x0, x1, x2, x3, x4 = model.encoder(images)
    d4, _ = model.dec4(x4, x3, active_mask=active_mask_map.get("dec4"))
    d3, _ = model.dec3(d4, x2, active_mask=active_mask_map.get("dec3"))
    d2, _ = model.dec2(d3, x1, active_mask=active_mask_map.get("dec2"))
    d1, _ = model.dec1(d2, x0, active_mask=active_mask_map.get("dec1"))
    logits = model.head(d1)
    logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    for name, stage in stages.items():
        stage.gate.force_uniform_router = False  # 다음 config에 영향 안 주도록 항상 리셋

    return logits


@torch.no_grad()
def evaluate_config(
    model: AttentionUNetResNet34MultiTask,
    loader: DataLoader,
    task_names: list[str],
    device: torch.device,
    active_mask_map: dict[str, torch.Tensor] | None = None,
    uniform_stage_names: set[str] | None = None,
) -> dict[str, float]:
    """설정 하나(baseline / uniform / head-removal)에 대해 전체 val을 한 번 돌며
    태스크별 Dice를 n_pos로 가중 평균해서 반환. {organ_name: mean_dice}."""
    weighted_sum = {name: 0.0 for name in task_names}
    weight_count = {name: 0 for name in task_names}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = run_forward(model, images, active_mask_map, uniform_stage_names)
        metrics = positive_segmentation_metrics_multitask(logits, masks, task_names)
        for name, (dice, _iou, n_pos) in metrics.items():
            if dice is None or n_pos == 0:
                continue
            weighted_sum[name] += dice * n_pos
            weight_count[name] += n_pos

    result = {}
    for name in task_names:
        if weight_count[name] == 0:
            result[name] = float("nan")
        else:
            result[name] = weighted_sum[name] / weight_count[name]
    return result


def mean_of(d: dict[str, float]) -> float:
    vals = [v for v in d.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")