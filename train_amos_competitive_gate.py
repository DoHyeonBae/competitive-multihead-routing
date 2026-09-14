"""
AMOS22 Competitive Head Router 실험 -- 지금까지 안 해본 첫 "구조" 변경.

이전 진단(diagnose_head_utilization.py, vanilla H4 seed42 체크포인트)에서:
    dec1(최고해상도): utilization CV=0.29(불균형), cosine 0.92~0.98(그나마 갈라짐)
    dec2:            CV=0.11, cosine 0.96~0.99
    dec3:            CV=0.06, cosine 0.99+
    dec4(최심부):     CV=0.00, cosine=1.0000 (전 head alpha=1로 완전 saturation)
가 나왔다. 즉 지금 문제는 "head들이 다르게 보라고 안 시켜서"가 아니라, 독립 sigmoid라서
head끼리 경쟁 자체가 없다는 것 -- alpha_h = sigmoid(psi_h(...))는 각 head가 "이 위치가
중요한가"와 "내가 담당할까"를 동시에 독립적으로 결정하고, 전부 1(dec4)이 돼도 loss
입장에서 전혀 나쁠 게 없다.

CompetitiveMultiHeadAttentionGate(attention_unet_multitask.py에 추가)는 이 두 역할을
분리한다:
    s(p)   = sigmoid(...)          "여기가 중요한가" (head 공유)
    r_h(p) = softmax_h(z_h/tau)    "중요하면 누가 담당하는가" (head끼리 직접 경쟁, sum=1)
    alpha_h(p) = s(p) * num_heads * r_h(p)

r_h가 uniform(1/H)이고 s=1이면 alpha_h=1(all h) -- 기존 dec4의 saturated 해를 그대로
재현 가능하다. 즉 이 구조가 어디에도 분업을 강제하지 않는다 -- 그냥 "가능해지게"만
한다. 실제로 갈라지는지는 population-level load-balancing loss(아래 head_balance_loss)
하나만 걸고 관찰한다. entropy term은 첫 실험에서는 일부러 안 넣는다(한 번에 하나씩).

L = L_seg + lambda_bal * L_balance
L_balance = sum_h (mean_p[r_h(p)] - 1/num_heads)^2   (stage별로 계산 후 평균)

이 실험 이후에 다시 diagnose_head_utilization.py 같은 방식으로 새 체크포인트의
stage별 utilization/cosine을 재보면(--gate-type 인자만 다르게) "예측대로 dec1은
갈라지고 dec3/dec4는 여전히 saturated로 남는지"를 검증할 수 있다.

[2026-09-13 수정] --balance-target {all,fg} 추가: load-balancing loss의 population
평균을 전체 픽셀(all, 기존 동작)이 아니라 foreground(장기 존재 픽셀, fg)에서만 계산할
수 있게 함. 논문의 G_h(global utilization) vs F_h(foreground utilization) 구분을
그대로 학습 목적함수에 반영한 버전 -- "population-level로는 balanced해도 foreground
에서는 여전히 불균형할 수 있다"는 관찰을, 아예 처음부터 foreground 기준으로 balance를
강제하면 어떻게 되는지 보는 대조실험용.

Run (Colab, 5 epoch 스크리닝 먼저):
    !python /content/drive/MyDrive/train_amos_competitive_gate.py \
        --csv-dir /content/data/amos_prepared \
        --output-dir /content/drive/MyDrive/amos22/outputs_competitive \
        --task-names liver,spleen,kidney,stomach,gall_bladder,pancreas,left_adrenal_gland,right_adrenal_gland \
        --num-heads 4 --epochs 5 --seed 42 --lambda-bal 0.1
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_unet_multitask import (
    AttentionUNetResNet34MultiTask,
    MultiTaskEpochResult,
    combined_loss_multitask,
    count_parameters,
    estimate_pixel_pos_weights,
    hard_organ_routing_loss,
    head_diversity_penalty,
    positive_segmentation_metrics_multitask,
    seed_everything,
    set_pos_weights,
)
from train_amos_fixed_heads import load_split, make_dataset


def head_balance_loss(alphas: list[torch.Tensor]) -> torch.Tensor:
    """alphas: [a1,a2,a3,a4], 각 (B,num_heads,H,W), alpha_h(p)=s(p)*num_heads*r_h(p) 형태
    (CompetitiveMultiHeadAttentionGate 전용 -- 다른 gate_type에도 호출은 되지만 head
    경쟁이 없는 게이트라 이 정규화 자체가 별 의미는 없음, 참고용으로만 봐도 무방).

    head 축으로 정규화하면 r_h(p) = alpha_h(p) / sum_h(alpha_h(p))를 정확히 복원할 수
    있다(s(p)가 나눗셈에서 소거됨). population(배치 전체 픽셀) 평균 r_h가 1/num_heads
    에서 벗어난 만큼 벌점을 준다 -- "특정 위치에서 쏠리는 건 허용, 데이터 전체에서
    특정 head 하나가 독식하는 것만 방지"가 목적."""
    losses = []
    for alpha in alphas:
        num_heads = alpha.shape[1]
        if num_heads < 2:
            continue
        r = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        r_bar = r.mean(dim=(0, 2, 3))
        target = 1.0 / num_heads
        losses.append(((r_bar - target) ** 2).sum())
    if not losses:
        return torch.zeros((), device=alphas[0].device if alphas else "cpu")
    return torch.stack(losses).mean()


def head_balance_loss_fg(alphas: list[torch.Tensor], fg_mask_full: torch.Tensor) -> torch.Tensor:
    """head_balance_loss의 foreground-aware 버전. population 평균을 전체 픽셀이 아니라
    foreground(장기 존재) 픽셀에서만 계산한다.

    fg_mask_full: (B,1,H0,W0), 원본(입력) 해상도의 이진 foreground 마스크(장기 합집합,
    배경=0). alpha는 dec1..dec4마다 해상도가 다르므로 각 stage 해상도로 다운샘플해서
    맞춘다 -- 이때 nearest/bilinear가 아니라 max-pool을 쓰는데, adrenal gland처럼 작은
    장기가 nearest/bilinear 다운샘플 과정에서 통째로 씻겨나가는 걸 막기 위함(해당 블록
    안에 foreground 픽셀이 하나라도 있으면 그 블록 전체를 foreground로 취급)."""
    losses = []
    for alpha in alphas:
        num_heads = alpha.shape[1]
        if num_heads < 2:
            continue
        r = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        fg_mask = torch.nn.functional.adaptive_max_pool2d(fg_mask_full, output_size=alpha.shape[-2:])
        denom = fg_mask.sum() + 1e-8
        r_bar = (r * fg_mask).sum(dim=(0, 2, 3)) / denom
        target = 1.0 / num_heads
        losses.append(((r_bar - target) ** 2).sum())
    if not losses:
        return torch.zeros((), device=alphas[0].device if alphas else "cpu")
    return torch.stack(losses).mean()


def head_entropy_loss(alphas: list[torch.Tensor], lambda_ent_per_stage: list[float]) -> torch.Tensor:
    """r_h(p)=alpha_h(p)/sum_h(alpha_h(p))의 정규화 entropy(0~1, 미분 가능)를 계산해서
    stage별로 다른 lambda를 곱한 뒤, "항상 stage 개수(len(alphas))로 나눈다" -- 이게
    핵심이다. lambda가 전부 같은 값(예: 0.05)이면 이전 균일-lambda 실험의
    0.05*mean(E_1..E_4)와 정확히 같은 스케일이 나오고, 특정 stage만 0으로 끄면
    "이전 공식에서 그 stage의 기여분만 뺀" 값이 된다(즉 나머지 lambda/스케일은 그대로
    유지). 이전에 활성 stage 수로만 나누는 버전은(lam==0인 stage를 애초에 합에서
    스킵하고 남은 것만 더함) dec4를 끄면 나머지 3개 stage에 실질적으로 4/3배 센
    압력이 걸리는 스케일 버그였음 -- 실제로 dec1~3 entropy가 이전 균일-lambda 실험보다
    훨씬 빨리 떨어진 원인이 이 버그였을 가능성이 큼(사용자 지적으로 발견/수정).

    lambda=0인 stage는 entropy 압력 자체가 0이 되어 그 stage는 balance_loss/seg_loss만
    받는다(예: dec4의 H2가 화면 위치 shortcut으로 가는 게 확인돼서, dec4만 압력을 빼고
    dec1~3는 "이전과 동일한 스케일로" 유지하고 싶을 때 사용)."""
    assert len(lambda_ent_per_stage) == len(alphas), "lambda_ent_per_stage 길이가 decoder stage 수와 달라야 함"
    weighted = []
    for alpha, lam in zip(alphas, lambda_ent_per_stage):
        num_heads = alpha.shape[1]
        if num_heads < 2 or lam == 0.0:
            weighted.append(torch.zeros((), device=alpha.device))
            continue
        r = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        ent = -(r * torch.log(r.clamp_min(1e-8))).sum(dim=1) / math.log(num_heads)  # (B,H,W)
        weighted.append(lam * ent.mean())
    return torch.stack(weighted).sum() / len(alphas)  # 항상 전체 stage 수로 나눔(핵심 수정)


def router_diagnostics(
    alphas: list[torch.Tensor],
) -> tuple[float, list[float], list[float], list[list[float]]]:
    """alpha_h(p) = s(p)*num_heads*r_h(p)에서 r_h를 복원(head 축 정규화)해서:
      - normalized router entropy(0~1, 1=완전 uniform/무분업[0.25,0.25,0.25,0.25],
        0=완전 sharp/one-hot 배정) -- 4개 decoder stage 평균
      - population 평균 utilization r_bar_h (4개 stage 평균, head별) -- *stage 평균만
        보면 dec1에서 H1이 독식하고 dec2에서 H4가 독식하는 식으로 서로 다른 stage의
        불균형이 평균에서 상쇄돼 안 보일 수 있음(사용자 지적)*
      - stage별 entropy 리스트(dec1..dec4 따로)
      - stage별 utilization 리스트(stage마다 head별 r_bar, 위 문제를 피하려고 따로 반환)
    를 반환한다. balance_loss만 보면 "uniform도 완벽하게 balanced한 최적해"라는 함정에
    빠지므로, 이게 진짜 분업이 일어났는지 판정하는 핵심 지표다."""
    entropies, r_bars = [], []
    for alpha in alphas:
        num_heads = alpha.shape[1]
        if num_heads < 2:
            continue
        r = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        ent = -(r * torch.log(r.clamp_min(1e-8))).sum(dim=1)  # (B,H,W)
        norm_ent = ent / math.log(num_heads)
        entropies.append(norm_ent.mean())
        r_bars.append(r.mean(dim=(0, 2, 3)))
    if not entropies:
        return float("nan"), [], [], []
    stage_entropies = [e.item() for e in entropies]
    stage_r_bars = [rb.tolist() for rb in r_bars]  # [[dec1_H1..H4], [dec2_H1..H4], ...]
    avg_entropy = torch.stack(entropies).mean().item()
    avg_r_bar = torch.stack(r_bars).mean(dim=0).tolist()
    return avg_entropy, avg_r_bar, stage_entropies, stage_r_bars


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    task_names: list[str],
    lambda_bal: float,
    lambda_ent_per_stage: list[float],
    lambda_hard: float = 0.0,
    hard_organs: list[str] | None = None,
    hard_organ_stages: list[str] | None = None,
    balance_target: str = "all",
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_seg_loss = 0.0
    total_bal_loss = 0.0
    total_ent_loss = 0.0  # 최적화에 실제로 쓰인 가중합(lambda 이미 곱해짐, stage마다 다를 수 있음)
    total_hard_loss = 0.0  # 최적화에 실제로 쓰인 가중합(lambda_hard 이미 곱해짐)
    total_samples = 0
    task_dice_sum = {name: 0.0 for name in task_names}
    task_iou_sum = {name: 0.0 for name in task_names}
    task_pos_count = {name: 0 for name in task_names}
    cos_sum, batches = 0.0, 0
    entropy_sum = 0.0          # 진단용(가중치 안 곱한, stage 평균, 0~1 해석 가능)
    stage_entropy_sum = None   # 진단용(stage별로 따로, dec1~3 vs dec4 비교용)
    r_bar_sum = None
    stage_r_bar_sum = None      # 진단용(stage x head 전부 따로 -- stage 평균이 서로 다른 stage의 독점을 가릴 수 있어서)

    progress = tqdm(loader, leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.set_grad_enabled(training):
            logits, alphas = model(images)
            seg_loss = combined_loss_multitask(logits, masks)
            if balance_target == "fg":
                fg_mask_full = (masks.sum(dim=1, keepdim=True) > 0).float()
                bal_loss = head_balance_loss_fg(alphas, fg_mask_full)
            else:
                bal_loss = head_balance_loss(alphas)
            ent_loss = head_entropy_loss(alphas, lambda_ent_per_stage)  # stage별 lambda 이미 곱해진 값
            if lambda_hard > 0.0:
                # model(images) 호출 직후라 각 gate.last_r_for_loss가 이미 채워져 있음(forward에서 캐싱).
                hard_loss_raw = hard_organ_routing_loss(model, masks, task_names, hard_organs, hard_organ_stages)
            else:
                hard_loss_raw = torch.zeros((), device=device)
            hard_loss = lambda_hard * hard_loss_raw
            loss = seg_loss + lambda_bal * bal_loss + ent_loss + hard_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        with torch.no_grad():
            cos = head_diversity_penalty(alphas)
            cos_sum += cos.item() if hasattr(cos, "item") else float(cos)
            ent_diag, r_bar, stage_ent, stage_rbar = router_diagnostics(alphas)  # 가중치 안 곱은 순수 진단용(0~1)
            entropy_sum += ent_diag
            if r_bar_sum is None:
                r_bar_sum = [0.0] * len(r_bar)
                stage_entropy_sum = [0.0] * len(stage_ent)
                stage_r_bar_sum = [[0.0] * len(hs) for hs in stage_rbar]
            r_bar_sum = [acc + v for acc, v in zip(r_bar_sum, r_bar)]
            stage_entropy_sum = [acc + v for acc, v in zip(stage_entropy_sum, stage_ent)]
            stage_r_bar_sum = [
                [acc + v for acc, v in zip(stage_acc, stage_vals)]
                for stage_acc, stage_vals in zip(stage_r_bar_sum, stage_rbar)
            ]
        batches += 1

        metrics = positive_segmentation_metrics_multitask(logits.detach(), masks, task_names)

        total_loss += loss.item() * batch_size
        total_seg_loss += seg_loss.item() * batch_size
        total_bal_loss += bal_loss.item() * batch_size
        total_ent_loss += ent_loss.item() * batch_size
        total_hard_loss += hard_loss.item() * batch_size
        total_samples += batch_size

        postfix = {
            "loss": f"{loss.item():.3f}", "bal": f"{bal_loss.item():.4f}",
            "ent_w": f"{ent_loss.item():.4f}", "hard_w": f"{hard_loss.item():.4f}",
        }
        for name in task_names:
            dice, iou, n_pos = metrics[name]
            if n_pos > 0:
                task_dice_sum[name] += dice * n_pos
                task_iou_sum[name] += iou * n_pos
                task_pos_count[name] += n_pos
            postfix[f"{name}_dice"] = f"{dice:.3f}" if dice is not None else "n/a"
        progress.set_postfix(**postfix)

    task_dice = {
        name: (task_dice_sum[name] / task_pos_count[name]) if task_pos_count[name] > 0 else 0.0
        for name in task_names
    }
    task_iou = {
        name: (task_iou_sum[name] / task_pos_count[name]) if task_pos_count[name] > 0 else 0.0
        for name in task_names
    }
    avg_cos = cos_sum / max(batches, 1)
    avg_bal = total_bal_loss / max(total_samples, 1)
    avg_entropy = entropy_sum / max(batches, 1)  # 가중치 없는 순수 진단값(0~1), stage 평균
    avg_r_bar = [v / max(batches, 1) for v in (r_bar_sum or [])]
    avg_stage_entropy = [v / max(batches, 1) for v in (stage_entropy_sum or [])]
    avg_stage_r_bar = [[v / max(batches, 1) for v in stage_vals] for stage_vals in (stage_r_bar_sum or [])]
    avg_hard_loss = total_hard_loss / max(total_samples, 1)
    return (
        MultiTaskEpochResult(loss=total_loss / total_samples, task_dice=task_dice, task_iou=task_iou),
        avg_cos,
        avg_bal,
        avg_entropy,
        avg_r_bar,
        avg_stage_entropy,
        avg_stage_r_bar,
        avg_hard_loss,
    )


def train_competitive(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
    task_names: list[str],
    num_heads: int,
    lambda_bal: float,
    lambda_ent_per_stage: list[float],
    gate_type: str,
    resume_from: str | None = None,
    lambda_hard: float = 0.0,
    hard_organs: list[str] | None = None,
    hard_organ_stages: list[str] | None = None,
    balance_target: str = "all",
):
    if lambda_hard > 0.0 and gate_type != "competitive_moe":
        raise ValueError(
            f"lambda_hard={lambda_hard} > 0인데 gate_type={gate_type!r} -- hard_organ_routing_loss는 "
            f"gate.last_r_for_loss(competitive_moe 전용)를 필요로 함. ChannelSplit 등 다른 gate_type에는 "
            f"이 개념 자체가 없음(head 간 경쟁이 없어서 s/r 분리도 없음)."
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    best_score = -1.0
    best_task_dice = {name: -1.0 for name in task_names}
    best_task_iou = {name: -1.0 for name in task_names}
    history: list[dict] = []
    start_epoch = 1

    if resume_from is not None:
        print(f"\n--resume-from={resume_from} 에서 이어서 학습")
        resume_ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            print("  optimizer 상태 복원됨(정확한 resume)")
        else:
            print("  optimizer 상태가 체크포인트에 없음(이 체크포인트는 resume 기능 추가 전에 저장된 것) "
                  "-- model weight만 복원하고 optimizer는 새로 시작(근사 resume). "
                  "momentum/adaptive lr 상태가 없어서 처음 1~2 epoch은 약간 흔들릴 수 있음.")
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        best_score = resume_ckpt.get("best_score", -1.0)
        for n in task_names:
            if f"best_val_{n}_pos_dice" in resume_ckpt:
                best_task_dice[n] = resume_ckpt[f"best_val_{n}_pos_dice"]
                best_task_iou[n] = resume_ckpt[f"best_val_{n}_pos_iou"]
        if best_score < 0 and all(v >= 0 for v in best_task_dice.values()):
            best_score = sum(best_task_dice.values()) / len(task_names)

        history_path_existing = output_dir / f"history_{name}.csv"
        if history_path_existing.exists():
            with history_path_existing.open(encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    history.append({k: (float(v) if k != "epoch" and k != "num_heads" else int(float(v))) for k, v in row.items()})

        # "epoch" 필드가 체크포인트에 없는 경우(resume 기능 추가 전 버전으로 저장된 구버전
        # 체크포인트 -- 지금 실제 중단된 run이 정확히 이 케이스) history CSV에 실제로 몇 epoch이
        # 기록됐는지로 유추한다. ckpt의 "epoch" 필드가 있으면 그게 더 정확하니 그걸 우선한다.
        if "epoch" in resume_ckpt:
            start_epoch = resume_ckpt["epoch"] + 1
        elif history:
            start_epoch = len(history) + 1
            print(f"  (체크포인트에 epoch 필드 없음 -> history CSV 행 수({len(history)})로 시작 epoch 유추)")
        else:
            start_epoch = 1

        if history:
            print(f"  기존 history {len(history)}개 epoch 이어붙임 (epoch {start_epoch}부터 새로 추가)")
        print(f"  epoch {start_epoch}부터 재개, best_score(so far)={best_score:.4f}\n")

    start = time.time()

    print(f"\n{name}")
    print(
        f"num_heads={num_heads}, lambda_bal={lambda_bal}, "
        f"lambda_ent_per_stage(dec1..dec4)={lambda_ent_per_stage}, task_names={task_names}, "
        f"balance_target={balance_target}"
    )
    print(f"Trainable parameters: {count_parameters(model):,}")

    for epoch in range(start_epoch, epochs + 1):
        train_result, train_cos, train_bal, train_ent, train_rbar, train_stage_ent, train_stage_rbar, train_hard = run_epoch(
            model, train_loader, optimizer, device, task_names, lambda_bal, lambda_ent_per_stage,
            lambda_hard, hard_organs, hard_organ_stages, balance_target,
        )
        val_result, val_cos, val_bal, val_ent, val_rbar, val_stage_ent, val_stage_rbar, val_hard = run_epoch(
            model, val_loader, None, device, task_names, lambda_bal, lambda_ent_per_stage,
            lambda_hard, hard_organs, hard_organ_stages, balance_target,
        )
        val_score = sum(val_result.task_dice.values()) / len(task_names)
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        train_str = ", ".join(f"train {n}_dice={train_result.task_dice[n]:.4f}" for n in task_names)
        val_str = ", ".join(
            f"val {n}_dice={val_result.task_dice[n]:.4f}, val {n}_iou={val_result.task_iou[n]:.4f}"
            for n in task_names
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss={train_result.loss:.4f}, {train_str} | "
            f"val loss={val_result.loss:.4f}, {val_str} | lr={current_lr:.2e}"
        )
        rbar_str = ", ".join(f"H{i+1}={v:.3f}" for i, v in enumerate(val_rbar))
        stage_ent_str = ", ".join(f"dec{i+1}={v:.4f}" for i, v in enumerate(val_stage_ent))
        print(
            f"  head_cosine(train/val) = {train_cos:.4f}/{val_cos:.4f} | "
            f"balance_loss(train/val) = {train_bal:.4f}/{val_bal:.4f} | "
            f"router_entropy_norm(train/val, 4stage평균) = {train_ent:.4f}/{val_ent:.4f} (1.0=완전uniform, 0=완전sharp)"
        )
        if lambda_hard > 0.0:
            print(
                f"  hard_organ_routing_loss(가중치 곱해진 값, train/val) = {train_hard:.4f}/{val_hard:.4f} "
                f"(organs={hard_organs}, stages={hard_organ_stages})"
            )
        print(f"  val router utilization(r_bar, stage평균): {rbar_str}")
        print(f"  val router entropy(stage별, dec1~dec4): {stage_ent_str}")
        for si, hs in enumerate(val_stage_rbar):
            hs_str = ", ".join(f"H{hi+1}={v:.3f}" for hi, v in enumerate(hs))
            print(f"    dec{si+1} utilization: {hs_str}")

        row = {
            "epoch": epoch,
            "num_heads": num_heads,
            "lambda_bal": lambda_bal,
            "balance_target": balance_target,
            "train_loss": train_result.loss,
            "val_loss": val_result.loss,
            "val_score": val_score,
            "head_cosine_train": train_cos,
            "head_cosine_val": val_cos,
            "balance_loss_train": train_bal,
            "balance_loss_val": val_bal,
            "router_entropy_norm_train": train_ent,
            "router_entropy_norm_val": val_ent,
            "lambda_hard": lambda_hard,
            "hard_organ_routing_loss_train": train_hard,
            "hard_organ_routing_loss_val": val_hard,
        }
        for i, v in enumerate(val_rbar):
            row[f"val_router_util_H{i+1}"] = v
        for i, v in enumerate(val_stage_ent):
            row[f"val_router_entropy_dec{i+1}"] = v
        for si, hs in enumerate(val_stage_rbar):
            for hi, v in enumerate(hs):
                row[f"val_util_dec{si+1}_H{hi+1}"] = v
        for n in task_names:
            row[f"train_{n}_pos_dice"] = train_result.task_dice[n]
            row[f"val_{n}_pos_dice"] = val_result.task_dice[n]
            row[f"val_{n}_pos_iou"] = val_result.task_iou[n]
        history.append(row)

        if val_score > best_score:
            best_score = val_score
            best_task_dice = dict(val_result.task_dice)
            best_task_iou = dict(val_result.task_iou)
            ckpt = {
                "model_name": name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "task_names": task_names,
                "num_heads": num_heads,
                "gate_type": gate_type,
                "lambda_bal": lambda_bal,
                "lambda_ent_per_stage": lambda_ent_per_stage,
                "lambda_hard": lambda_hard,
                "hard_organs": hard_organs,
                "hard_organ_stages": hard_organ_stages,
                "balance_target": balance_target,
                "head_cosine_at_best": val_cos,
            }
            for n in task_names:
                ckpt[f"best_val_{n}_pos_dice"] = best_task_dice[n]
                ckpt[f"best_val_{n}_pos_iou"] = best_task_iou[n]
            torch.save(ckpt, output_dir / f"{name}.pt")

        history_path = output_dir / f"history_{name}.csv"
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    elapsed = time.time() - start
    print(f"Epoch별 기록 저장: {history_path}")

    result = {
        "model": name,
        "num_heads": num_heads,
        "lambda_bal": lambda_bal,
        "lambda_ent_per_stage": lambda_ent_per_stage,
        "balance_target": balance_target,
        "lambda_hard": lambda_hard,
        "hard_organs": hard_organs,
        "hard_organ_stages": hard_organ_stages,
        "parameters": count_parameters(model),
        "training_seconds": elapsed,
        "final_head_cosine_val": val_cos,
        "final_router_entropy_norm_val": val_ent,
    }
    for i, v in enumerate(val_rbar):
        result[f"final_val_router_util_H{i+1}"] = v
    for i, v in enumerate(val_stage_ent):
        result[f"final_val_router_entropy_dec{i+1}"] = v
    for n in task_names:
        result[f"best_val_{n}_pos_dice"] = best_task_dice[n]
        result[f"best_val_{n}_pos_iou"] = best_task_iou[n]
    result["best_val_mean_dice"] = sum(best_task_dice.values()) / len(task_names)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=str, default="/content/data/amos_prepared")
    parser.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/amos22/outputs_competitive")
    parser.add_argument("--task-names", type=str, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument(
        "--gate-type", type=str, default="competitive_moe",
        choices=["competitive_moe", "multi_split"],
        help="'competitive_moe'(기본값) = 새로 설계한 shared-relevance+softmax competitive router. "
        "'multi_split' = 기존 독립 sigmoid 게이트(ChannelSplitMultiHeadAttentionGate) 위에 "
        "balance/entropy loss만 얹은 버전 -- 2026-09 seed42/43이 파일 버전 문제로 실제 이걸로 "
        "돌아갔었던 걸 재현/추가 seed 확보용으로 명시적으로 고를 수 있게 함.",
    )
    parser.add_argument(
        "--lambda-bal", type=float, required=True,
        help="load-balancing loss 가중치. 0이면 balance loss 없이 competitive softmax gate만 (그것도 "
        "그 자체로 의미 있는 비교: 경쟁만 넣었을 때 저절로 안 갈라지는지 확인 가능).",
    )
    parser.add_argument(
        "--balance-target", type=str, default="all", choices=["all", "fg"],
        help="load-balancing loss의 population 평균을 어디서 계산할지. 'all'(기본값)=전체 픽셀 "
        "기준(기존 동작과 100%% 동일). 'fg'=foreground(장기 존재) 픽셀만 -- G_h가 아니라 F_h를 "
        "직접 균형화 목표로 삼는 대조실험용. gate_type='competitive_moe' 전용.",
    )
    parser.add_argument(
        "--lambda-ent", type=float, default=0.0,
        help="entropy minimization 가중치(개별 픽셀에서 라우팅을 sharp하게), 4개 stage에 동일하게 "
        "적용. 0(기본값)이면 competitive softmax + balance loss만. --lambda-ent-per-stage가 "
        "주어지면 이 값은 무시됨.",
    )
    parser.add_argument(
        "--lambda-ent-per-stage", type=str, default=None,
        help="콤마 4개(dec1,dec2,dec3,dec4 순서)로 stage별 다른 lambda_ent를 줄 때 사용. 예: "
        "'0.05,0.05,0.05,0.0'은 dec4만 entropy 압력을 끄는 것(dec4 shortcut 진단 결과 확인용). "
        "주어지면 --lambda-ent는 무시됨.",
    )
    parser.add_argument(
        "--lambda-hard", type=float, default=0.0,
        help="Hard-Organ Routing Confidence Loss 가중치. 0(기본값)이면 완전히 비활성 "
        "(기존 실험과 100%% 동일 동작). >0이면 --hard-organs로 지정한 장기의 GT 마스크 안에서만 "
        "--hard-organ-stages로 지정한 stage(들)의 router entropy를 최소화 -- gate_type='competitive_moe' "
        "전용(ChannelSplit에는 이 개념 자체가 없음).",
    )
    parser.add_argument(
        "--hard-organs", type=str,
        default="gall_bladder,pancreas,left_adrenal_gland,right_adrenal_gland",
        help="Hard-Organ Routing Confidence Loss를 적용할 장기 이름(콤마 구분, task_names 표기와 일치해야 함). "
        "--lambda-hard=0이면 무시됨.",
    )
    parser.add_argument(
        "--hard-organ-stages", type=str, default="dec3",
        help="Hard-Organ Routing Confidence Loss를 적용할 decoder stage(콤마 구분, 예: 'dec3' 또는 "
        "'dec2,dec3'). 두 seed에서 Spearman correlation이 제일 강하고 안정적으로 재현된 stage가 dec3라서 "
        "기본값으로 둠. --lambda-hard=0이면 무시됨.",
    )
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--cache-dataset", action="store_true",
        help="image+mask를 resize까지 끝낸 상태로 프로세스 메모리에 캐싱(두 번째 epoch부터 "
             "디스크 I/O 생략). HDD I/O가 병목일 때 사용. task 수*샘플 수가 크면 메모리를 "
             "많이 쓰니(대략 (1+num_tasks)*image_size^2 bytes/sample) 켜기 전에 여유 RAM 확인.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-imagenet-pretrained", action="store_true")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="중단된 학습을 이어서 할 체크포인트(.pt) 경로. 주어지면 그 체크포인트의 model/optimizer/"
        "scheduler 상태(있으면)를 복원하고 저장된 epoch+1부터 --epochs까지 이어서 학습함. "
        "output-dir/task-names/num-heads/gate-type/lambda 값들은 원래 실행과 동일하게 맞춰서 줘야 함 "
        "(모델 이름이 똑같이 나와야 같은 history/체크포인트 파일에 이어붙임).",
    )
    args = parser.parse_args()

    if args.balance_target == "fg" and args.gate_type != "competitive_moe":
        raise ValueError(
            f"--balance-target fg는 gate_type='competitive_moe' 전용임(받은 gate_type={args.gate_type!r}). "
            f"ChannelSplit 등 다른 gate_type에는 head 간 경쟁이 없어서 이 정규화 자체가 의미 없음."
        )

    seed_everything(args.seed)

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    task_names = [t.strip() for t in args.task_names.split(",")]
    print(f"task_names ({len(task_names)}개): {task_names}")

    train_df = load_split(csv_dir, "train", task_names, args.min_organs_present)
    val_df = load_split(csv_dir, "val", task_names, args.min_organs_present)

    train_dataset = make_dataset(
        train_df, task_names, args.image_size, augment=True, cache_in_memory=args.cache_dataset,
    )
    val_dataset = make_dataset(
        val_df, task_names, args.image_size, augment=False, cache_in_memory=args.cache_dataset,
    )
    if args.cache_dataset:
        est_gb = (1 + len(task_names)) * args.image_size ** 2 * (len(train_dataset) + len(val_dataset)) / 1e9
        print(f"--cache-dataset 켜짐 -- 대략 {est_gb:.1f}GB RAM 예상(전부 캐싱됐을 때 최대치)")

    ratios = ", ".join(f"{n} 양성 {train_dataset.positive_ratio(n):.3f}" for n in task_names)
    print(f"이미지 수 -> train: {len(train_dataset)} ({ratios}), val: {len(val_dataset)}")

    pos_weights = estimate_pixel_pos_weights(train_dataset)
    set_pos_weights(pos_weights)
    print("픽셀 단위 고정 pos_weight -> " + ", ".join(f"{n}: {w:.2f}" for n, w in zip(task_names, pos_weights)))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )

    if args.lambda_ent_per_stage is not None:
        lambda_ent_per_stage = [float(v) for v in args.lambda_ent_per_stage.split(",")]
        if len(lambda_ent_per_stage) != 4:
            raise ValueError(
                f"--lambda-ent-per-stage는 dec1~dec4 4개 값이어야 함(받은 값: {lambda_ent_per_stage})"
            )
    else:
        lambda_ent_per_stage = [args.lambda_ent] * 4
    print(f"lambda_ent_per_stage(dec1..dec4) = {lambda_ent_per_stage}")

    imagenet_pretrained = not args.no_imagenet_pretrained
    bal_tag = f"bal{args.lambda_bal:.3f}".replace(".", "")
    ent_tag = "ent" + "-".join(f"{v:.3f}".replace(".", "") for v in lambda_ent_per_stage)
    gate_tag = "cmoe" if args.gate_type == "competitive_moe" else "chsplit"
    bal_scope_tag = "" if args.balance_target == "all" else "_fgbal"

    hard_organs = [o.strip() for o in args.hard_organs.split(",")] if args.lambda_hard > 0.0 else None
    hard_organ_stages = [s.strip() for s in args.hard_organ_stages.split(",")] if args.lambda_hard > 0.0 else None
    if args.lambda_hard > 0.0:
        hard_tag = f"_hard{args.lambda_hard:.3f}".replace(".", "") + "-" + "-".join(hard_organ_stages)
        print(f"Hard-Organ Routing Confidence Loss 활성: lambda_hard={args.lambda_hard}, "
              f"organs={hard_organs}, stages={hard_organ_stages}")
    else:
        hard_tag = ""

    name = (
        f"multihead_attention_unet_amos_H{args.num_heads}_{len(task_names)}organs_"
        f"{gate_tag}_{bal_tag}_{ent_tag}{hard_tag}{bal_scope_tag}_seed{args.seed}"
    )
    model = AttentionUNetResNet34MultiTask(
        num_tasks=len(task_names), gate_type=args.gate_type, num_heads=args.num_heads,
        imagenet_pretrained=imagenet_pretrained,
    )

    result = train_competitive(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, learning_rate=args.lr, output_dir=output_dir,
        task_names=task_names, num_heads=args.num_heads, lambda_bal=args.lambda_bal,
        lambda_ent_per_stage=lambda_ent_per_stage, gate_type=args.gate_type,
        resume_from=args.resume_from,
        lambda_hard=args.lambda_hard, hard_organs=hard_organs, hard_organ_stages=hard_organ_stages,
        balance_target=args.balance_target,
    )
    result["seed"] = args.seed

    csv_path = output_dir / "comparison_amos_competitive.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print("\nComparison (AMOS Competitive Gate)")
    print(result)
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()