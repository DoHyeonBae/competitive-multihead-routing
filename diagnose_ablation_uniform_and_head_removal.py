"""
Uniform-router ablation + Head-removal ablation -- "routing이 장기별로 특화됐다"
(head_organ 분석)는 이미 확인했으니, 그 다음 질문 "그 routing이 실제 segmentation
성능에 필요한가"에 재학습 없이 답하는 스크립트.

둘 다 기존 competitive_moe 체크포인트 하나로, forward pass만으로 확인 가능
(backward/재학습 전혀 필요 없음):

1) Uniform-router ablation
   CompetitiveMultiHeadAttentionGate가 추론 시 학습된 r_h(p)를 버리고 1/num_heads로
   강제 치환(s(p)는 그대로 둠) -- attention_unet_multitask.py의
   `gate.force_uniform_router = True` 스위치를 사용. 이걸 전체 stage에 걸어서
   mean Dice가 baseline 대비 얼마나 떨어지는지 보고, 추가로 stage 하나씩만 uniform으로
   바꿔서 "어느 stage의 routing이 제일 중요한가"도 같이 확인한다.

2) Head-removal ablation
   각 decoder stage에서 head를 하나씩 꺼서(active_mask로 그 head의 alpha를 통째로
   0으로 눌러 skip feature 기여를 제거) 장기별 Dice 변화(delta)를 측정. 이걸
   head_organ 분석 heatmap(어느 head가 어느 장기의 routing을 많이 담당하는가)과
   나란히 놓고 일치하는지 확인하는 게 목적 -- 상관관계 증거를 기능적/causal 증거로
   연결.

   주의: 모델 전체 forward()는 active_mask 하나를 4개 decoder stage에 전부 동일하게
   적용하도록 짜여 있어서(model.forward의 기존 시그니처), stage별로 다른 head를
   끄려면 이 스크립트에서 encoder/dec4~dec1을 직접 순서대로 호출해야 한다(아래
   run_forward 참고) -- model.forward()를 그대로 재사용하지 않는 이유가 이것.

출력(전부 표준출력 + summary_ablation.csv):
    Baseline mean Dice = ...
    Baseline per-organ Dice: ...

    Uniform router (all stages): mean Dice=..., Δ=...
    Uniform router (dec4 only):  mean Dice=..., Δ=...
    Uniform router (dec3 only):  mean Dice=..., Δ=...
    Uniform router (dec2 only):  mean Dice=..., Δ=...
    Uniform router (dec1 only):  mean Dice=..., Δ=...

    Stage dec4
      remove H1: mean Δ=..., per-organ Δ: liver=..., spleen=..., ...
      remove H2: ...
      ...

    (그리고 stage마다 organ x removed-head Δ-Dice heatmap PNG 저장)

Run:
    python diagnose_ablation_uniform_and_head_removal.py \
        --checkpoint outputs_competitive_REAL/multihead_attention_unet_amos_H4_8organs_cmoe_bal0100_ent0000-0000-0000-0000_seed42.pt \
        --csv-dir data/content/data/amos_prepared \
        --num-samples 2242 \
        --output-dir outputs_competitive_REAL/ablation_analysis

    먼저 --num-samples 500 정도로 빠르게 한 번 돌려서 스크립트가 잘 도는지,
    baseline mean Dice가 기존에 알던 값(~0.789 근처)과 비슷하게 나오는지부터
    확인한 다음, 전체 2242장으로 최종 수치를 뽑는 걸 권장.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attention_unet_multitask import (
    AttentionUNetResNet34MultiTask,
    positive_segmentation_metrics_multitask,
)
from train_amos_fixed_heads import load_split, make_dataset

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
    완전히 동일하고, active_mask/force_uniform_router만 stage마다 다르게 건다."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv-dir", type=str, default="/content/data/amos_prepared")
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=2242)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--small-organs", type=str, nargs="+",
        default=["gall_bladder", "pancreas", "left_adrenal_gland", "right_adrenal_gland"],
        help="roadmap 8단계 small-organ mean Dice 계산용 organ 이름 목록(task_names와 일치해야 함)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    task_names = ckpt["task_names"]
    num_heads = ckpt["num_heads"]
    gate_type = detect_gate_type(ckpt)
    num_organs = len(task_names)
    print(f"checkpoint: {ckpt.get('model_name', args.checkpoint)}")
    print(f"gate_type={gate_type}, num_heads={num_heads}, task_names={task_names}")

    if gate_type != "competitive_moe":
        raise ValueError(
            f"gate_type={gate_type!r} -- uniform-router ablation은 CompetitiveMultiHeadAttentionGate "
            f"전용임(force_uniform_router 스위치가 이 게이트에만 있음). ChannelSplit 체크포인트는 "
            f"head-removal(active_mask)만 별도로 테스트 가능하지만, 이 스크립트는 둘을 같이 다루므로 "
            f"competitive_moe 체크포인트로 실행할 것."
        )

    csv_dir = Path(args.csv_dir)
    val_df = load_split(csv_dir, "val", task_names, args.min_organs_present)
    val_df = val_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    sample_df = val_df.iloc[: args.num_samples]
    val_dataset = make_dataset(sample_df, task_names, args.image_size, augment=False)
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"val 샘플 수: {len(val_dataset)}\n")

    model = AttentionUNetResNet34MultiTask(
        num_tasks=num_organs, gate_type=gate_type, num_heads=num_heads, imagenet_pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    small_organs = [o for o in args.small_organs if o in task_names]
    missing_small = [o for o in args.small_organs if o not in task_names]
    if missing_small:
        print(f"경고: --small-organs 중 task_names에 없는 이름 무시됨: {missing_small}")

    def report(tag: str, per_organ: dict[str, float], baseline_per_organ: dict[str, float] | None = None) -> None:
        mean_dice = mean_of(per_organ)
        small_dice = mean_of({o: per_organ[o] for o in small_organs}) if small_organs else float("nan")
        line = f"{tag}: mean Dice={mean_dice:.4f}, small-organ mean Dice={small_dice:.4f}"
        if baseline_per_organ is not None:
            delta = mean_dice - mean_of(baseline_per_organ)
            line += f", Δ={delta:+.4f}"
        print(line)
        for organ in task_names:
            d = per_organ[organ]
            extra = ""
            if baseline_per_organ is not None:
                extra = f" (Δ={d - baseline_per_organ[organ]:+.4f})"
            print(f"    {organ:>22s}: {d:.4f}{extra}")
        print()

    all_rows = []  # summary_ablation.csv용

    # ---------------------------------------------------------------
    # 0) Baseline (수정 없음)
    # ---------------------------------------------------------------
    print("=" * 70)
    print("[0] Baseline")
    print("=" * 70)
    baseline = evaluate_config(model, loader, task_names, device)
    report("Baseline", baseline)
    for organ, d in baseline.items():
        all_rows.append({"config": "baseline", "organ": organ, "dice": d, "delta": 0.0})

    # ---------------------------------------------------------------
    # 1) Uniform-router ablation: 전체 stage + stage 하나씩
    # ---------------------------------------------------------------
    print("=" * 70)
    print("[1] Uniform-router ablation (r_h(p) -> 1/num_heads, s(p)는 유지)")
    print("=" * 70)

    uniform_all = evaluate_config(model, loader, task_names, device, uniform_stage_names=set(STAGE_ORDER))
    report("Uniform router (ALL stages)", uniform_all, baseline)
    for organ, d in uniform_all.items():
        all_rows.append({"config": "uniform_all", "organ": organ, "dice": d, "delta": d - baseline[organ]})

    uniform_per_stage: dict[str, dict[str, float]] = {}
    for stage_name in STAGE_ORDER:
        res = evaluate_config(model, loader, task_names, device, uniform_stage_names={stage_name})
        uniform_per_stage[stage_name] = res
        report(f"Uniform router ({stage_name} only)", res, baseline)
        for organ, d in res.items():
            all_rows.append({
                "config": f"uniform_{stage_name}_only", "organ": organ,
                "dice": d, "delta": d - baseline[organ],
            })

    # ---------------------------------------------------------------
    # 2) Head-removal ablation: stage x head 조합
    # ---------------------------------------------------------------
    print("=" * 70)
    print("[2] Head-removal ablation (stage별 head 하나씩 alpha=0으로 제거)")
    print("=" * 70)

    # heatmap용 delta 저장: delta_grid[stage][organ_idx, head_idx] = baseline - removed
    delta_grid = {stage: np.zeros((num_organs, num_heads)) for stage in STAGE_ORDER}

    for stage_name in STAGE_ORDER:
        print(f"\nStage {STAGE_LABELS[stage_name]}")
        for head_idx in range(num_heads):
            mask = torch.ones(num_heads, device=device)
            mask[head_idx] = 0.0
            res = evaluate_config(model, loader, task_names, device, active_mask_map={stage_name: mask})
            mean_delta = mean_of(res) - mean_of(baseline)
            print(f"  remove H{head_idx + 1}: mean Δ={mean_delta:+.4f}")
            for organ_idx, organ in enumerate(task_names):
                delta = res[organ] - baseline[organ]
                delta_grid[stage_name][organ_idx, head_idx] = delta
                print(f"      {organ:>22s}: {delta:+.4f}")
                all_rows.append({
                    "config": f"remove_{stage_name}_H{head_idx + 1}", "organ": organ,
                    "dice": res[organ], "delta": delta,
                })
        print()

    # ---------------------------------------------------------------
    # 3) organ x removed-head Δ-Dice heatmap (stage마다 1장)
    # ---------------------------------------------------------------
    for stage_name in STAGE_ORDER:
        grid = delta_grid[stage_name]  # (num_organs, num_heads), 음수=제거 시 성능 하락(그 head가 중요했다는 뜻)
        fig, ax = plt.subplots(figsize=(1.6 * num_heads + 2, 0.5 * num_organs + 2))
        vmax = np.abs(grid).max() if np.abs(grid).max() > 0 else 1e-4
        im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"remove H{h + 1}" for h in range(num_heads)])
        ax.set_yticks(range(num_organs))
        ax.set_yticklabels(task_names)
        for i in range(num_organs):
            for j in range(num_heads):
                ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{STAGE_LABELS[stage_name]} -- organ x removed-head ΔDice (파랑=하락=그 head가 중요)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="ΔDice (removed - baseline)")
        plt.tight_layout()
        out_path = output_dir / f"head_removal_delta_{stage_name}.png"
        plt.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"저장: {out_path}")

    # ---------------------------------------------------------------
    # 4) CSV로도 저장(추후 표/그림 재사용 용이하게)
    # ---------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "summary_ablation.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 결과 CSV 저장: {csv_path}")
    print(f"총 {len(val_dataset)}장으로 평가 완료")


if __name__ == "__main__":
    main()