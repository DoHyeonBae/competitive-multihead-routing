"""
H1 vs H2/H4(등)를 이미지 단위로 짝지어(paired) 비교 -- "H1이 어려워하는 이미지에서
multi-head가 특히 더 잘하는가"를 확인.

배경: 지금까지의 모든 비교("H1/H2/H4 Mean Dice가 노이즈 문턱 안에서 구분 안 됨")는
seed 하나당 숫자 하나뿐이라 통계 검정 자체가 불가능했다(그래서 8-organ에서 측정한
노이즈 문턱을 대리 지표로 썼다). 그런데 같은 validation 이미지 각각에 대해 여러
모델의 per-image Dice를 직접 비교하면(같은 이미지 i에 대해 Dice_H2(i) - Dice_H1(i)),
seed 재현 없이도 수천 장 단위의 paired sample을 얻어 Wilcoxon signed-rank 같은
통계 검정을 바로 돌릴 수 있다 -- "그래도 H1이면 충분하지 않나?" 질문에 대한 가장
저렴하고 통계적으로 힘 있는 답.

per-image Dice 정의: positive_segmentation_metrics_multitask와 동일한 방식(threshold
0.5, 장기가 실제로 존재하는 경우만)으로 organ별 Dice를 구하고, 그 이미지에 존재하는
장기들에 대해서만 평균낸 값(한 이미지=하나의 스칼라). 학습에 쓴 기존 "positive Dice"
정의와 동일하게 맞춰서 다른 곳의 숫자와 바로 비교 가능하게 했다.

출력:
  1) per_image_dice.csv: image_index, model, dice, n_organs_present
  2) 콘솔에 모델 쌍마다:
     - mean(Delta), Wilcoxon signed-rank p-value(짝지은 두 모델의 차이가 0인지)
     - baseline(보통 H1) 자체 난이도와 Delta의 Spearman 상관(음의 상관이면
       "H1이 어려워하는 이미지일수록 격차가 커진다"는 뜻)
     - baseline 기준 하위/상위 1/3(가장 어려운 쪽 vs 가장 쉬운 쪽) 이미지에서
       각각 mean(Delta) 비교

주의: 이건 "평균 Dice가 노이즈 안"이라는 기존 결론과 모순되지 않는다 -- 평균은
동일해도 이미지별 편차가 상쇄돼서 평균만 같아 보일 수 있고, 이 분석은 바로 그
가능성(상쇄되는 편차에 구조가 있는가)을 보는 것이다. scipy 필요: pip install scipy

Run:
    python analyze_per_case_difficulty.py \
        --csv-dir data\\content\\data\\amos_prepared \
        --baseline-checkpoint outputs_competitive_12organ\\multihead_attention_unet_amos_H1_12organs_cmoe_bal0100_ent0000-0000-0000-0000_seed42.pt \
        --baseline-label H1 \
        --compare-checkpoints outputs_competitive_12organ\\multihead_attention_unet_amos_H2_12organs_cmoe_bal0100_ent0000-0000-0000-0000_seed42.pt,outputs_competitive_12organ\\multihead_attention_unet_amos_H4_12organs_cmoe_bal0100_ent0000-0000-0000-0000_seed42.pt \
        --compare-labels H2,H4 \
        --output-dir outputs_percase
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from attention_unet_multitask import AttentionUNetResNet34MultiTask
from train_amos_fixed_heads import load_split, make_dataset


def detect_gate_type(ckpt: dict) -> str:
    state_keys = ckpt["model_state_dict"].keys()
    if any("proj_heads" in k for k in state_keys):
        return "head_proj_moe"
    if any("psi_router" in k for k in state_keys):
        return "competitive_moe"
    if any("psi_heads" in k for k in state_keys):
        return "multi_split"
    return ckpt.get("gate_type", "competitive_moe")


def load_model(checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    task_names = ckpt["task_names"]
    num_heads = ckpt["num_heads"]
    gate_type = detect_gate_type(ckpt)
    model = AttentionUNetResNet34MultiTask(
        num_tasks=len(task_names), gate_type=gate_type, num_heads=num_heads, imagenet_pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"  로드: {checkpoint_path} (gate_type={gate_type}, num_heads={num_heads})")
    return model, task_names


@torch.no_grad()
def per_image_dice(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """이미지별 Dice(존재하는 장기 평균)와 존재 장기 수를 순서대로 반환."""
    dices, n_present_list = [], []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)  # (B, num_tasks, H, W)
        logits, _alphas = model(images)
        pred = (torch.sigmoid(logits) >= 0.5).float()

        target_sum = masks.sum(dim=(2, 3))  # (B, num_tasks)
        positive = target_sum > 0

        intersection = (pred * masks).sum(dim=(2, 3))
        dice_per_task = (2.0 * intersection + 1e-7) / (pred.sum(dim=(2, 3)) + target_sum + 1e-7)  # (B, num_tasks)

        dice_masked = torch.where(positive, dice_per_task, torch.zeros_like(dice_per_task))
        n_present = positive.sum(dim=1).clamp(min=1)  # 0으로 나누기 방지(이론상 val 필터링으로 항상 >=1)
        image_dice = dice_masked.sum(dim=1) / n_present

        dices.append(image_dice.cpu().numpy())
        n_present_list.append(positive.sum(dim=1).cpu().numpy())

    return np.concatenate(dices), np.concatenate(n_present_list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=str, required=True)
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--baseline-label", type=str, default="H1")
    parser.add_argument("--compare-checkpoints", type=str, required=True, help="쉼표로 구분된 경로들")
    parser.add_argument("--compare-labels", type=str, required=True, help="쉼표로 구분, compare-checkpoints와 순서 맞춰야 함")
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=0, help="0이면 validation 전체 사용")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    compare_paths = [p.strip() for p in args.compare_checkpoints.split(",")]
    compare_labels = [l.strip() for l in args.compare_labels.split(",")]
    if len(compare_paths) != len(compare_labels):
        raise ValueError("--compare-checkpoints와 --compare-labels 개수가 안 맞음")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{args.baseline_label}] 로드")
    baseline_model, task_names = load_model(args.baseline_checkpoint, device)

    csv_dir = Path(args.csv_dir)
    val_df = load_split(csv_dir, "val", task_names, args.min_organs_present)
    if args.num_samples > 0:
        # 재현 가능하게 고정 seed로 셔플 후 자르되, 모든 모델이 "동일한" val_df를
        # 쓰도록(이미지 정렬을 위해) 이 시점에 딱 한 번만 샘플링한다.
        val_df = val_df.sample(frac=1.0, random_state=123).reset_index(drop=True).iloc[: args.num_samples]
    val_dataset = make_dataset(val_df, task_names, args.image_size, augment=False)
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"validation 이미지 수: {len(val_dataset)} (모든 모델이 이 순서 그대로 평가됨)")

    print(f"[{args.baseline_label}] per-image Dice 계산 중...")
    base_dice, n_present = per_image_dice(baseline_model, loader, device)
    del baseline_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    all_rows = [
        {"image_index": i, "model": args.baseline_label, "dice": float(base_dice[i]), "n_organs_present": int(n_present[i])}
        for i in range(len(base_dice))
    ]

    print(f"\n=== {args.baseline_label} 기준 paired 비교 ===")
    for path, label in zip(compare_paths, compare_labels):
        print(f"\n[{label}] 로드 및 계산")
        model, model_task_names = load_model(path, device)
        if model_task_names != task_names:
            raise ValueError(f"{label}의 task_names가 {args.baseline_label}와 다름 -- 같은 12-organ 세팅인지 확인할 것")
        cmp_dice, _ = per_image_dice(model, loader, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        all_rows.extend([
            {"image_index": i, "model": label, "dice": float(cmp_dice[i]), "n_organs_present": int(n_present[i])}
            for i in range(len(cmp_dice))
        ])

        delta = cmp_dice - base_dice
        stat, p_value = stats.wilcoxon(delta)
        rho, rho_p = stats.spearmanr(base_dice, delta)

        order = np.argsort(base_dice)  # 오름차순: 앞쪽=baseline이 어려워한 이미지
        third = len(order) // 3
        hard_idx = order[:third]
        easy_idx = order[-third:]

        print(f"  {label} - {args.baseline_label} (n={len(delta)}장):")
        print(f"    mean(Delta)={delta.mean():.4f}, median(Delta)={np.median(delta):.4f}")
        print(f"    Wilcoxon signed-rank: stat={stat:.1f}, p={p_value:.2e} "
              f"({'유의함(p<0.05)' if p_value < 0.05 else '유의하지 않음'})")
        print(f"    Spearman({args.baseline_label} 난이도, Delta) rho={rho:.3f}, p={rho_p:.2e} "
              f"(음수면 '{args.baseline_label}이 어려워한 이미지일수록 격차가 커짐'을 의미)")
        print(f"    {args.baseline_label} 기준 가장 어려운 1/3: mean(Delta)={delta[hard_idx].mean():.4f}")
        print(f"    {args.baseline_label} 기준 가장 쉬운   1/3: mean(Delta)={delta[easy_idx].mean():.4f}")

    csv_path = output_dir / "per_image_dice.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n원본 저장: {csv_path}")


if __name__ == "__main__":
    main()
