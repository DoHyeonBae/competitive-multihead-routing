"""
챗지피티 로드맵의 ①Spearman correlation, ②near-zero head 동시 제거(redundancy 확인)를
한 스크립트로 처리.

Part A -- Spearman correlation (재학습/추가 forward pass 전혀 필요 없음, 순수 CSV 분석)
    diagnose_head_organ_specialization.py가 저장한 head_organ_share.csv(stage,organ,
    head,share,n_pixels)와 diagnose_ablation_uniform_and_head_removal.py가 저장한
    summary_ablation.csv(config,organ,dice,delta)를 (stage,organ,head)로 merge해서,
    "이 head가 이 장기의 routing을 얼마나 많이 담당하는가(share)"와 "이 head를 지우면
    이 장기 Dice가 얼마나 떨어지는가(delta)"의 Spearman 순위상관을 계산한다.
    가설이 맞다면 rho(share, delta) < 0 (share 높을수록 delta가 더 음수, 즉 더 많이
    떨어짐)이어야 한다. 전체 pooled 상관 + stage별 상관을 둘 다 보고한다(stage별로
    보는 이유: dec1처럼 애초에 routing이 거의 uniform인 stage는 share 자체의 분산이
    작아서 상관이 약하게 나올 수 있는데, 이게 "특화-성능 연결이 약하다"는 뜻이 아니라
    "애초에 특화가 없었다"는 뜻이므로 구분해서 봐야 함).

Part B -- Near-zero head 동시 제거(redundancy 확인, forward pass만 필요, 재학습 없음)
    summary_ablation.csv에서 개별 제거 효과가 --redundancy-threshold(기본 0.005) 미만인
    head들을 stage별로 자동으로 골라서, 그 head들을 한꺼번에 꺼서 다시 평가한다.
    "개별 delta 합" vs "동시 제거 실측 delta"를 비교:
        interaction = joint_delta - sum(individual_deltas)
    interaction이 0에 가까우면 그 head들은 서로 독립적으로 정말 안 쓰이는 것(진짜
    redundant). interaction이 뚜렷하게 음수(추가로 더 떨어짐)면, 개별로는 안 보이던
    상호보완(hidden redundancy)이 있었다는 뜻 -- 개별 ablation만으로는 몰랐던 정보라
    체크포인트 하나로 사실상 공짜로 얻는 셈.

Run (Part A만, 체크포인트 불필요):
    python analyze_correlation_and_joint_removal.py \
        --share-csv outputs_competitive_REAL/head_organ_analysis_full/head_organ_share.csv \
        --ablation-csv outputs_competitive_REAL/ablation_analysis/summary_ablation.csv \
        --output-dir outputs_competitive_REAL/correlation_analysis

Run (Part A + Part B, --checkpoint 추가로 주면 joint removal까지):
    python analyze_correlation_and_joint_removal.py \
        --share-csv outputs_competitive_REAL/head_organ_analysis_full/head_organ_share.csv \
        --ablation-csv outputs_competitive_REAL/ablation_analysis/summary_ablation.csv \
        --checkpoint outputs_competitive_REAL/multihead_attention_unet_amos_H4_8organs_cmoe_bal0100_ent0000-0000-0000-0000_seed42.pt \
        --csv-dir data/content/data/amos_prepared \
        --num-samples 2242 \
        --output-dir outputs_competitive_REAL/correlation_analysis
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REMOVE_RE = re.compile(r"^remove_(dec[1-4])_H(\d+)$")


def spearman(x: pd.Series, y: pd.Series) -> float:
    """scipy 없이(사용자 venv에 scipy가 없을 수 있어서 -- pandas.Series.corr(method='spearman')은
    내부적으로 scipy.stats.spearmanr을 import해서 scipy 미설치 시 에러남) Spearman 순위상관을
    계산. 정의상 Spearman rho = 두 변수의 순위(rank)에 대한 Pearson 상관과 완전히 동일하므로,
    pandas 기본 rank() + 기본(pearson) corr()만으로 정확히 같은 값이 나온다."""
    return float(x.rank().corr(y.rank()))


def load_and_merge(share_csv: Path, ablation_csv: Path) -> pd.DataFrame:
    share_df = pd.read_csv(share_csv)
    share_df = share_df[share_df["organ"] != "background"].copy()  # ablation엔 background 자체가 없음

    ablation_df = pd.read_csv(ablation_csv)
    parsed = ablation_df["config"].str.extract(REMOVE_RE)
    ablation_df = ablation_df.assign(stage=parsed[0], head=parsed[1])
    ablation_df = ablation_df.dropna(subset=["stage", "head"]).copy()
    ablation_df["head"] = ablation_df["head"].astype(int)

    merged = pd.merge(
        share_df, ablation_df[["stage", "organ", "head", "delta"]],
        on=["stage", "organ", "head"], how="inner",
    )
    if len(merged) == 0:
        raise ValueError(
            "merge 결과가 비어 있음 -- share_csv/ablation_csv의 stage/organ/head 표기가 서로 "
            "안 맞는지 확인할 것(예: organ 이름 철자, stage 이름 'dec1' 형식 일치 여부)."
        )
    return merged


def report_spearman(merged: pd.DataFrame) -> None:
    print("=" * 70)
    print("[Part A] Spearman correlation: head organ-routing share <-> head-removal ΔDice")
    print("=" * 70)
    print(
        "가설: share가 높을수록(그 head가 그 장기를 많이 담당할수록) 그 head를 지웠을 때 "
        "delta가 더 음수(더 많이 떨어짐) -- 즉 rho(share, delta) < 0 이 나와야 함.\n"
    )

    rho_all = spearman(merged["share"], merged["delta"])
    n_all = len(merged)
    print(f"[전체 pooled] n={n_all}, rho(share, delta) = {rho_all:.4f}  (rho(share, -delta) = {-rho_all:.4f})")
    print()

    for stage in ["dec1", "dec2", "dec3", "dec4"]:
        sub = merged[merged["stage"] == stage]
        if len(sub) < 3 or sub["share"].nunique() < 2:
            print(f"[{stage}] n={len(sub)} -- 표본/분산 부족으로 상관 생략")
            continue
        rho = spearman(sub["share"], sub["delta"])
        print(f"[{stage}] n={len(sub)}, rho(share, delta) = {rho:.4f}")
    print()


def save_scatter(merged: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"dec1": "tab:blue", "dec2": "tab:green", "dec3": "tab:orange", "dec4": "tab:red"}
    for stage, color in colors.items():
        sub = merged[merged["stage"] == stage]
        ax.scatter(sub["share"], sub["delta"], label=stage, color=color, alpha=0.7)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("head organ-routing share")
    ax.set_ylabel("ΔDice (head 제거 후 - baseline)")
    ax.set_title("routing share vs head-removal 성능 손실 (stage별 색)")
    ax.legend()
    fig.tight_layout()
    out_path = output_dir / "share_vs_delta_scatter.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"저장: {out_path}")


def run_joint_removal(
    ablation_df_path: Path, checkpoint: str, csv_dir: str, num_samples: int,
    batch_size: int, image_size: int, min_organs_present: int,
    output_dir: Path, redundancy_threshold: float,
) -> None:
    from ablation_utils import detect_gate_type, evaluate_config, mean_of
    from attention_unet_multitask import AttentionUNetResNet34MultiTask
    from train_amos_fixed_heads import load_split, make_dataset

    print("=" * 70)
    print("[Part B] Near-zero head 동시 제거 (redundancy 확인)")
    print("=" * 70)

    ablation_df = pd.read_csv(ablation_df_path)
    parsed = ablation_df["config"].str.extract(REMOVE_RE)
    ablation_df = ablation_df.assign(stage=parsed[0], head=parsed[1])
    ablation_df = ablation_df.dropna(subset=["stage", "head"]).copy()
    ablation_df["head"] = ablation_df["head"].astype(int)

    # stage x head별 organ 평균 delta(=summary에 찍혔던 "mean Δ")를 그대로 재계산
    stage_head_mean = ablation_df.groupby(["stage", "head"])["delta"].mean()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    task_names = ckpt["task_names"]
    num_heads = ckpt["num_heads"]
    gate_type = detect_gate_type(ckpt)
    if gate_type != "competitive_moe":
        raise ValueError(f"gate_type={gate_type!r} -- 이 joint-removal은 competitive_moe 체크포인트 전용.")

    val_df = load_split(Path(csv_dir), "val", task_names, min_organs_present)
    val_df = val_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    sample_df = val_df.iloc[:num_samples]
    val_dataset = make_dataset(sample_df, task_names, image_size, augment=False)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"val 샘플 수: {len(val_dataset)}")

    model = AttentionUNetResNet34MultiTask(
        num_tasks=len(task_names), gate_type=gate_type, num_heads=num_heads, imagenet_pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    baseline = evaluate_config(model, loader, task_names, device)

    for stage in ["dec4", "dec3", "dec2", "dec1"]:
        heads_in_stage = [h for (s, h) in stage_head_mean.index if s == stage]
        near_zero = [h for h in heads_in_stage if abs(stage_head_mean[(stage, h)]) < redundancy_threshold]
        print(f"\nStage {stage}: 개별 |mean Δ| < {redundancy_threshold} 인 head = {near_zero}")
        if len(near_zero) < 2:
            print("  -> 동시에 제거할 head가 2개 미만이라 joint removal 생략(개별 결과와 다를 이유가 적음)")
            continue

        mask = torch.ones(num_heads, device=device)
        for h in near_zero:
            mask[h - 1] = 0.0
        joint_result = evaluate_config(model, loader, task_names, device, active_mask_map={stage: mask})
        joint_mean_delta = mean_of(joint_result) - mean_of(baseline)

        # 개별 delta 합(같은 head들의 stage_head_mean 합)
        sum_individual = sum(stage_head_mean[(stage, h)] for h in near_zero)
        interaction = joint_mean_delta - sum_individual

        print(f"  개별 delta 합(mean, {len(near_zero)}개 head) = {sum_individual:+.4f}")
        print(f"  동시 제거 실측 mean Δ                     = {joint_mean_delta:+.4f}")
        print(f"  interaction(실측 - 개별합)                = {interaction:+.4f}"
              f"  {'(거의 0 -> 진짜 redundant)' if abs(interaction) < redundancy_threshold * len(near_zero) else '(무시 못할 상호작용 존재)'}")
        print("  장기별:")
        for organ in task_names:
            d = joint_result[organ] - baseline[organ]
            print(f"      {organ:>22s}: {d:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share-csv", type=str, required=True)
    parser.add_argument("--ablation-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    # Part B(joint removal)는 --checkpoint를 줄 때만 실행됨(안 주면 Part A만 함)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--csv-dir", type=str, default="/content/data/amos_prepared")
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=2242)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--redundancy-threshold", type=float, default=0.005)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_merge(Path(args.share_csv), Path(args.ablation_csv))
    merged.to_csv(output_dir / "correlation_data.csv", index=False, encoding="utf-8-sig")
    report_spearman(merged)
    save_scatter(merged, output_dir)

    if args.checkpoint is not None:
        run_joint_removal(
            Path(args.ablation_csv), args.checkpoint, args.csv_dir, args.num_samples,
            args.batch_size, args.image_size, args.min_organs_present,
            output_dir, args.redundancy_threshold,
        )
    else:
        print("\n--checkpoint 안 줌 -- Part B(joint removal)는 생략. 필요하면 --checkpoint/--csv-dir 추가해서 재실행.")


if __name__ == "__main__":
    main()