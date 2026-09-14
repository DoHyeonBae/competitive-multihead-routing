"""
analyze_per_case_difficulty.py의 결과에 남아있던 두 가지 통계적 약점을 보완:

1) Circularity: "H1이 어려워한 이미지"를 H1 자신의 Dice로 정의하고, 그 위에서
   Delta=(다른 모델-H1)를 다시 H1이 들어간 식으로 비교했다. H1이 양쪽에 다 관여하므로
   일부 상관은 진짜 난이도 신호가 아니라 순환 참조/노이즈일 수 있다.
   -> 해결: 비교하는 두 모델(baseline, compare)을 모두 제외한 "나머지 모델들의 평균
   Dice"를 외부 난이도 지표(consensus difficulty)로 쓴다. 이러면 baseline도 compare도
   난이도 정의에 전혀 관여하지 않으므로 순환성이 원천적으로 없다.

2) Pseudo-replication: validation 3,521장은 서로 독립적인 3,521개 표본이 아니라
   더 적은 수의 환자(patient_id)에서 나온 슬라이스들이다(같은 환자의 슬라이스는
   해부학적으로 매우 비슷해서 사실상 상관돼 있음). image 단위로 Wilcoxon을 돌리면
   실제 독립 표본 수보다 n을 부풀려서 p-value가 과대평가(너무 작게)될 위험이 있다.
   -> 해결: patient_id로 묶어서 환자별 평균 Dice를 낸 뒤, 그 환자 단위(n=환자 수)에서
   같은 통계 검정을 반복한다. image-level과 patient-level 결과가 같은 방향이면
   pseudo-replication 우려가 크지 않다는 뜻이고, 사라지면 image-level 결과를 그대로
   믿으면 안 된다는 뜻이다.

patient_id는 prepare_amos_dataset.py가 amos_{split}.csv에 이미 저장해둔 컬럼이라
새로 계산할 필요가 없다(이미지 파일 자체를 다시 읽지 않고, per_image_dice.csv의
image_index와 val_df를 같은 필터·같은 순서로 다시 불러와 위치 기준으로 합치기만
하면 됨 -- 재추론 불필요).

Run:
    python analyze_difficulty_rigorous.py \
        --dice-csv outputs_percase\\per_image_dice.csv \
        --csv-dir data\\content\\data\\amos_prepared \
        --task-names liver,spleen,right_kidney,left_kidney,stomach,pancreas,gall_bladder,right_adrenal_gland,left_adrenal_gland,duodenum,esophagus,bladder \
        --baseline-label H1 \
        --output-dir outputs_percase
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from train_amos_fixed_heads import load_split


def hard_easy_tertile(difficulty: np.ndarray, delta: np.ndarray) -> tuple[float, float]:
    """difficulty가 낮을수록(=Dice가 낮을수록) 어려운 것으로 정의. 하위 1/3=어려움,
    상위 1/3=쉬움 구간에서 delta 평균을 각각 반환."""
    order = np.argsort(difficulty)
    third = len(order) // 3
    hard_idx = order[:third]
    easy_idx = order[-third:]
    return float(delta[hard_idx].mean()), float(delta[easy_idx].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dice-csv", type=str, required=True)
    parser.add_argument("--csv-dir", type=str, required=True)
    parser.add_argument("--task-names", type=str, required=True)
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--baseline-label", type=str, default="H1")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    task_names = [t.strip() for t in args.task_names.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dice_df = pd.read_csv(args.dice_csv)
    wide = dice_df.pivot(index="image_index", columns="model", values="dice").sort_index()
    models = list(wide.columns)
    if list(wide.index) != list(range(len(wide))):
        raise ValueError(
            "per_image_dice.csv의 image_index가 0..N-1 연속이 아님 -- "
            "analyze_per_case_difficulty.py를 --num-samples 0(전체 val)으로 돌렸는지 확인할 것"
        )
    if args.baseline_label not in models:
        raise ValueError(f"--baseline-label={args.baseline_label!r}이 CSV의 모델 목록 {models}에 없음")

    val_df = load_split(Path(args.csv_dir), "val", task_names, args.min_organs_present)
    if len(val_df) != len(wide):
        raise ValueError(
            f"val_df 길이({len(val_df)})와 per_image_dice.csv 이미지 수({len(wide)})가 다름 -- "
            f"--csv-dir/--task-names/--min-organs-present가 analyze_per_case_difficulty.py 실행 때와 "
            f"똑같은지 확인할 것(순서가 안 맞으면 아래 분석 전체가 무의미해짐)."
        )
    wide["patient_id"] = val_df["patient_id"].values

    n_patients = wide["patient_id"].nunique()
    print(f"이미지 {len(wide)}장, 환자 {n_patients}명 (평균 {len(wide)/n_patients:.1f}장/환자)\n")

    baseline = args.baseline_label
    compare_models = [m for m in models if m != baseline]

    print("=" * 100)
    print("[1] Leave-two-out consensus difficulty (baseline/compare 둘 다 난이도 정의에서 제외 -- circularity 제거)")
    print("=" * 100)
    consensus_rows = []
    for compare in compare_models:
        others = [m for m in models if m not in (baseline, compare)]
        if not others:
            print(f"  {compare}: 제외할 다른 모델이 없어서 스킵")
            continue
        consensus_diff = wide[others].mean(axis=1).values  # 높을수록 쉬움(다른 모델들 기준)
        delta = (wide[compare] - wide[baseline]).values

        rho, p = stats.spearmanr(consensus_diff, delta)
        hard_mean, easy_mean = hard_easy_tertile(consensus_diff, delta)
        stat, wp = stats.wilcoxon(delta)

        print(f"\n{compare} - {baseline} (외부 난이도 = {others} 평균, n={len(delta)}장):")
        print(f"  mean(Delta)={delta.mean():.4f} | Wilcoxon p={wp:.2e}")
        print(f"  Spearman(consensus_difficulty, Delta) rho={rho:.3f}, p={p:.2e} "
              f"(음수면 '다른 모델들 기준으로도 어려운 이미지일수록 격차가 커짐')")
        print(f"  외부 기준 가장 어려운 1/3: mean(Delta)={hard_mean:.4f} | 가장 쉬운 1/3: mean(Delta)={easy_mean:.4f}")
        consensus_rows.append({
            "compare": compare, "level": "image", "mean_delta": delta.mean(), "wilcoxon_p": wp,
            "spearman_rho": rho, "spearman_p": p, "hard_third_delta": hard_mean, "easy_third_delta": easy_mean,
        })

    print("\n" + "=" * 100)
    print("[2] Patient-level 집계 (같은 환자의 슬라이스는 독립 표본이 아님 -- pseudo-replication 보정)")
    print("=" * 100)
    patient_wide = wide.groupby("patient_id")[models].mean()
    print(f"환자 단위 표본 수: n={len(patient_wide)}\n")

    patient_rows = []
    for compare in compare_models:
        others = [m for m in models if m not in (baseline, compare)]
        delta_p = (patient_wide[compare] - patient_wide[baseline]).values
        stat, wp = stats.wilcoxon(delta_p)
        print(f"{compare} - {baseline} (환자 단위, n={len(delta_p)}명):")
        print(f"  mean(Delta)={delta_p.mean():.4f}, median(Delta)={np.median(delta_p):.4f}")
        print(f"  Wilcoxon signed-rank: p={wp:.2e} ({'유의함(p<0.05)' if wp < 0.05 else '유의하지 않음'})")

        if others:
            consensus_diff_p = patient_wide[others].mean(axis=1).values
            rho, p = stats.spearmanr(consensus_diff_p, delta_p)
            hard_mean, easy_mean = hard_easy_tertile(consensus_diff_p, delta_p)
            print(f"  Spearman(환자단위 외부 난이도, Delta) rho={rho:.3f}, p={p:.2e}")
            print(f"  외부 기준 가장 어려운 환자 1/3: mean(Delta)={hard_mean:.4f} | 가장 쉬운 1/3: mean(Delta)={easy_mean:.4f}")
        else:
            rho, p, hard_mean, easy_mean = float("nan"), float("nan"), float("nan"), float("nan")
        print()
        patient_rows.append({
            "compare": compare, "level": "patient", "n": len(delta_p), "mean_delta": delta_p.mean(),
            "wilcoxon_p": wp, "spearman_rho": rho, "spearman_p": p,
            "hard_third_delta": hard_mean, "easy_third_delta": easy_mean,
        })

    pd.DataFrame(consensus_rows).to_csv(output_dir / "consensus_difficulty_image_level.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(patient_rows).to_csv(output_dir / "patient_level_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {output_dir / 'consensus_difficulty_image_level.csv'}, {output_dir / 'patient_level_comparison.csv'}")


if __name__ == "__main__":
    main()
