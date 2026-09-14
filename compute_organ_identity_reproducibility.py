# -*- coding: utf-8 -*-
"""
Head-organ identity의 seed 간 permutation-invariant 재현성 검정.

배경: competitive gate의 head 번호(H1/H2)는 독립된 학습(seed)마다 임의로 뒤바뀔 수 있다
(head-permutation invariance). 따라서 "seed42의 H1"과 "seed43의 H1"이 같은 장기를
담당하는지를 그대로 비교하면 안 되고, 각 장기가 "얼마나 한쪽 head에 쏠려 있는가"라는
head-label-무관 지표로 비교해야 한다.

지표: specialization_strength(stage, organ) = |2 * share_head1 - 1|
  - 0   -> 두 head가 정확히 50:50으로 나눠 담당 (분화 없음)
  - 1   -> 한 head가 100% 독점 (완전 분화)
  - head1/head2 라벨이 통째로 뒤바뀌어도 |2*share_head1-1| == |2*share_head2-1|이라
    이 값은 라벨 순서에 불변(permutation-invariant)이다.

이 스크립트는 두 seed의 head_organ_share.csv(diagnose_head_organ_specialization.py
--num-samples 3521로 생성)를 읽어, stage별로 specialization_strength 벡터를 만들고
seed 간 Spearman 상관을 계산한다. 상관이 높고 유의하면 "어느 장기가 강하게 분화되는가"
라는 구조적 패턴이 seed에 무관하게 재현된다는 뜻이다(비록 어느 head 번호가 그 장기를
맡는지는 바뀔 수 있어도).

입력 CSV 스키마(diagnose_head_organ_specialization.py 출력, long format):
    stage,organ,head,share,n_pixels
    dec1,liver,1,0.516965974436207,5087984
    dec1,liver,2,0.4830340262535596,5087984
    ...
    (head는 1,2,... 정수, organ에는 "background" 포함)

Run:
    python compute_organ_identity_reproducibility.py `
        --seed-a-csv outputs_headorgan_seed42\head_organ_share.csv `
        --seed-b-csv outputs_headorgan_seed43\head_organ_share.csv `
        --seed-a-label 42 --seed-b-label 43 `
        --output-dir csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def load_strength(csv_path: str) -> pd.DataFrame:
    """head_organ_share.csv(long format) -> (stage, organ) 별 specialization_strength(head 라벨 무관)."""
    df = pd.read_csv(csv_path)
    required_cols = {"stage", "organ", "head", "share"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}에 예상 컬럼이 없음: {missing} (있는 컬럼: {list(df.columns)})")

    heads = sorted(df["head"].unique())
    if len(heads) != 2:
        raise ValueError(
            f"{csv_path}: head가 2개가 아님({heads}) -- 이 스크립트는 2-head 모델(H2) 전용. "
            "다른 head 수 모델은 별도 처리가 필요함."
        )
    head_a, head_b = heads[0], heads[1]

    wide = df.pivot_table(index=["stage", "organ"], columns="head", values="share").reset_index()
    wide = wide.rename(columns={head_a: "share_head_a", head_b: "share_head_b"})

    # 두 head의 share 합이 1에 가까운지 sanity check (아니면 데이터가 이상한 것)
    sum_check = (wide["share_head_a"] + wide["share_head_b"] - 1.0).abs()
    if (sum_check > 1e-3).any():
        bad = wide[sum_check > 1e-3]
        raise ValueError(f"{csv_path}: head별 share 합이 1이 아닌 (stage,organ) 존재:\n{bad}")

    wide["specialization_strength"] = (2 * wide["share_head_a"] - 1).abs()
    return wide[["stage", "organ", "specialization_strength"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-a-csv", type=str, required=True)
    parser.add_argument("--seed-b-csv", type=str, required=True)
    parser.add_argument("--seed-a-label", type=str, default="42")
    parser.add_argument("--seed-b-label", type=str, default="43")
    parser.add_argument("--output-dir", type=str, default="csv")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strength_a = load_strength(args.seed_a_csv).rename(
        columns={"specialization_strength": f"strength_seed{args.seed_a_label}"}
    )
    strength_b = load_strength(args.seed_b_csv).rename(
        columns={"specialization_strength": f"strength_seed{args.seed_b_label}"}
    )

    merged = strength_a.merge(strength_b, on=["stage", "organ"], how="inner")
    if len(merged) != len(strength_a) or len(merged) != len(strength_b):
        raise ValueError(
            "두 CSV의 (stage, organ) 조합이 정확히 일치하지 않음 -- "
            "같은 --task-names/--min-organs-present로 생성했는지 확인할 것."
        )

    col_a = f"strength_seed{args.seed_a_label}"
    col_b = f"strength_seed{args.seed_b_label}"
    per_organ_path = out_dir / "organ_identity_reproducibility_per_organ.csv"
    merged.to_csv(per_organ_path, index=False, encoding="utf-8-sig")

    print(f"{'stage':6}{'rho(전체)':>12}{'p':>10}{'rho(배경제외)':>14}{'p':>10}{'n':>5}")
    summary_rows = []
    for stage, g in merged.groupby("stage", sort=True):
        rho_all, p_all = spearmanr(g[col_a], g[col_b])
        g_nobg = g[g["organ"] != "background"]
        rho_nobg, p_nobg = spearmanr(g_nobg[col_a], g_nobg[col_b])
        n_all = len(g)
        print(f"{stage:6}{rho_all:12.3f}{p_all:10.4f}{rho_nobg:14.3f}{p_nobg:10.4f}{n_all:5d}")
        summary_rows.append(
            {
                "stage": stage,
                "rho_all_organs": rho_all,
                "p_all_organs": p_all,
                "n_all_organs": n_all,
                "rho_foreground_only": rho_nobg,
                "p_foreground_only": p_nobg,
                "n_foreground_only": len(g_nobg),
            }
        )

    summary_path = out_dir / "organ_identity_reproducibility_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print()
    print(f"저장: {per_organ_path}")
    print(f"저장: {summary_path}")


if __name__ == "__main__":
    main()
