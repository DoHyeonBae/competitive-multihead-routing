"""
Seed43 regression-to-the-mean control용 병합 스크립트.

목적: H2(seed43) vs H1(seed43) paired 비교를 할 때, "외부 난이도"를 여전히
원래 논문에 쓴 것과 똑같은 4개 모델(H4/Stage-Adaptive/Head-Projection 2종, 전부
seed42)의 평균으로 정의하고 싶다. 그런데 원본 per_image_dice.csv에는 H1(seed42),
H2(seed42)도 같이 들어있어서, 그대로 병합하면 analyze_difficulty_rigorous.py가
"others"를 계산할 때 이 seed42 H1/H2까지 섞어서 6개 모델 평균을 쓰게 된다 -- 이러면
원래 분석의 "4개 모델 평균" 방법론과 안 맞는다(seed42 H1은 같은 아키텍처라 완전히
무관하다고 보기도 애매함).

그래서 이 스크립트는:
  1) 원본 CSV에서 H1(seed42), H2(seed42) 두 컬럼(행)을 제거하고
  2) 새로 뽑은 H1_seed43, H2_seed43을 추가한다
결과 CSV는 [H4, Stage-Adaptive, Head-Projection(2), Head-Projection+SA, H1_seed43,
H2_seed43] 6개 모델만 남는다. 이후 analyze_difficulty_rigorous.py를
--baseline-label H1_seed43으로 돌리면, H2_seed43과의 비교에서 "others"가 정확히
원래 논문과 동일한 4개 모델이 된다.

Run:
    python merge_seed43_pair.py `
        --original outputs_percase\per_image_dice.csv `
        --seed43 outputs_percase_seed43only\per_image_dice.csv `
        --drop-labels H1,H2 `
        --keep-new-labels H1_seed43,H2_seed43 `
        --output outputs_percase_seed43only\per_image_dice_merged.csv
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--seed43", type=str, required=True)
    parser.add_argument("--drop-labels", type=str, required=True, help="원본에서 제거할 모델 라벨, 쉼표 구분(예: H1,H2)")
    parser.add_argument("--keep-new-labels", type=str, required=True, help="seed43 CSV에서 가져올 모델 라벨, 쉼표 구분")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    drop_labels = [s.strip() for s in args.drop_labels.split(",")]
    keep_new_labels = [s.strip() for s in args.keep_new_labels.split(",")]

    orig = pd.read_csv(args.original)
    s43 = pd.read_csv(args.seed43)

    missing_drop = [d for d in drop_labels if d not in orig["model"].unique()]
    if missing_drop:
        raise ValueError(f"원본에 없는 모델을 지우려 함: {missing_drop} (있는 모델: {sorted(orig['model'].unique())})")

    orig_kept = orig[~orig["model"].isin(drop_labels)].copy()

    missing_new = [n for n in keep_new_labels if n not in s43["model"].unique()]
    if missing_new:
        raise ValueError(f"seed43 CSV에 없는 모델: {missing_new} (있는 모델: {sorted(s43['model'].unique())})")
    s43_kept = s43[s43["model"].isin(keep_new_labels)].copy()

    n_orig_images = orig["image_index"].nunique()
    for label in keep_new_labels:
        n = len(s43_kept[s43_kept["model"] == label])
        if n != n_orig_images:
            raise ValueError(
                f"{label}의 이미지 수({n})가 원본({n_orig_images})과 다름 -- "
                "analyze_per_case_difficulty.py를 원본과 동일한 --csv-dir/--task-names/"
                "--min-organs-present/--num-samples로 돌렸는지 확인할 것."
            )

    combined = pd.concat([orig_kept, s43_kept], ignore_index=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"병합 완료: {args.output}")
    print(f"남은 모델: {sorted(combined['model'].unique())}")
    print(f"모델당 이미지 수: {combined.groupby('model').size().to_dict()}")


if __name__ == "__main__":
    main()
