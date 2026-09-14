"""
analyze_per_case_difficulty.py가 저장한 per_image_dice.csv를 갖고, "H1이 어려워한
이미지"라는 정의가 H1 하나의 seed-specific 노이즈가 아니라 이미지 자체의 진짜
난이도(작은 장기, 애매한 경계 등)를 반영하는지 사전 점검.

방법: 모델들의 "원본"(차이 아님) per-image Dice끼리 Spearman 상관을 본다. 만약
서로 다른 architecture(H1 vs H2, H1 vs H4 등)의 원본 Dice가 강하게 양의 상관이면
(같은 이미지를 다 같이 어려워하거나 다 같이 쉬워함), "어렵다"는 게 이미지 고유의
성질이라는 뜻이라 analyze_per_case_difficulty.py의 "H1 기준 어려운 1/3" 분석이
H1 seed-specific 노이즈가 아니라 진짜 난이도를 잡고 있다는 근거가 된다. 상관이
약하면(예: rho<0.3), 그 분석 결과를 신뢰하기 전에 H1을 다른 seed로 하나 더 돌려서
"H1(seed43)도 H1(seed42)이 어려워한 이미지에서 비슷한 패턴을 보이는가"(순수
regression-to-mean 대조군)를 반드시 확인해야 한다.

추가로 n_organs_present(한 이미지에 실제로 존재하는 장기 수)와 H1 Dice의 상관도
같이 본다 -- 장기가 많이 겹친 이미지일수록 어려운 경향이 있다면(음의 상관), 이것도
"어렵다"가 임의 노이즈가 아니라 구조적 이유가 있다는 또 다른 방증이다.

Run:
    python verify_difficulty_is_real.py --csv outputs_percase\\per_image_dice.csv
"""

from __future__ import annotations

import argparse

import pandas as pd
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    wide = df.pivot(index="image_index", columns="model", values="dice")
    n_present = df.groupby("image_index")["n_organs_present"].first()

    models = list(wide.columns)
    print(f"모델: {models}\n")

    print("=== 원본(raw) per-image Dice끼리 Spearman 상관 (대각선 제외 전부 봐야 함) ===")
    print("해석: rho가 크게 낮은 쌍이 있으면 그 두 모델은 서로 다른 이미지를 어려워한다는 뜻 -- ")
    print("      전체적으로 다 높게(예 rho>0.5) 나와야 '난이도'가 이미지 고유의 성질이라는 근거가 강해짐.\n")
    header = "        " + "".join(f"{m:>12s}" for m in models)
    print(header)
    for m1 in models:
        row = []
        for m2 in models:
            if m1 == m2:
                row.append(f"{'--':>12s}")
            else:
                rho, _ = stats.spearmanr(wide[m1], wide[m2])
                row.append(f"{rho:12.3f}")
        print(f"{m1:8s}" + "".join(row))

    print("\n=== n_organs_present(한 이미지 내 장기 수) vs 각 모델 Dice 상관 ===")
    for m in models:
        rho, p = stats.spearmanr(n_present, wide[m])
        print(f"  {m}: rho={rho:.3f}, p={p:.2e}")


if __name__ == "__main__":
    main()
