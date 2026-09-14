"""
head_organ_share.csv(diagnose_head_organ_specialization.py 출력)를 재집계해서
stage x head별로 r_h^all / r_h^foreground / r_h^background를 비교.

배경: population-level head_balance_loss(L_bal)는 "전체 픽셀 평균" 이용률만
1/num_heads에 맞추라고 요구한다. AMOS 슬라이스는 배경 픽셀이 압도적으로 많으므로
(장기 픽셀은 슬라이스의 일부일 뿐), "전체 평균이 균형"이어도 그게 배경 픽셀
위주로 맞춰진 결과이고 실제 장기(foreground) 픽셀에서는 라우팅이 크게 쏠려
있을 수 있다 -- "일부 head가 balance loss quota를 배경 라우팅으로 채우고
foreground에는 거의 기여하지 않는다"는 가설.

diagnose_head_organ_specialization.py가 이미 stage x head x (organ+background)
share와 각 칸의 n_pixels를 CSV로 저장해두므로, 재학습/재추론 없이 그 CSV를
다시 집계하기만 하면 이 가설을 바로 검증할 수 있다:
    r_h^background = CSV의 organ="background" 행의 share를 그대로 사용
    r_h^foreground = organ!=background인 행들을 n_pixels로 가중평균
    r_h^all        = foreground/background를 각각의 총 n_pixels로 가중평균

주의(근사): 장기 마스크가 서로 겹치는 경계 픽셀이 있으면 foreground n_pixels
합이 실제 "belongs to >=1 organ" 픽셀 수보다 약간 커질 수 있음(중복 계산).
AMOS 개별 장기끼리 겹침은 일반적으로 미미하므로 근사로 충분하지만, 큰 효과
크기(예: H1 vs H4 배 이상 차이)를 보는 목적에는 문제 없음.

Run:
    python analyze_foreground_background_balance.py --csv outputs_.../head_organ_share.csv
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    stages = sorted(df["stage"].unique())
    heads = sorted(df["head"].unique())

    print(f"{'stage':8s} {'head':6s} {'r_all':>8s} {'r_fg':>8s} {'r_bg':>8s}")
    rows_out = []
    for stage in stages:
        sub = df[df["stage"] == stage]
        bg = sub[sub["organ"] == "background"]
        fg = sub[sub["organ"] != "background"]

        stage_fg_vals, stage_all_vals = [], []
        for head in heads:
            bg_row = bg[bg["head"] == head]
            fg_rows = fg[fg["head"] == head]
            n_bg = float(bg_row["n_pixels"].iloc[0])
            share_bg = float(bg_row["share"].iloc[0])
            n_fg = float(fg_rows["n_pixels"].sum())
            share_fg = float((fg_rows["share"] * fg_rows["n_pixels"]).sum() / n_fg) if n_fg > 0 else float("nan")
            n_all = n_bg + n_fg
            share_all = (share_bg * n_bg + share_fg * n_fg) / n_all if n_all > 0 else float("nan")

            print(f"{stage:8s} H{head:<5d} {share_all:8.4f} {share_fg:8.4f} {share_bg:8.4f}")
            rows_out.append({"stage": stage, "head": head, "r_all": share_all, "r_fg": share_fg, "r_bg": share_bg})
            stage_fg_vals.append(share_fg)
            stage_all_vals.append(share_all)

        ratio_all = max(stage_all_vals) / min(stage_all_vals) if min(stage_all_vals) > 0 else float("inf")
        ratio_fg = max(stage_fg_vals) / min(stage_fg_vals) if min(stage_fg_vals) > 0 else float("inf")
        print(f"  -> {stage} 불균형비(max/min): all={ratio_all:.2f}x, foreground={ratio_fg:.2f}x\n")

    out_path = args.csv.replace(".csv", "_fg_bg_summary.csv")
    pd.DataFrame(rows_out).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
