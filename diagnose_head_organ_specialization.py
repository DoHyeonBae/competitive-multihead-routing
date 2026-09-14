"""
Head x Organ 특화 분석 -- 각 head가 실제로 "어느 장기"에 더 강하게 반응하는지
정량적으로 확인.

flip-equivariance/cross-image correlation은 "이 head가 content-driven이냐
shortcut이냐"만 구분해줬지, "그래서 결국 어느 head가 어느 장기를 담당하는가"는
답해주지 않았다. 이 스크립트가 그 질문에 답한다.

방법: alpha_h(p) = s(p) * num_heads * r_h(p) (s: head 공유 relevance, r_h: head간
경쟁하는 순수 routing, sum_h r_h(p)=1). "이 장기를 어느 head가 담당하는가"를 보려면
s를 소거한 r_h만 봐야 하는데, 픽셀마다 먼저 나눠서 r_h를 복원한 뒤 평균내면 s(p)가
0에 가까운 픽셀에서 나눗셈이 수치적으로 불안정해진다(실측: alpha 채널합이 1e-18까지
떨어지는 픽셀 존재, epsilon이 결과를 완전히 왜곡함). 그래서 organ 영역 전체(여러
이미지에 걸쳐)에서 alpha를 먼저 다 더한 뒤 딱 한 번만 나눈다:
    share_h(organ) = sum_p[mask*alpha_h(p)] / sum_p[mask*sum_h'(alpha_h'(p))]
분모가 organ 전체·여러 이미지 픽셀의 합이라 0에 가까워질 일이 없고(epsilon 불필요),
수학적으로 s(p)로 가중평균한 routing과 정확히 같다(relevance가 낮다고 판단된 픽셀은
자동으로 덜 반영됨 -- "이 장기를 누가 담당하나" 질문에 맞는 가중치). 어떤 장기
마스크에도 안 걸치는 배경 영역도 대조군으로 같이 계산한다.

출력(stage마다):
  1) share 표: head x (organ + background), 각 셀 = 해당 head가 해당 장기를
     담당하는 비중(0~1). 같은 organ 열 안에서 4개 head 값은 항상 정확히 1로
     합쳐진다(별도 정규화 불필요, 구성상 보장됨) -- "이 장기를 몇 대 몇으로
     나눠 담당하는가"를 직접 보여줌.
  2) heatmap PNG(share 표 기준) -- 한눈에 어느 head가 어느 장기에 몰리는지 확인.

Run:
    python diagnose_head_organ_specialization.py \
        --checkpoint outputs_competitive_REAL/multihead_attention_unet_amos_H4_8organs_cmoe_bal0100_ent0000-0000-0000-0000_seed42.pt \
        --csv-dir data/content/data/amos_prepared \
        --num-samples 200 \
        --output-dir outputs_competitive_REAL/head_organ_analysis
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

from attention_unet_multitask import AttentionUNetResNet34MultiTask
from train_amos_fixed_heads import load_split, make_dataset

STAGE_NAMES = ["dec1(112x112)", "dec2(56x56)", "dec3(28x28)", "dec4(14x14)"]


def detect_gate_type(ckpt: dict) -> str:
    state_keys = ckpt["model_state_dict"].keys()
    if any("psi_heads" in k for k in state_keys):
        return "multi_split"
    elif any("psi_router" in k for k in state_keys):
        return "competitive_moe"
    return ckpt.get("gate_type", "competitive_moe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv-dir", type=str, default="/content/data/amos_prepared")
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
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

    csv_dir = Path(args.csv_dir)
    val_df = load_split(csv_dir, "val", task_names, args.min_organs_present)
    val_df = val_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    sample_df = val_df.iloc[: args.num_samples]
    val_dataset = make_dataset(sample_df, task_names, args.image_size, augment=False)
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if gate_type != "competitive_moe":
        raise ValueError(
            f"gate_type={gate_type!r} -- 이 분석은 CompetitiveMultiHeadAttentionGate 전용임. "
            f"그 gate만 s(shared relevance)와 r(순수 routing)이 구조적으로 분리돼 있어서 "
            f"model.dec{{n}}.gate.last_r을 직접 읽을 수 있음. ChannelSplitMultiHeadAttentionGate "
            f"(multi_split)는 애초에 이런 분리가 없어서 이 스크립트로 분석할 수 없음."
        )

    model = AttentionUNetResNet34MultiTask(
        num_tasks=num_organs, gate_type=gate_type, num_heads=num_heads, imagenet_pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    num_stages = 4
    stage_gates = [model.dec1.gate, model.dec2.gate, model.dec3.gate, model.dec4.gate]
    # r_h(p) = softmax(psi_router(combined))_h -- s(shared relevance)와는 완전히 별개의
    # conv+softmax에서 나오는 값이라 s가 뭐든 상관없이 항상 채널합=1로 잘 정의됨(나눗셈/epsilon
    # 전혀 불필요, 이전에 alpha에서 나눗셈으로 복원하려다 겪은 수치 불안정 문제가 원천적으로 없음).
    # 그래서 organ 마스크 안에서 그냥 픽셀 단위로 단순 평균 내면 그게 바로 "이 장기를 이 head가
    # 얼마나 담당하는가"의 편향 없는 답이 됨(s와 무관, organ마다 s 크기가 달라도 영향 없음).
    r_sums = [[[0.0] * (num_organs + 1) for _ in range(num_heads)] for _ in range(num_stages)]
    pixel_counts = [[0] * (num_organs + 1) for _ in range(num_stages)]  # 몇 개 픽셀이 기여했는지(표본 크기 확인용)
    organ_image_counts = [0] * num_organs  # 이 장기가 몇 장의 이미지에 등장했는지

    n_images = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)  # (B, num_organs, H, W), 0/1
            model(images)  # forward만 돌리면 각 gate.last_r이 채워짐(logits/alpha 자체는 여기선 안 씀)

            bg_mask = (masks.sum(dim=1, keepdim=True) == 0).float()  # (B,1,H,W)

            for organ_idx in range(num_organs):
                organ_image_counts[organ_idx] += int(
                    (masks[:, organ_idx, :, :].sum(dim=(1, 2)) > 0).sum().item()
                )

            for stage_idx, gate in enumerate(stage_gates):
                r_up = F.interpolate(
                    gate.last_r, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )  # (B, num_heads, H, W), 채널합 항상 ~1 (선형보간이라 softmax의 sum=1 성질이 그대로 유지됨)

                for organ_idx in range(num_organs):
                    organ_mask = masks[:, organ_idx : organ_idx + 1, :, :]  # (B,1,H,W)
                    for head in range(num_heads):
                        r_sums[stage_idx][head][organ_idx] += (
                            r_up[:, head : head + 1, :, :] * organ_mask
                        ).sum().item()
                    pixel_counts[stage_idx][organ_idx] += int(organ_mask.sum().item())

                for head in range(num_heads):
                    r_sums[stage_idx][head][num_organs] += (
                        r_up[:, head : head + 1, :, :] * bg_mask
                    ).sum().item()
                pixel_counts[stage_idx][num_organs] += int(bg_mask.sum().item())

            n_images += images.size(0)

    for organ_idx in range(num_organs):
        if organ_image_counts[organ_idx] == 0:
            print(f"  경고: '{task_names[organ_idx]}'가 샘플링된 {n_images}장 안에 한 번도 없음 "
                  f"-- 이 장기 열은 전부 nan으로 나옴. --num-samples를 늘려서 다시 실행 권장.")
        elif organ_image_counts[organ_idx] < 20:
            print(f"  참고: '{task_names[organ_idx]}'가 {organ_image_counts[organ_idx]}장에만 등장 "
                  f"(표본이 적어 추정이 불안정할 수 있음)")

    print(f"\n{n_images}장으로 head x organ 분석\n")

    col_names = list(task_names) + ["background"]
    csv_rows = []  # (stage, organ, head, share, n_pixels) -- Spearman correlation 등 후속 분석용
    for stage_idx in range(num_stages):
        # share_h(organ) = organ 마스크 안 픽셀들의 r_h(p) 단순 평균. r_h는 s와 무관하게
        # 항상 채널합=1로 잘 정의돼 있어서(softmax) 나눗셈/epsilon 없이 그냥 평균만 내면
        # 되고, 4개 head 합은 항상 정확히 1(별도 정규화 불필요).
        share = np.zeros((num_heads, num_organs + 1))
        for col in range(num_organs + 1):
            n_pixels = pixel_counts[stage_idx][col]
            for head in range(num_heads):
                share[head, col] = (
                    r_sums[stage_idx][head][col] / n_pixels if n_pixels > 0 else float("nan")
                )

        print(f"=== {STAGE_NAMES[stage_idx]} ===")
        header = "        " + "".join(f"{c[:10]:>12s}" for c in col_names)
        print(header)
        print("share_h(organ) -- 이 장기를 head별로 몇 대 몇으로 나눠 담당하는가(열 합=1):")
        for head in range(num_heads):
            row = "".join(f"{share[head, c]:12.4f}" for c in range(num_organs + 1))
            print(f"  H{head+1}: {row}")
        print("열 합 확인(1.0에 가까워야 정상):", ", ".join(f"{share[:, c].sum():.4f}" for c in range(num_organs + 1)))
        print(
            "기여 픽셀 수(표본 크기): "
            + ", ".join(f"{col_names[c]}={pixel_counts[stage_idx][c]:,}" for c in range(num_organs + 1))
        )

        dominant = share.argmax(axis=0)
        dom_str = ", ".join(f"{col_names[c]}=H{dominant[c]+1}({share[dominant[c],c]:.2f})" for c in range(num_organs + 1))
        print(f"  장기별 최다 담당 head: {dom_str}\n")

        for c in range(num_organs + 1):
            for head in range(num_heads):
                csv_rows.append({
                    "stage": STAGE_NAMES[stage_idx].split("(")[0],  # dec1~dec4
                    "organ": col_names[c],
                    "head": head + 1,
                    "share": share[head, c],
                    "n_pixels": pixel_counts[stage_idx][c],
                })

        # heatmap 저장
        fig, ax = plt.subplots(figsize=(1.4 * (num_organs + 1) + 2, 1.2 * num_heads + 2))
        im = ax.imshow(share, cmap="viridis", vmin=0, vmax=share.max())
        ax.set_xticks(range(num_organs + 1))
        ax.set_xticklabels(col_names, rotation=45, ha="right")
        ax.set_yticks(range(num_heads))
        ax.set_yticklabels([f"H{h+1}" for h in range(num_heads)])
        for h in range(num_heads):
            for c in range(num_organs + 1):
                ax.text(c, h, f"{share[h,c]:.2f}", ha="center", va="center",
                        color="white" if share[h, c] < share.max() * 0.6 else "black", fontsize=8)
        ax.set_title(f"{STAGE_NAMES[stage_idx]} -- head share per organ")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = output_dir / f"head_organ_{STAGE_NAMES[stage_idx].split('(')[0]}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  heatmap 저장: {out_path}")

    csv_path = output_dir / "head_organ_share.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"share 표 CSV 저장: {csv_path}")

    print(f"\n분석 완료: {n_images}장, {output_dir}에 stage별 heatmap 4장 저장됨")


if __name__ == "__main__":
    main()