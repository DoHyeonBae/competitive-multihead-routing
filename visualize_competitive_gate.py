"""
Competitive gate 체크포인트(competitive_moe) 대상, stage x head별 attention map을
실제 이미지 위에 시각화. 숫자(cosine/entropy/utilization)만으로는 "정말 의미 있는
anatomy를 보는가"를 알 수 없어서, val 이미지 몇 장을 골라 각 head가 실제로 어디를
보는지 눈으로 확인하기 위한 스크립트.

organ이 여러 개 동시에 존재하는 슬라이스를 우선으로 골라서(단일 장기만 있으면
head들이 갈릴 이유 자체가 적어서 재미없는 예시가 될 수 있음), 이미지당:
    - 원본 CT + GT 마스크(장기별 색으로 오버레이)
    - 4개 decoder stage(고해상도->저해상도) x num_heads개 head의 alpha heatmap
을 하나의 큰 그림으로 저장한다. 같은 stage 안에서는 공유 컬러스케일(그 stage의
전체 head 중 min~max)을 써서, "어느 head가 이 위치에서 상대적으로 더 켜져
있는지"가 heatmap 밝기로 바로 비교되게 한다.

Run (Colab):
    !pip install matplotlib -q  # 보통 이미 있음
    !python /content/drive/MyDrive/visualize_competitive_gate.py \
        --checkpoint /content/drive/MyDrive/amos22/outputs_competitive/multihead_attention_unet_amos_H4_8organs_competitive_bal0100_ent0050_seed42.pt \
        --csv-dir /content/data/amos_prepared \
        --output-dir /content/drive/MyDrive/amos22/outputs_competitive/viz \
        --num-samples 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from attention_unet_multitask import AttentionUNetResNet34MultiTask
from train_amos_fixed_heads import load_split, make_dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# 장기별 오버레이 색(간단히 matplotlib tab10 팔레트에서 가져옴)
ORGAN_COLORS = plt.get_cmap("tab10").colors


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) 정규화된 텐서 -> (H,W) 0~1 그레이스케일(3채널이 원래 다 같은 값이라 평균)."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.mean(axis=2)
    return np.clip(img, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv-dir", type=str, default="/content/data/amos_prepared")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    task_names = ckpt["task_names"]
    num_heads = ckpt["num_heads"]
    gate_type = ckpt.get("gate_type", "competitive_moe")
    print(f"checkpoint: {ckpt.get('model_name', args.checkpoint)}")
    print(f"task_names={task_names}, num_heads={num_heads}, gate_type={gate_type}")

    csv_dir = Path(args.csv_dir)
    val_df = load_split(csv_dir, "val", task_names, args.min_organs_present)

    # organ이 여러 개 동시에 존재하는 슬라이스 우선(재미없는 단일-장기 예시 피하기)
    content_cols = [f"content_{t}" for t in task_names]
    val_df = val_df.copy()
    val_df["_n_organs"] = val_df[content_cols].sum(axis=1)
    val_df = val_df.sort_values("_n_organs", ascending=False).reset_index(drop=True)
    sample_df = val_df.iloc[: args.num_samples * 5 : 5].reset_index(drop=True)  # 다양성 위해 듬성듬성
    sample_df = sample_df.iloc[: args.num_samples]

    val_dataset = make_dataset(sample_df, task_names, args.image_size, augment=False)

    model = AttentionUNetResNet34MultiTask(
        num_tasks=len(task_names), gate_type=gate_type, num_heads=num_heads, imagenet_pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    stage_names = ["dec1(112x112)", "dec2(56x56)", "dec3(28x28)", "dec4(14x14)"]

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            image, mask = val_dataset[idx]
            image_b = image.unsqueeze(0).to(device)
            _logits, alphas = model(image_b)

            img_disp = denormalize_image(image)
            mask_np = mask.cpu().numpy()  # (num_tasks, H, W)

            n_stages = len(alphas)
            fig, axes = plt.subplots(
                n_stages + 1, num_heads + 1,
                figsize=(2.4 * (num_heads + 1), 2.4 * (n_stages + 1)),
            )

            # 첫 행: 원본 + GT 오버레이(첫 칸만 채우고 나머지는 비움)
            axes[0, 0].imshow(img_disp, cmap="gray")
            axes[0, 0].set_title("input CT", fontsize=9)
            axes[0, 0].axis("off")

            overlay = np.stack([img_disp] * 3, axis=-1)
            for t, name in enumerate(task_names):
                color = np.array(ORGAN_COLORS[t % 10])
                m = mask_np[t] > 0.5
                overlay[m] = 0.5 * overlay[m] + 0.5 * color
            axes[0, 1].imshow(overlay)
            axes[0, 1].set_title("GT organs (색상=장기)", fontsize=9)
            axes[0, 1].axis("off")
            for c in range(2, num_heads + 1):
                axes[0, c].axis("off")

            H, W = img_disp.shape
            for s, (alpha, sname) in enumerate(zip(alphas, stage_names)):
                a = alpha[0].cpu().numpy()  # (num_heads, h, w)
                a_up = np.stack(
                    [
                        np.array(
                            torch.nn.functional.interpolate(
                                torch.from_numpy(a[h : h + 1]).unsqueeze(0),
                                size=(H, W), mode="bilinear", align_corners=False,
                            )[0, 0]
                        )
                        for h in range(num_heads)
                    ]
                )
                vmin, vmax = a_up.min(), a_up.max()

                axes[s + 1, 0].imshow(img_disp, cmap="gray")
                axes[s + 1, 0].set_ylabel(sname, fontsize=9)
                axes[s + 1, 0].set_xticks([])
                axes[s + 1, 0].set_yticks([])

                for h in range(num_heads):
                    ax = axes[s + 1, h + 1]
                    ax.imshow(img_disp, cmap="gray")
                    ax.imshow(a_up[h], cmap="inferno", alpha=0.55, vmin=vmin, vmax=vmax)
                    if s == 0:
                        ax.set_title(f"H{h+1}", fontsize=9)
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.tight_layout()
            out_path = output_dir / f"sample_{idx:02d}.png"
            plt.savefig(out_path, dpi=110)
            plt.close(fig)
            print(f"저장: {out_path}")

    print(f"\n총 {len(val_dataset)}장 저장 완료: {output_dir}")


if __name__ == "__main__":
    main()
