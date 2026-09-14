"""
Horizontal-flip equivariance test -- "dec4가 진짜 anatomy를 보는가, 아니면 화면상
고정 위치를 고집하는가"를 직접 확정하는 실험. 재학습 필요 없음, 기존 체크포인트로
바로 확인.

논리:
    원본 이미지 x를 넣었을 때의 alpha(x)와, 좌우 flip한 이미지 flip(x)를 넣었을 때의
    alpha(flip(x))를 다시 좌우로 flip-back한 것을 비교한다.

    - content-driven(해부학적 위치를 따라감)이라면: 이미지를 뒤집으면 장기도 화면상
      반대쪽으로 이동하므로, head가 "그 장기"를 계속 따라가서 attention도 같이
      뒤집혀야 한다 -> flip-back한 attention이 원본 attention과 잘 맞아야 한다
      (correlation 높음).
    - 화면상 고정 위치 shortcut이라면: 이미지를 뒤집어도 head는 여전히 "화면의
      오른쪽"처럼 고정된 위치를 켠다 -> flip-back하면 원본과 반대쪽이 되어버려서
      correlation이 낮아진다(혹은 음의 상관까지 갈 수 있음).

Run (Colab):
    !python /content/drive/MyDrive/diagnose_flip_equivariance.py \
        --checkpoint /content/drive/MyDrive/amos22/outputs_competitive/multihead_attention_unet_amos_H4_8organs_competitive_bal0100_ent0050_seed42.pt \
        --csv-dir /content/data/amos_prepared \
        --num-samples 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from attention_unet_multitask import AttentionUNetResNet34MultiTask
from train_amos_fixed_heads import load_split, make_dataset


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((a @ b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--csv-dir", type=str, default="/content/data/amos_prepared")
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-samples", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    task_names = ckpt["task_names"]
    num_heads = ckpt["num_heads"]
    gate_type = ckpt.get("gate_type", "competitive_moe")
    print(f"checkpoint: {ckpt.get('model_name', args.checkpoint)}")

    csv_dir = Path(args.csv_dir)
    val_df = load_split(csv_dir, "val", task_names, args.min_organs_present)
    val_df = val_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    sample_df = val_df.iloc[: args.num_samples]
    val_dataset = make_dataset(sample_df, task_names, args.image_size, augment=False)

    model = AttentionUNetResNet34MultiTask(
        num_tasks=len(task_names), gate_type=gate_type, num_heads=num_heads, imagenet_pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    stage_names = ["dec1(112x112)", "dec2(56x56)", "dec3(28x28)", "dec4(14x14)"]
    corr_sums = None  # [stage][head] 누적
    n_images = 0

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            image, _mask = val_dataset[idx]
            image_b = image.unsqueeze(0).to(device)
            image_flip = torch.flip(image_b, dims=[-1])  # 좌우 flip

            _logits_o, alphas_o = model(image_b)
            _logits_f, alphas_f = model(image_flip)

            if corr_sums is None:
                corr_sums = [[0.0] * num_heads for _ in alphas_o]

            for s, (a_o, a_f) in enumerate(zip(alphas_o, alphas_f)):
                a_fb = torch.flip(a_f, dims=[-1])  # flip-back
                for h in range(num_heads):
                    orig_map = a_o[0, h].cpu().numpy().flatten()
                    flipback_map = a_fb[0, h].cpu().numpy().flatten()
                    corr_sums[s][h] += pearson_corr(orig_map, flipback_map)
            n_images += 1

    print(f"\n{n_images}장으로 flip-equivariance 진단\n")
    print("orig vs flip-back correlation: 높을수록(>0.7) content-driven(장기를 따라감),")
    print("낮을수록(<0.3, 특히 음수면 확실) 화면상 고정 위치 shortcut일 가능성 높음\n")

    for s, sname in enumerate(stage_names):
        print(f"[{sname}]")
        for h in range(num_heads):
            avg = corr_sums[s][h] / n_images
            print(f"  H{h+1}: orig vs flip-back correlation = {avg:.4f}")
        print()


if __name__ == "__main__":
    main()
