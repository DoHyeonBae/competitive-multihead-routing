"""
"dec4 패턴이 6장 다 똑같아 보인 게, 5 epoch이라 아직 안 갈라진 것뿐 아니냐"는
질문에 대한 직접적인, 재학습 없는 답. 지금 있는 체크포인트 그대로 val 이미지를
훨씬 많이(예: 40장) 통과시켜서, 같은 head가 서로 다른 이미지에서 내는 alpha map이
서로 얼마나 닮았는지(=cross-image correlation)를 stage별로 계산한다.

논리:
    진짜 content-dependent라면(이미지 내용에 따라 라우팅이 달라진다면) 서로 다른
    환자/슬라이스의 alpha map은 서로 별로 안 닮아야 한다(상관계수 낮음).
    반대로 "화면상 고정된 위치를 항상 켠다"는 shortcut이라면, 어떤 이미지를 넣든
    같은 head는 거의 똑같은 모양을 내야 한다(상관계수가 1에 가까움).

이건 "5 epoch이라 덜 갈라진 것"과는 다른 문제다 -- 언더피팅이면 patterns이 다소
흐릿하고 약하더라도 이미지마다 달라야 하는데, 지금 의심되는 건 애초에 이미지
내용을 아예 안 본다는 것(고정 위치)이라서, 더 학습한다고 저절로 고쳐질 문제가
아닐 수 있다. 이 스크립트로 재학습 없이 바로 확인 가능.

Run (Colab):
    !python /content/drive/MyDrive/diagnose_content_vs_position.py \
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
    # 골고루 퍼진 샘플(앞에서부터 듬성듬성) -- 이전 시각화(organ 많은 순 정렬)와
    # 겹치지 않는 다양한 환자/슬라이스를 보기 위해 셔플 후 추출
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
    # stage별, head별로 모든 이미지의 alpha map(flatten)을 모아둠
    collected = None  # collected[stage][head] = list of flattened np arrays

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            image, _mask = val_dataset[idx]
            image_b = image.unsqueeze(0).to(device)
            _logits, alphas = model(image_b)

            if collected is None:
                collected = [[[] for _ in range(num_heads)] for _ in alphas]

            for s, alpha in enumerate(alphas):
                a = alpha[0].cpu().numpy()  # (num_heads, h, w)
                for h in range(num_heads):
                    collected[s][h].append(a[h].flatten())

    print(f"\n{len(val_dataset)}장으로 진단\n")
    print("cross-image correlation: 같은 head가 서로 다른 이미지에서 내는 alpha map끼리 상관계수")
    print("  1.0에 가까움 = 이미지 내용과 무관하게 거의 똑같은 패턴(위치 고정/shortcut 의심)")
    print("  낮을수록(0.3~0.6대) = 이미지마다 패턴이 달라짐(content-dependent 가능성)\n")

    for s, sname in enumerate(stage_names):
        print(f"[{sname}]")
        for h in range(num_heads):
            maps = np.stack(collected[s][h])  # (N, pixels)
            maps_c = maps - maps.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(maps_c, axis=1, keepdims=True) + 1e-8
            maps_n = maps_c / norms
            corr = maps_n @ maps_n.T  # (N,N) pearson correlation
            n = corr.shape[0]
            iu = np.triu_indices(n, k=1)
            avg_corr = corr[iu].mean()
            print(f"  H{h+1}: 평균 cross-image correlation = {avg_corr:.4f}")
        print()


if __name__ == "__main__":
    main()
