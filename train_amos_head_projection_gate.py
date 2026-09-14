"""
H2 + Head-Specific Projection 실험 (H1/H2/H4/stage-adaptive[2,2,2,1] 다음 실험).

CompetitiveMultiHeadAttentionGate는 head끼리 theta_x/phi_g/psi_shared를 공유하고,
head마다 다른 건 raw skip feature를 채널 축으로 잘라 쓰는 것뿐이었다(x_splits =
x.chunk(num_heads)). 이 실험은 그 대신 head마다 완전히 독립된 1x1 conv 투영
P_h(x)와 독립된 attention 서브네트워크 A_h(P_h(x), g)를 준다 -- attention을 계산하기
전부터 feature representation 자체를 head별로 분리해서, representation-level
specialization이 raw Dice 개선으로 이어지는지 확인하는 것.

    F_h = P_h(x)
    A_h = sigmoid(psi_h(relu(theta_x_h(F_h) + phi_g_h(g))))
    Z_h = A_h * F_h
    r_h = softmax_h(psi_router(relu(theta_x_shared(x) + phi_g_shared(g))))  -- 라우팅은
          기존과 동일한 방식으로 경쟁시켜서 결정(balance/entropy loss 그대로 호환)
    Z   = fusion(concat_h[r_h * Z_h])

주의 -- 이건 검증된 결론에서 나온 실험이 아니라 새로운 가설 테스트다. H1/H2/H4/
stage-adaptive 결과는 "필요한 head 수는 nominal보다 적다"는 걸 보여줬을 뿐, "왜 raw
Dice가 안 오르는가"에 대한 직접적 증거는 아니었다. 그리고 이미 CompetitiveMultiHead
(shared feature) 구조로도 causal ablation에서 depth-dependent specialization이 실재
한다는 게 확인됐기 때문(rho~=-0.78), "shared feature라서 specialization이 안 된다"는
전제 자체도 완전히 들어맞지는 않는다 -- 그래도 "더 독립적인 representation을 주면 raw
Dice가 오르는가"는 아직 테스트 안 해본 질문이라 시도할 가치는 있다.

파라미터 비교(12-organ, H2 기준): competitive_moe 24,672,216개 vs head_proj_moe
25,076,700개 (+1.6%, 큰 증가는 아님).

--num-heads로 dec1~dec4 전부 같은 head 수(H2 재현이면 2), 또는 --heads-per-stage로
stage-adaptive 실험(train_amos_stage_adaptive_heads.py)에서 확인된 배분(예: 2,2,2,1)을
head_proj_moe 구조에도 그대로 적용 가능(dec1,dec2,dec3,dec4 순서, 둘 중 하나만 지정).
diagnose_*.py 진단 스크립트들은 아직 이 gate_type을 지원하지 않음(전부 competitive_moe
전용) -- 우선 raw Dice로 1차 스크리닝하고, 유의미하면 그때 진단 스크립트 확장을 별도 진행할 것.

Run (스모크 테스트, 1 epoch):
    python train_amos_head_projection_gate.py \
        --csv-dir data\\content\\data\\amos_prepared \
        --output-dir outputs_headproj_smoketest \
        --task-names liver,spleen,right_kidney,left_kidney,stomach,pancreas,gall_bladder,right_adrenal_gland,left_adrenal_gland,duodenum,esophagus,bladder \
        --num-heads 2 --lambda-bal 0.1 --epochs 1 --seed 42

Run (본 실험, 10 epoch -- H2 스크리닝과 동일 조건):
    python train_amos_head_projection_gate.py \
        --csv-dir data\\content\\data\\amos_prepared \
        --output-dir outputs_competitive_12organ \
        --task-names liver,spleen,right_kidney,left_kidney,stomach,pancreas,gall_bladder,right_adrenal_gland,left_adrenal_gland,duodenum,esophagus,bladder \
        --num-heads 2 --lambda-bal 0.1 --epochs 10 --seed 42
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from attention_unet_multitask import (
    AttentionUNetResNet34MultiTask,
    estimate_pixel_pos_weights,
    seed_everything,
    set_pos_weights,
)
from train_amos_competitive_gate import train_competitive
from train_amos_fixed_heads import load_split, make_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--task-names", type=str, required=True)
    parser.add_argument(
        "--num-heads", type=int, default=None,
        help="dec1~dec4 전부 동일한 head 수. H2 재현이면 2. --heads-per-stage와 동시에 줄 수 없음.",
    )
    parser.add_argument(
        "--heads-per-stage", type=str, default=None,
        help="콤마 4개, dec1,dec2,dec3,dec4 순서(예: '2,2,2,1'). stage-adaptive 실험(train_amos_"
        "stage_adaptive_heads.py)에서 확인된 head 배분을 head_proj_moe 구조에도 그대로 적용해볼 때 "
        "사용. --num-heads와 동시에 줄 수 없음(둘 중 하나만).",
    )
    parser.add_argument("--lambda-bal", type=float, required=True)
    parser.add_argument("--lambda-ent", type=float, default=0.0)
    parser.add_argument("--lambda-ent-per-stage", type=str, default=None)
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cache-dataset", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-imagenet-pretrained", action="store_true")
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="기존 체크포인트(.pt) 경로. epoch 수를 늘려서(예: 10->20) 이어서 학습할 때 사용. "
        "optimizer/scheduler 상태까지 복원(train_amos_competitive_gate.train_competitive와 동일 로직).",
    )
    args = parser.parse_args()

    if (args.heads_per_stage is None) == (args.num_heads is None):
        raise ValueError(
            "--heads-per-stage와 --num-heads 중 정확히 하나만 줘야 함 "
            f"(heads_per_stage={args.heads_per_stage!r}, num_heads={args.num_heads!r})"
        )
    if args.heads_per_stage is not None:
        heads_per_stage = [int(v) for v in args.heads_per_stage.split(",")]
        if len(heads_per_stage) != 4:
            raise ValueError(f"--heads-per-stage는 dec1,dec2,dec3,dec4 4개 값이어야 함(받은 값: {heads_per_stage})")
        num_heads_for_model: int | list[int] = heads_per_stage
        heads_tag = "-".join(str(h) for h in heads_per_stage)
    else:
        num_heads_for_model = args.num_heads
        heads_tag = str(args.num_heads)

    seed_everything(args.seed)

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    task_names = [t.strip() for t in args.task_names.split(",")]
    print(f"task_names ({len(task_names)}개): {task_names}")

    train_df = load_split(csv_dir, "train", task_names, args.min_organs_present)
    val_df = load_split(csv_dir, "val", task_names, args.min_organs_present)

    train_dataset = make_dataset(
        train_df, task_names, args.image_size, augment=True, cache_in_memory=args.cache_dataset,
    )
    val_dataset = make_dataset(
        val_df, task_names, args.image_size, augment=False, cache_in_memory=args.cache_dataset,
    )
    if args.cache_dataset:
        est_gb = (1 + len(task_names)) * args.image_size ** 2 * (len(train_dataset) + len(val_dataset)) / 1e9
        print(f"--cache-dataset 켜짐 -- 대략 {est_gb:.1f}GB RAM 예상(전부 캐싱됐을 때 최대치)")

    ratios = ", ".join(f"{n} 양성 {train_dataset.positive_ratio(n):.3f}" for n in task_names)
    print(f"이미지 수 -> train: {len(train_dataset)} ({ratios}), val: {len(val_dataset)}")

    pos_weights = estimate_pixel_pos_weights(train_dataset)
    set_pos_weights(pos_weights)
    print("픽셀 단위 고정 pos_weight -> " + ", ".join(f"{n}: {w:.2f}" for n, w in zip(task_names, pos_weights)))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )

    if args.lambda_ent_per_stage is not None:
        lambda_ent_per_stage = [float(v) for v in args.lambda_ent_per_stage.split(",")]
        if len(lambda_ent_per_stage) != 4:
            raise ValueError(f"--lambda-ent-per-stage는 dec1~dec4 4개 값이어야 함(받은 값: {lambda_ent_per_stage})")
    else:
        lambda_ent_per_stage = [args.lambda_ent] * 4
    print(f"lambda_ent_per_stage(dec1..dec4) = {lambda_ent_per_stage}")

    imagenet_pretrained = not args.no_imagenet_pretrained
    bal_tag = f"bal{args.lambda_bal:.3f}".replace(".", "")
    ent_tag = "ent" + "-".join(f"{v:.3f}".replace(".", "") for v in lambda_ent_per_stage)

    # 기존 H{n}(competitive_moe)/Hadapt(stage-adaptive competitive_moe) 이름과 겹치지 않도록
    # "Hproj" 접두사 사용. --heads-per-stage로 줬으면 "Hproj2-2-2-1"처럼 배분이 그대로 이름에 남음.
    name = (
        f"multihead_attention_unet_amos_Hproj{heads_tag}_{len(task_names)}organs_"
        f"headproj_{bal_tag}_{ent_tag}_seed{args.seed}"
    )

    model = AttentionUNetResNet34MultiTask(
        num_tasks=len(task_names), gate_type="head_proj_moe", num_heads=num_heads_for_model,
        imagenet_pretrained=imagenet_pretrained,
    )
    print(f"[확인] model.heads_per_stage(dec1,dec2,dec3,dec4) = {model.heads_per_stage}")

    result = train_competitive(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, learning_rate=args.lr, output_dir=output_dir,
        task_names=task_names, num_heads=num_heads_for_model, lambda_bal=args.lambda_bal,
        lambda_ent_per_stage=lambda_ent_per_stage, gate_type="head_proj_moe",
        resume_from=args.resume_from,
    )
    result["seed"] = args.seed

    csv_path = output_dir / "comparison_amos_headproj.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print("\nComparison (AMOS Head-Projection Gate)")
    print(result)
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()
