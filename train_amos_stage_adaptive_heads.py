"""
Stage-Adaptive Competitive Head Allocation 실험.

배경: H4(12-organ, 4-head competitive_moe) causal ablation(diagnose_ablation_uniform_and_head_removal.py,
전체 val 3521장)에서 stage별로 실제 causal하게 기여하는 head 수가 nominal head 수(4)보다
훨씬 적다는 게 확인됐다 -- Part B(near-zero head 동시제거) 결과 기준:
    dec4: H1만 활성, H2/H3/H4는 동시제거해도 interaction~0 (진짜 redundant) -> effective~1
    dec3: H1,H4 활성, H2/H3 redundant                                  -> effective~2
    dec2: H2(+일부 H1) 활성, H3/H4 redundant                            -> effective~2
    dec1: H2(+일부 H1) 활성, H3/H4 redundant                            -> effective~2

이 스크립트는 "그럼 처음부터 그 effective head 수만큼만 nominal capacity를 주고
재학습해도 H2/H4와 비슷한 성능이 나오는가"를 검증하기 위한 것 -- 즉 사후적으로
학습된 H4에서 head를 제거하는 게 아니라(그건 이미 diagnose_ablation로 확인함),
"처음부터 작게 태어난 모델도 똑같이 잘 학습되는가"를 새로 확인하는 것.

이건 새로운 loss나 forcing mechanism이 아니다 -- 기존 competitive_moe 게이트/
head_balance_loss/head_entropy_loss/학습 루프를 전부 그대로 재사용하고(전부
train_amos_competitive_gate.py에서 그대로 import), 유일하게 바뀌는 건
"dec1~dec4에 몇 개의 head를 만들 것인가"라는 nominal architecture capacity뿐이다.
동일 조건(encoder/decoder/seg loss/lambda_bal/dataset/augmentation/seed/epoch 수)은
전부 H2/H4 스크리닝과 동일하게 맞춘다.

첫 후보로 --heads-per-stage 2,2,2,1 을 추천 (dec1,dec2,dec3,dec4 순서 -- 기존
--lambda-ent-per-stage 옵션과 동일한 순서 컨벤션). dec1~3는 H2 수준(2)을 유지하고
dec4만 1로 줄인 것 -- H4 ablation에서 dec4가 가장 명확하게(H1 하나만 남기고
interaction~0으로) redundancy가 확인됐고, dec1~3는 H2(2-head) 자체가 이미
효과가 검증된 baseline이라 안전한 하한이기 때문. dec1/dec2를 1로 더 줄이는 건
H4 ablation에서 두 번째 head(H1 or H2)의 기여가 완전히 0은 아니었어서(-0.01~-0.13
수준으로 존재) 지금 데이터로 정당화하기엔 이름.

--num-heads 를 그대로 주면(기존과 동일하게 4개 stage 전부 같은 head 수)
기존 train_amos_competitive_gate.py와 100% 동일하게 동작한다 -- 리팩터링이
기존 H1/H2/H4 결과를 깨뜨리지 않았는지 확인하는 회귀 테스트 용도로 먼저
--num-heads 4 --epochs 1 정도로 한 번 돌려서 기존 H4 epoch01 로그와 비교해보는
걸 추천한다(smoke test).

Run (스모크 테스트, 1 epoch 먼저):
    python train_amos_stage_adaptive_heads.py \
        --csv-dir data\\content\\data\\amos_prepared \
        --output-dir outputs_stage_adaptive_smoketest \
        --task-names liver,spleen,right_kidney,left_kidney,stomach,pancreas,gall_bladder,right_adrenal_gland,left_adrenal_gland,duodenum,esophagus,bladder \
        --heads-per-stage 2,2,2,1 --lambda-bal 0.1 --epochs 1 --seed 42

Run (본 실험, 10 epoch -- H2/H4 스크리닝과 동일 조건):
    python train_amos_stage_adaptive_heads.py \
        --csv-dir data\\content\\data\\amos_prepared \
        --output-dir outputs_competitive_12organ \
        --task-names liver,spleen,right_kidney,left_kidney,stomach,pancreas,gall_bladder,right_adrenal_gland,left_adrenal_gland,duodenum,esophagus,bladder \
        --heads-per-stage 2,2,2,1 --lambda-bal 0.1 --epochs 10 --seed 42
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
        "--heads-per-stage", type=str, default=None,
        help="콤마 4개, dec1,dec2,dec3,dec4 순서(예: '2,2,2,1'). stage마다 다른 nominal "
        "head 수를 줄 때 사용. --num-heads와 동시에 줄 수 없음(둘 중 하나만).",
    )
    parser.add_argument(
        "--num-heads", type=int, default=None,
        help="기존 train_amos_competitive_gate.py와 동일하게 dec1~dec4 전부 같은 head 수. "
        "리팩터링 회귀 테스트(기존 H1/H2/H4와 동일 결과 재현 확인)용. --heads-per-stage와 "
        "동시에 줄 수 없음(둘 중 하나만).",
    )
    parser.add_argument("--gate-type", type=str, default="competitive_moe", choices=["competitive_moe", "multi_split"])
    parser.add_argument("--lambda-bal", type=float, required=True)
    parser.add_argument("--lambda-ent", type=float, default=0.0)
    parser.add_argument("--lambda-ent-per-stage", type=str, default=None)
    parser.add_argument("--min-organs-present", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--cache-dataset", action="store_true",
        help="attention_unet_multitask.MultiTaskCsvDataset의 in-memory 캐싱. HDD I/O 병목일 때 켜기 "
        "(--workers 0과 같이 쓸 것). H2/H4 스크리닝과 조건을 맞추려면 그때와 동일하게 켰는지/껐는지 확인.",
    )
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
            raise ValueError(
                f"--heads-per-stage는 dec1,dec2,dec3,dec4 4개 값이어야 함(받은 값: {heads_per_stage})"
            )
        num_heads_for_model: int | list[int] = heads_per_stage
        heads_tag = "-".join(str(h) for h in heads_per_stage)
    else:
        num_heads_for_model = args.num_heads
        heads_tag = str(args.num_heads)  # --num-heads로 회귀 테스트할 때는 기존과 동일한 태그

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
            raise ValueError(
                f"--lambda-ent-per-stage는 dec1~dec4 4개 값이어야 함(받은 값: {lambda_ent_per_stage})"
            )
    else:
        lambda_ent_per_stage = [args.lambda_ent] * 4
    print(f"lambda_ent_per_stage(dec1..dec4) = {lambda_ent_per_stage}")

    imagenet_pretrained = not args.no_imagenet_pretrained
    bal_tag = f"bal{args.lambda_bal:.3f}".replace(".", "")
    ent_tag = "ent" + "-".join(f"{v:.3f}".replace(".", "") for v in lambda_ent_per_stage)
    gate_tag = "cmoe" if args.gate_type == "competitive_moe" else "chsplit"

    # 기존 H1/H2/H4/H8과 이름이 절대 겹치지 않도록 "Hadapt" 접두사를 붙인다(같은
    # --output-dir을 실수로 재사용해도 기존 체크포인트를 덮어쓰지 않음). --num-heads로
    # 회귀 테스트할 때만 예외적으로 기존과 동일한 "H{n}" 이름을 써서 진짜로 기존 결과와
    # 1:1 비교(같은 파일 이름 = 같은 seed/조건이면 같은 숫자가 나와야 함)할 수 있게 한다.
    head_name_part = f"H{heads_tag}" if args.num_heads is not None else f"Hadapt{heads_tag}"
    name = (
        f"multihead_attention_unet_amos_{head_name_part}_{len(task_names)}organs_"
        f"{gate_tag}_{bal_tag}_{ent_tag}_seed{args.seed}"
    )

    model = AttentionUNetResNet34MultiTask(
        num_tasks=len(task_names), gate_type=args.gate_type, num_heads=num_heads_for_model,
        imagenet_pretrained=imagenet_pretrained,
    )
    # 실제로 dec1..dec4에 배정된 head 수를 한 번 더 명시적으로 확인 출력 -- 순서가
    # 뒤바뀌는 실수(예: dec4에 2를 주고 dec1에 1을 주는 등)를 학습 시작 전에 바로 알아채기 위함.
    print(
        f"[확인] model.heads_per_stage(dec1,dec2,dec3,dec4) = {model.heads_per_stage} "
        f"(요청한 --heads-per-stage/--num-heads와 순서가 일치하는지 확인할 것)"
    )

    result = train_competitive(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, learning_rate=args.lr, output_dir=output_dir,
        task_names=task_names, num_heads=num_heads_for_model, lambda_bal=args.lambda_bal,
        lambda_ent_per_stage=lambda_ent_per_stage, gate_type=args.gate_type,
        resume_from=args.resume_from,
    )
    result["seed"] = args.seed
    result["heads_per_stage"] = model.heads_per_stage

    csv_path = output_dir / "comparison_amos_stage_adaptive.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print("\nComparison (AMOS Stage-Adaptive Heads)")
    print(result)
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()
