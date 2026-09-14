"""
AMOS22 Fixed-Head-Count baseline (H1/H2/H4/H8) -- SCHR/gate/routing/penalty
전부 안 씀. "다장기 CT segmentation에서 head 수를 늘리면 실제로 성능이
오르는가?"라는 첫 번째 질문에만 답하기 위한, 가장 단순한 실험.

prepare_amos_dataset.py가 15개 장기를 전부 뽑아서 하나의 CSV에 mask_<장기>/
content_<장기> 컬럼으로 다 넣어뒀기 때문에, 어떤 장기 조합으로 학습할지는
이 스크립트의 --task-names에서 그때그때 고른다(전처리를 장기 조합마다 다시
돌릴 필요 없음). 예:
    --task-names liver,spleen,pancreas,duodenum
    --task-names liver,spleen,kidney,pancreas,gallbladder,duodenum,left_adrenal_gland,right_adrenal_gland

--task-names로 고른 장기들의 content_<장기> 컬럼이 전부 True인 행(=고른 장기가
전부 존재하는 슬라이스)만 걸러서 쓴다 -- disc/cup이 "disc 양성 1.000, cup 양성
1.000"이었던 것과 동일한 설계(항상 전부 존재하는 슬라이스만 학습에 사용).

H1/H2/H4/H8 비교 시 지켜야 하는 것: 같은 --seed, 같은 --task-names, 같은
--epochs, 같은 학습률/옵티마이저/이미지 크기 -- num_heads만 바꿔서 돌려야
head 수 자체의 효과만 분리해서 볼 수 있다.

매 epoch 로그에 (진단용, loss에는 안 씀):
  - 장기별 Dice/IoU
  - head 간 평균 코사인 유사도(head_diversity_penalty, no_grad) -- head 수가
    늘어날 때 head들이 실제로 달라지는지(낮은 코사인) 아니면 여전히 redundant한지
    (높은 코사인, disc/cup에서 관찰된 것과 같은 패턴인지) 바로 비교 가능하게.

Run (Colab, screening -- 5 epoch로 먼저 감 잡기):
    python train_amos_fixed_heads.py \
        --csv-dir /content/data/amos_prepared \
        --output-dir /content/drive/MyDrive/outputs_AMOS \
        --task-names liver,spleen,pancreas,duodenum,gallbladder,kidney,left_adrenal_gland,right_adrenal_gland \
        --epochs 5 \
        --num-heads 1   # 그다음 2, 4, 8로 반복 (다른 인자 전부 동일하게)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from attention_unet_multitask import (
    AttentionUNetResNet34MultiTask,
    MultiTaskCsvDataset,
    MultiTaskEpochResult,
    combined_loss_multitask,
    count_parameters,
    estimate_pixel_pos_weights,
    head_diversity_penalty,
    positive_segmentation_metrics_multitask,
    seed_everything,
    set_pos_weights,
)


def load_split(csv_dir: Path, split: str, task_names: list[str], min_organs_present: int) -> pd.DataFrame:
    """min_organs_present: 이 슬라이스를 쓰려면 task_names 중 최소 몇 개가 존재해야 하는지.
    disc/cup은 "둘 다 항상 존재"가 자연스러웠지만(min_organs_present=len(task_names)와 동일),
    AMOS는 장기별 z축 범위가 서로 많이 달라서(특히 adrenal gland처럼 작은 장기) 전부 동시에
    있는 슬라이스만 쓰면 데이터가 너무 줄어들 수 있다. 그래서 기본은 "1개 이상"으로 느슨하게
    두고, 존재하지 않는 장기는 그 슬라이스에서 마스크가 전부 0(음성 샘플)으로 자연스럽게
    처리된다 -- combined_loss_multitask/positive_segmentation_metrics_multitask 둘 다 이미
    "양성 샘플에서만 집계"하도록 짜여 있어서 이렇게 써도 지표가 왜곡되지 않는다."""
    df = pd.read_csv(csv_dir / f"amos_{split}.csv")
    content_cols = [f"content_{t}" for t in task_names]
    missing = [c for c in content_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{csv_dir / f'amos_{split}.csv'}에 다음 컬럼이 없음: {missing}. "
            f"--task-names 철자가 prepare_amos_dataset.py가 만든 컬럼명(공백->'_', 소문자)과 맞는지 확인하세요."
        )
    before = len(df)
    present_count = df[content_cols].sum(axis=1)
    df = df[present_count >= min_organs_present].reset_index(drop=True)
    print(
        f"  {split}: {before}장 중 지정 장기({task_names}) {min_organs_present}개 이상 존재하는 "
        f"{len(df)}장만 사용"
    )
    return df


def make_dataset(
    df: pd.DataFrame,
    task_names: list[str],
    image_size: int,
    augment: bool,
    cache_in_memory: bool = False,
) -> MultiTaskCsvDataset:
    return MultiTaskCsvDataset(
        df=df,
        task_names=task_names,
        mask_cols=[f"mask_{t}" for t in task_names],
        image_col="filepath",
        content_cols=[f"content_{t}" for t in task_names],
        image_size=image_size,
        augment=augment,
        image_mode="L",  # CT는 그레이스케일
        cache_in_memory=cache_in_memory,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    task_names: list[str],
) -> tuple[MultiTaskEpochResult, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    task_dice_sum = {name: 0.0 for name in task_names}
    task_iou_sum = {name: 0.0 for name in task_names}
    task_pos_count = {name: 0 for name in task_names}
    cos_sum, batches = 0.0, 0

    progress = tqdm(loader, leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.set_grad_enabled(training):
            logits, alphas = model(images)
            loss = combined_loss_multitask(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        with torch.no_grad():
            cos = head_diversity_penalty(alphas)
            cos_sum += cos.item() if hasattr(cos, "item") else float(cos)
        batches += 1

        metrics = positive_segmentation_metrics_multitask(logits.detach(), masks, task_names)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        postfix = {"loss": f"{loss.item():.3f}"}
        for name in task_names:
            dice, iou, n_pos = metrics[name]
            if n_pos > 0:
                task_dice_sum[name] += dice * n_pos
                task_iou_sum[name] += iou * n_pos
                task_pos_count[name] += n_pos
            postfix[f"{name}_dice"] = f"{dice:.3f}" if dice is not None else "n/a"
        progress.set_postfix(**postfix)

    task_dice = {
        name: (task_dice_sum[name] / task_pos_count[name]) if task_pos_count[name] > 0 else 0.0
        for name in task_names
    }
    task_iou = {
        name: (task_iou_sum[name] / task_pos_count[name]) if task_pos_count[name] > 0 else 0.0
        for name in task_names
    }
    avg_cos = cos_sum / max(batches, 1)
    return MultiTaskEpochResult(loss=total_loss / total_samples, task_dice=task_dice, task_iou=task_iou), avg_cos


def train_fixed_heads(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
    task_names: list[str],
    num_heads: int,
) -> dict[str, float | str]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)

    best_score = -1.0
    best_task_dice = {name: -1.0 for name in task_names}
    best_task_iou = {name: -1.0 for name in task_names}
    start = time.time()
    history: list[dict] = []

    print(f"\n{name}")
    print(f"num_heads={num_heads}, task_names={task_names}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    for epoch in range(1, epochs + 1):
        train_result, train_cos = run_epoch(model, train_loader, optimizer, device, task_names)
        val_result, val_cos = run_epoch(model, val_loader, None, device, task_names)
        val_score = sum(val_result.task_dice.values()) / len(task_names)
        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        train_str = ", ".join(f"train {n}_dice={train_result.task_dice[n]:.4f}" for n in task_names)
        val_str = ", ".join(
            f"val {n}_dice={val_result.task_dice[n]:.4f}, val {n}_iou={val_result.task_iou[n]:.4f}"
            for n in task_names
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss={train_result.loss:.4f}, {train_str} | "
            f"val loss={val_result.loss:.4f}, {val_str} | lr={current_lr:.2e}"
        )
        print(f"  head_cosine(train/val) = {train_cos:.4f}/{val_cos:.4f}")

        row = {
            "epoch": epoch,
            "num_heads": num_heads,
            "train_loss": train_result.loss,
            "val_loss": val_result.loss,
            "val_score": val_score,
            "head_cosine_train": train_cos,
            "head_cosine_val": val_cos,
        }
        for n in task_names:
            row[f"train_{n}_pos_dice"] = train_result.task_dice[n]
            row[f"val_{n}_pos_dice"] = val_result.task_dice[n]
            row[f"val_{n}_pos_iou"] = val_result.task_iou[n]
        history.append(row)

        if val_score > best_score:
            best_score = val_score
            best_task_dice = dict(val_result.task_dice)
            best_task_iou = dict(val_result.task_iou)
            ckpt = {
                "model_name": name,
                "model_state_dict": model.state_dict(),
                "task_names": task_names,
                "num_heads": num_heads,
                "head_cosine_at_best": val_cos,
            }
            for n in task_names:
                ckpt[f"best_val_{n}_pos_dice"] = best_task_dice[n]
                ckpt[f"best_val_{n}_pos_iou"] = best_task_iou[n]
            torch.save(ckpt, output_dir / f"{name}.pt")

        history_path = output_dir / f"history_{name}.csv"
        with history_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    elapsed = time.time() - start
    print(f"Epoch별 기록 저장: {history_path}")

    result = {
        "model": name,
        "num_heads": num_heads,
        "parameters": count_parameters(model),
        "training_seconds": elapsed,
        "final_head_cosine_val": val_cos,
    }
    for n in task_names:
        result[f"best_val_{n}_pos_dice"] = best_task_dice[n]
        result[f"best_val_{n}_pos_iou"] = best_task_iou[n]
    result["best_val_mean_dice"] = sum(best_task_dice.values()) / len(task_names)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=str, default="/content/data/amos_prepared")
    parser.add_argument("--output-dir", type=str, default="/content/drive/MyDrive/outputs_AMOS")
    parser.add_argument(
        "--task-names", type=str, required=True,
        help="콤마로 구분한 장기 이름(prepare_amos_dataset.py가 만든 컬럼명 기준, 예: "
        "liver,spleen,pancreas,duodenum). 이 조합의 content_* 컬럼이 전부 True인 슬라이스만 사용.",
    )
    parser.add_argument("--num-heads", type=int, required=True, help="1, 2, 4, 8 등 -- H1/H2/H4/H8 비교의 그 값")
    parser.add_argument(
        "--min-organs-present", type=int, default=1,
        help="슬라이스에 --task-names 중 최소 몇 개가 있어야 학습에 쓸지. 기본 1(느슨). "
        "disc/cup처럼 전부 동시 존재하는 것만 쓰고 싶으면 --task-names 개수와 같은 값을 주면 됨.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--cache-dataset", action="store_true",
        help="image+mask를 resize까지 끝낸 상태로 메모리에 캐싱(두 번째 epoch부터 디스크 I/O "
             "생략). HDD I/O 병목일 때 사용, --workers 0과 같이 쓸 것(멀티프로세스 워커별로 "
             "캐시가 따로 생겨 메모리가 중복될 수 있음).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-imagenet-pretrained", action="store_true")
    args = parser.parse_args()

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

    imagenet_pretrained = not args.no_imagenet_pretrained
    name = f"multihead_attention_unet_amos_H{args.num_heads}_{len(task_names)}organs_seed{args.seed}"
    model = AttentionUNetResNet34MultiTask(
        num_tasks=len(task_names), gate_type="multi_split", num_heads=args.num_heads,
        imagenet_pretrained=imagenet_pretrained,
    )

    result = train_fixed_heads(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, epochs=args.epochs, learning_rate=args.lr, output_dir=output_dir,
        task_names=task_names, num_heads=args.num_heads,
    )

    result["seed"] = args.seed

    csv_path = output_dir / "comparison_amos_fixed_heads.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print("\nComparison (AMOS fixed-head)")
    print(result)
    print(f"\nSaved results to: {csv_path}")


if __name__ == "__main__":
    main()
