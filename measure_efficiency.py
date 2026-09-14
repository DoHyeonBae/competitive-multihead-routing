"""
6개 architecture(H1/H2/H4/Stage-Adaptive/Head-Projection x2)의 Params / FLOPs(MACs) /
inference 시간 / peak VRAM을 동일 입력 크기에서 측정.

배경: "Reviewer 2" 스타일 critique #4 -- head 수를 줄이는 것(Stage-Adaptive)이 실제로
효율성 이득으로 이어진다는 증거가 없다는 지적에 대한 답. 학습 데이터/체크포인트가
전혀 필요 없고(dummy tensor로 forward/backward만 돌림), GPU 한 대에서 수 분이면
6개 configuration 전부 측정 가능하다.

측정 항목(모델당):
    - Params: 학습 가능한 파라미터 수
    - FLOPs(MACs x2): thop으로 batch=1 기준 1회 forward pass 계산량
    - Inference time: model.eval() + torch.no_grad(), forward-only, batch=1과
      batch=8(학습 때 쓴 배치 크기) 각각 측정. warmup 후 --iters회 반복, 평균/표준편차 ms 보고.
    - Peak VRAM (inference): 위 forward-only 루프에서 torch.cuda.max_memory_allocated()
    - Peak VRAM (train step): forward+backward+optimizer.step() 1 스텝 기준 peak.
      (AdamW optimizer state까지 포함 -- 실제 학습 중 VRAM 사용량에 더 가까움)

주의: CUDA가 없는 환경(CPU-only)에서도 동작은 하지만, inference time/VRAM 숫자는
GPU에서 측정한 것이라야 논문에 의미가 있다(6개 모델을 "같은 하드웨어"에서 상대
비교하는 게 핵심이므로, 반드시 학습에 썼던 것과 같은 GPU에서 실행할 것). thop이
설치되어 있어야 함: pip install thop

Run:
    python measure_efficiency.py --output-dir outputs_efficiency --batch-sizes 1,8 --iters 50
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from attention_unet_multitask import AttentionUNetResNet34MultiTask, count_parameters

CONFIGS: list[tuple[str, str, "int | list[int]"]] = [
    ("H1", "competitive_moe", 1),
    ("H2", "competitive_moe", 2),
    ("H4", "competitive_moe", 4),
    ("Stage-Adaptive[2,2,2,1]", "competitive_moe", [2, 2, 2, 1]),
    ("Head-Projection(H2)", "head_proj_moe", 2),
    ("Head-Projection[2,2,2,1]", "head_proj_moe", [2, 2, 2, 1]),
]


def measure_flops_params(model: torch.nn.Module, image_size: int, device: torch.device) -> tuple[float, int]:
    from thop import profile

    x = torch.randn(1, 3, image_size, image_size, device=device)
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)
    return macs * 2, int(params)  # FLOPs = 2 * MACs (표준 관례)


def measure_inference(
    model: torch.nn.Module, batch_size: int, image_size: int, device: torch.device, warmup: int, iters: int,
) -> tuple[float, float, float]:
    """Forward-only 추론 시간(배치당 ms 평균/표준편차)과 peak VRAM(MB) 반환."""
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        times = []
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else float("nan")
    mean_ms = sum(times) / len(times)
    var = sum((t - mean_ms) ** 2 for t in times) / len(times)
    return mean_ms, var ** 0.5, peak_mb


def measure_train_step(
    model: torch.nn.Module, batch_size: int, image_size: int, num_tasks: int, device: torch.device,
    warmup: int, iters: int,
) -> tuple[float, float, float]:
    """Forward+backward+optimizer.step() 기준 스텝당 시간(평균 ms, 표준편차)과 peak VRAM(MB).
    warmup 스텝을 먼저 돌려서 cuDNN 알고리즘 탐색/캐시 워밍업 오버헤드가 측정치에 안 섞이게
    한다(예전 버전은 단 1 스텝만 재서 이 오버헤드가 그대로 노이즈로 들어갔었음)."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    # forward()가 (logits, [a1,a2,a3,a4]) 튜플을 반환함(attention_unet_multitask.py 참고) --
    # logits shape=(B, num_tasks, H, W). alpha map들은 loss 계산에 직접 안 쓰이므로 무시.
    target = torch.randint(0, 2, (batch_size, num_tasks, image_size, image_size), device=device).float()

    def one_step() -> None:
        optimizer.zero_grad()
        logits, _alphas = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else float("nan")
    mean_ms = sum(times) / len(times)
    var = sum((t - mean_ms) ** 2 for t in times) / len(times)
    return mean_ms, var ** 0.5, peak_mb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs_efficiency")
    parser.add_argument("--num-tasks", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-sizes", type=str, default="1,8", help="쉼표로 구분(예: '1,8'). 8은 학습 때 배치 크기.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("[경고] CUDA 없이 CPU로 측정 중 -- inference time/VRAM 숫자는 논문에 쓸 수 없음 "
              "(6개 모델 상대비교용 하드웨어가 학습에 쓴 GPU와 달라짐). Params/FLOPs만 참고할 것.")

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, gate_type, num_heads in CONFIGS:
        print(f"\n=== {name} (gate_type={gate_type}, num_heads={num_heads}) ===")
        model = AttentionUNetResNet34MultiTask(
            num_tasks=args.num_tasks, gate_type=gate_type, num_heads=num_heads, imagenet_pretrained=False,
        ).to(device)

        flops, params = measure_flops_params(model, args.image_size, device)
        print(f"  Params={params:,} | FLOPs(batch=1)={flops/1e9:.3f} GFLOPs")

        row = {"model": name, "gate_type": gate_type, "num_heads": str(num_heads), "params": params, "gflops_bs1": flops / 1e9}

        for bs in batch_sizes:
            mean_ms, std_ms, peak_mb = measure_inference(model, bs, args.image_size, device, args.warmup, args.iters)
            print(f"  [inference bs={bs}] {mean_ms:.2f}±{std_ms:.2f} ms/batch ({mean_ms/bs:.2f} ms/image) | peak VRAM={peak_mb:.1f} MB")
            row[f"infer_ms_bs{bs}"] = mean_ms
            row[f"infer_ms_per_image_bs{bs}"] = mean_ms / bs
            row[f"infer_peak_vram_mb_bs{bs}"] = peak_mb

            train_ms, train_std_ms, train_peak_mb = measure_train_step(
                model, bs, args.image_size, args.num_tasks, device, args.warmup, args.iters,
            )
            print(f"  [train-step bs={bs}] {train_ms:.2f}±{train_std_ms:.2f} ms/step | peak VRAM={train_peak_mb:.1f} MB")
            row[f"train_step_ms_bs{bs}"] = train_ms
            row[f"train_step_std_ms_bs{bs}"] = train_std_ms
            row[f"train_peak_vram_mb_bs{bs}"] = train_peak_mb

        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "efficiency_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n결과 저장: {csv_path}")


if __name__ == "__main__":
    main()