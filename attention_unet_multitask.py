
"""
Multi-Head Attention Gate — 데이터셋/태스크에 구애받지 않는 범용 멀티태스크 버전.

모델/Dataset/loss/metric 전부 "타겟이 몇 개인지, 이름이 뭔지"를 모른 채로
동작한다 (num_tasks, task_names를 인자로만 받음). 특정 데이터셋(LiTS의 간+종양
등) 관련 설정은 이 파일에 전혀 없고, 이 파일을 사용하는 진입점 스크립트
(예: train_multitask.py)에서 CSV 컬럼명/타겟 이름을 넘겨준다. 그래서 다른
멀티태스크 데이터셋(예: 결절+갑상선 전체)이 생겨도 이 파일은 그대로 두고
진입점 스크립트만 새로 짜면 된다.

Attention Gate 계열은 head-to-task를 미리 정해주지 않는 대칭(symmetric) 구조를
유지한다 — "타겟이 여러 개면 head들이 자연스럽게 갈라지는가"를 관찰하는 게
목적이지, 강제로 분업시키는 게 아니기 때문.

Install:
    pip install torch torchvision tqdm pandas pillow
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset: 범용 멀티태스크 2D 분할 (CSV 인덱스 기반)
# ============================================================

class MultiTaskCsvDataset(Dataset):
    """task_names 순서대로 mask_cols에 지정된 마스크 컬럼들을 채널로 쌓아서
    (num_tasks, H, W) 마스크 텐서를 반환하는 범용 데이터셋. 어떤 타겟을 몇 개
    쓰는지는 전부 생성자 인자로 결정되고, 이 클래스 자체는 "liver"/"tumor" 같은
    특정 이름을 전혀 모른다.

    Args:
        df: image_col/mask_cols가 실제 파일 경로를 담고 있는 DataFrame.
        task_names: 채널 순서에 대응하는 타겟 이름 리스트 (길이 = num_tasks).
        mask_cols: task_names와 같은 순서의 마스크 파일 경로 컬럼명 리스트.
        image_col: 원본 이미지 파일 경로 컬럼명.
        filter_col: 주어지면 이 컬럼이 True인 행만 사용 (예: 관심 장기가 아예
            안 보이는 슬라이스를 미리 제외하고 싶을 때).
        content_cols: task_names와 같은 순서의 "양성 여부" boolean 컬럼명 리스트
            (있으면 positive_ratio() 계산이 빠름). 없으면 마스크 파일을 직접
            스캔해서 계산한다(느리지만 어떤 데이터셋에도 동작).
        image_mode: "L"(그레이스케일, CT/초음파처럼 흑백 원본을 3채널로 복제 -
            LiTS 등 기존 흐름과 동일, 기본값) 또는 "RGB"(안저 사진처럼 색상
            정보 자체가 판별에 중요한 원본 컬러 이미지를 그대로 사용).
        cache_in_memory: True면 (resize까지 끝낸, augment/normalize 전 단계의)
            image/mask를 프로세스 메모리에 dict로 캐싱해서 두 번째 epoch부터는
            디스크 I/O(Image.open)를 건너뛴다. HDD처럼 random-access 비용이
            큰 저장 매체에서 "task 수 * 샘플 수"가 커져 매 epoch마다 같은
            파일을 반복해서 여는 게 병목일 때 켠다(예: AMOS 15-organ, 21925
            샘플 x 16개 파일/샘플). 기본 False -- 메모리를 새로 쓰는 옵션이라
            호출부가 명시적으로 켜야 함. flip augmentation은 캐시와 무관하게
            매번 새로 적용되므로 augment=True에서도 매 epoch 다른 flip이 나옴.
            메모리 추정: (H*W + num_tasks*H*W) bytes/sample (uint8 기준) --
            image_size=224, num_tasks=15면 약 803KB/sample.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task_names: list[str],
        mask_cols: list[str],
        image_col: str = "filepath",
        filter_col: str | None = None,
        content_cols: list[str] | None = None,
        image_size: int = 224,
        augment: bool = False,
        image_mode: str = "L",
        cache_in_memory: bool = False,
    ) -> None:
        if len(task_names) != len(mask_cols):
            raise ValueError("task_names와 mask_cols 길이가 달라요")
        if content_cols is not None and len(content_cols) != len(task_names):
            raise ValueError("content_cols와 task_names 길이가 달라요")

        if image_mode not in ("L", "RGB"):
            raise ValueError(f"image_mode는 'L' 또는 'RGB'만 지원함: {image_mode}")

        self.task_names = list(task_names)
        self.mask_cols = list(mask_cols)
        self.image_col = image_col
        self.content_cols = list(content_cols) if content_cols is not None else None
        self.image_size = image_size
        self.augment = augment
        self.image_mode = image_mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = {} if cache_in_memory else None

        data = df[df[filter_col]] if filter_col is not None else df
        self.data = data.reset_index(drop=True)

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)

    def positive_ratio(self, task_name: str) -> float:
        idx = self.task_names.index(task_name)
        if self.content_cols is not None:
            return float(self.data[self.content_cols[idx]].mean())

        # content_cols가 없으면 마스크 파일을 직접 열어서 계산 (느림, fallback용)
        mask_col = self.mask_cols[idx]

        def _has_content(path: str) -> bool:
            m = np.array(Image.open(path).convert("L"))
            return bool((m > 0).any())

        return float(self.data[mask_col].map(_has_content).mean())

    def __len__(self) -> int:
        return len(self.data)

    def _load_resized(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """디스크에서 image+mask를 읽어 resize까지 끝낸 (augment/normalize 전) 형태로
        반환. image_np: (H,W) 또는 (H,W,3) uint8. mask_np: (num_tasks,H,W) uint8(0/1).
        cache_in_memory=True면 이 결과 자체가 캐싱 대상(디스크 I/O가 여기 전부 있음).

        content_cols가 주어져 있고 해당 슬라이스에 그 장기가 없으면(content=False)
        마스크 PNG는 정의상 전부 0이므로(load_split 필터링 로직과 동일 전제)
        파일을 열지 않고 바로 zero 배열을 만든다 -- 결과에 영향 없는 순수 I/O
        최적화. task 수가 많고(예: 15) 슬라이스당 실제 존재하는 장기 수가 적을 때
        (AMOS 실측 평균 ~4.3/15) mask read를 크게 줄여준다."""
        row = self.data.iloc[index]

        image = Image.open(row[self.image_col]).convert(self.image_mode)
        image = TF.resize(
            image, [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        image_np = np.array(image, dtype=np.uint8)  # 0~255 그대로 캐싱(메모리 절약, float 변환은 매번 새로)

        mask_channels = []
        for i, col in enumerate(self.mask_cols):
            if self.content_cols is not None and not bool(row[self.content_cols[i]]):
                mask_channels.append(np.zeros((self.image_size, self.image_size), dtype=np.uint8))
                continue
            m = Image.open(row[col]).convert("L")
            m = TF.resize(m, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
            mask_channels.append((np.array(m) > 0).astype(np.uint8))
        mask_np = np.stack(mask_channels, axis=0)  # (num_tasks,H,W)
        return image_np, mask_np

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            cached = self._cache.get(index)
            if cached is None:
                cached = self._load_resized(index)
                self._cache[index] = cached
            image_np, mask_np = cached
        else:
            image_np, mask_np = self._load_resized(index)

        # flip은 캐시 여부와 무관하게 매번 새로 굴려서, 캐시를 켜도 매 epoch 다른
        # augmentation이 나오게 함(캐시된 배열 자체는 절대 in-place 수정 안 함).
        if self.augment and random.random() < 0.5:
            image_np = np.flip(image_np, axis=1)  # (H,W[,3]) -> W축(=axis1) 반전
            mask_np = np.flip(mask_np, axis=2)     # (num_tasks,H,W) -> W축(=axis2) 반전

        image_f = np.ascontiguousarray(image_np, dtype=np.float32) / 255.0
        if self.image_mode == "RGB":
            image_chw = image_f.transpose(2, 0, 1)  # (H,W,3) -> (3,H,W)
        else:
            image_chw = np.stack([image_f, image_f, image_f], axis=0)  # 1ch -> 3ch 복제
        image_tensor = torch.from_numpy(image_chw).float()
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)

        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_np, dtype=np.float32))

        return image_tensor, mask_tensor


# ============================================================
# Model (Attention Gate 계열: 인코더 feature에서만 동작, 타겟 수와 무관)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """원조 단일 Attention Gate (baseline용)."""

    def __init__(self, skip_channels: int, gating_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi_g = nn.Conv2d(gating_channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_proj = self.theta_x(x)
        g_proj = self.phi_g(g)
        g_proj = F.interpolate(g_proj, size=x_proj.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(self.relu(x_proj + g_proj)))
        return x * alpha, alpha


class _SingleHeadGate(nn.Module):
    def __init__(self, skip_channels, gating_channels, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi_g = nn.Conv2d(gating_channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        x_proj = self.theta_x(x)
        g_proj = self.phi_g(g)
        g_proj = F.interpolate(g_proj, size=x_proj.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(self.relu(x_proj + g_proj)))
        return x * alpha, alpha


class MultiHeadAttentionGate(nn.Module):
    """독립형 Multi-Head Attention Gate (head 늘수록 파라미터도 늘어남, confound 있음)."""

    def __init__(self, skip_channels: int, gating_channels: int, inter_channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [_SingleHeadGate(skip_channels, gating_channels, inter_channels) for _ in range(num_heads)]
        )
        self.fusion = nn.Conv2d(skip_channels * num_heads, skip_channels, kernel_size=1, bias=False)

    def forward(self, x, g):
        weighted, alphas = [], []
        for head in self.heads:
            feat, alpha = head(x, g)
            weighted.append(feat)
            alphas.append(alpha)
        fused = self.fusion(torch.cat(weighted, dim=1))
        alpha_all = torch.cat(alphas, dim=1)
        return fused, alpha_all


class ChannelSplitMultiHeadAttentionGate(nn.Module):
    """파라미터 수를 head 수와 무관하게 고정한 Multi-Head Attention Gate
    (Transformer의 multi-head attention과 동일한 방식)."""

    def __init__(self, skip_channels: int, gating_channels: int, inter_channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if skip_channels % num_heads != 0 or inter_channels % num_heads != 0:
            raise ValueError(
                f"skip_channels({skip_channels}) / inter_channels({inter_channels})가 "
                f"num_heads({num_heads})로 나눠떨어지지 않음"
            )
        self.num_heads = num_heads
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi_g = nn.Conv2d(gating_channels, inter_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        inter_per_head = inter_channels // num_heads
        self.psi_heads = nn.ModuleList([nn.Conv2d(inter_per_head, 1, 1) for _ in range(num_heads)])
        self.fusion = nn.Conv2d(skip_channels, skip_channels, 1, bias=False)

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """active_mask: (num_heads,) 0/1 텐서. 주어지면 해당 head의 alpha를 통째로
        0으로 눌러서(=그 head의 skip feature 기여가 사라짐) 마치 그 head가 없는
        것처럼 동작시킨다 -- adaptive head 실험용. None이면 기존과 완전히 동일
        (전 head 항상 켜짐, 기존 실험들과 하위호환)."""
        x_proj = self.theta_x(x)
        g_proj = self.phi_g(g)
        g_proj = F.interpolate(g_proj, size=x_proj.shape[-2:], mode="bilinear", align_corners=False)
        combined = self.relu(x_proj + g_proj)

        combined_splits = combined.chunk(self.num_heads, dim=1)
        x_splits = x.chunk(self.num_heads, dim=1)

        gated_splits, alphas = [], []
        for i in range(self.num_heads):
            alpha_h = torch.sigmoid(self.psi_heads[i](combined_splits[i]))
            if active_mask is not None:
                alpha_h = alpha_h * active_mask[i]
            gated_splits.append(x_splits[i] * alpha_h)
            alphas.append(alpha_h)

        gated = self.fusion(torch.cat(gated_splits, dim=1))
        alpha_all = torch.cat(alphas, dim=1)
        return gated, alpha_all


class CompetitiveMultiHeadAttentionGate(nn.Module):
    """Shared relevance gate + competitive head router (독립 sigmoid의 "경쟁 없음" 문제를
    겨냥한 구조).

    기존 ChannelSplitMultiHeadAttentionGate는 head마다 독립적인
        alpha_h(p) = sigmoid(psi_h(...))
    라서 "이 위치가 중요한가"와 "어느 head가 담당하는가"를 각 head가 따로 판단한다.
    경쟁이 없으니 전부 1로(dec3/dec4에서 관찰된 saturation) 혹은 전부 비슷하게(cosine
    0.97+, dec1/dec2) 수렴해도 loss 입장에서 전혀 나쁠 게 없다.

    여기서는 두 역할을 분리한다:
        s(p)   = sigmoid(psi_shared(...))         -- "이 위치가 중요한가" (head 공유, 1채널)
        r_h(p) = softmax_h(z_h(p) / temperature)  -- "중요하다면 어느 head가 담당하는가"
                                                       (head끼리 직접 경쟁, sum_h r_h(p)=1)
        alpha_h(p) = s(p) * num_heads * r_h(p)

    r_h가 모든 head에서 1/num_heads(uniform)이고 s=1이면 alpha_h=1 (all h) -- 기존에
    dec3/dec4에서 관찰된 saturated 해를 그대로 재현 가능하다. 즉 이 구조가 어디에서든
    강제로 분업을 시키는 게 아니라, 분업이 "가능해지도록" 경쟁 메커니즘만 추가한 것 --
    실제로 갈라질지는 학습이 정한다(symmetric 철학 유지).

    population-level load-balancing loss(head_balance_loss, 학습 스크립트 쪽)는 이
    gate가 반환하는 alpha에서 r_h(p) = alpha_h(p) / sum_h(alpha_h(p))로 정확히 복원
    가능하므로, forward()의 반환 시그니처는 기존 게이트들과 동일하게 (gated, alpha)로
    유지한다(다른 스크립트/체크포인트 호환성 그대로)."""

    def __init__(
        self, skip_channels: int, gating_channels: int, inter_channels: int,
        num_heads: int = 4, temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if skip_channels % num_heads != 0:
            raise ValueError(f"skip_channels({skip_channels})가 num_heads({num_heads})로 안 나눠떨어짐")
        self.num_heads = num_heads
        self.temperature = temperature
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi_g = nn.Conv2d(gating_channels, inter_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.psi_shared = nn.Conv2d(inter_channels, 1, 1)          # s(p)
        self.psi_router = nn.Conv2d(inter_channels, num_heads, 1)  # z_h(p), head별 로짓
        self.fusion = nn.Conv2d(skip_channels, skip_channels, 1, bias=False)

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_proj = self.theta_x(x)
        g_proj = self.phi_g(g)
        g_proj = F.interpolate(g_proj, size=x_proj.shape[-2:], mode="bilinear", align_corners=False)
        combined = self.relu(x_proj + g_proj)

        s = torch.sigmoid(self.psi_shared(combined))                       # (B,1,H,W)
        logits = self.psi_router(combined) / self.temperature              # (B,num_heads,H,W)

        # top_k: 재학습 실험용 hard sparse routing 스위치(기본 None=꺼짐, 기존 실험/체크포인트와
        # 완전히 하위호환). force_uniform_router와 동일한 컨벤션으로 인스턴스 속성으로만 켠다
        # (`gate.top_k = 2`처럼). 픽셀별로 상위 top_k개 logit만 남기고 나머지는 -inf로 눌러서
        # softmax하면 그 픽셀에서 top_k 밖 head는 정확히 r_h=0(=alpha_h=0)이 된다 -- Switch
        # Transformer/GShard류가 쓰는 표준 top-k gating과 동일한 방식. top_k 선택(어떤 head가
        # 상위인지) 자체는 미분 불가능(hard argmax류라 당연함)이지만, 살아남은 top_k개 사이의
        # 가중치 배분은 정상적으로 미분 가능해서 학습이 된다.
        top_k = getattr(self, "top_k", None)
        if top_k is not None and top_k < self.num_heads:
            topk_vals, topk_idx = logits.topk(top_k, dim=1)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, topk_idx, topk_vals)

        r = torch.softmax(logits, dim=1)                                   # head 축 경쟁, sum_h r=1

        # force_uniform_router: 재학습 없는 causal ablation용 스위치(기본 False, 기존
        # 실험/체크포인트와 완전히 하위호환). True로 켜면 학습된 r_h(p)를 버리고 1/num_heads로
        # 강제 치환하되 s(p)는 그대로 둔다 -- "학습된 routing이 실제 예측 성능에 기여하는가"를
        # 확인하는 uniform-router ablation 전용 hook. diagnose_ablation_*.py에서
        # `gate.force_uniform_router = True`처럼 인스턴스 속성으로만 켜고 끈다(클래스 기본값
        # 없이 getattr로 조회하므로, 이 속성을 아예 안 건드리면 항상 기존과 동일하게 동작).
        if getattr(self, "force_uniform_router", False):
            r = torch.full_like(r, 1.0 / self.num_heads)

        # last_r_for_loss: 미분 가능한 r(학습 loss용, 예: hard_organ_routing_loss).
        # last_r(아래)은 detach돼 있어서 그걸로 loss를 만들면 gradient가 psi_router까지
        # 전혀 안 들어가는 치명적 실수가 됨 -- 반드시 이쪽을 써야 함. alpha_h/sum(alpha_h)로
        # r을 재구성하는 것도 하지 말 것(s≈0 근처에서 수치 불안정했던 이력이 head-organ
        # 분석 v2/v3에서 이미 확인됨) -- softmax에서 나온 이 r을 그대로 쓰는 게 맞다.
        self.last_r_for_loss = r
        self.last_r = r.detach()  # 진단용(head-organ 분석 등): s와 완전히 독립인 진짜 routing
        self.last_s = s.detach()  # 진단용: 위치 relevance 자체가 궁금할 때

        x_splits = x.chunk(self.num_heads, dim=1)
        gated_splits, alphas = [], []
        for i in range(self.num_heads):
            alpha_h = s * self.num_heads * r[:, i : i + 1, :, :]
            if active_mask is not None:
                alpha_h = alpha_h * active_mask[i]
            gated_splits.append(x_splits[i] * alpha_h)
            alphas.append(alpha_h)

        gated = self.fusion(torch.cat(gated_splits, dim=1))
        alpha_all = torch.cat(alphas, dim=1)
        return gated, alpha_all


class HeadProjectionMultiHeadAttentionGate(nn.Module):
    """"Head-Specific Projection" 실험용 게이트 (H1/H2/H4/stage-adaptive[2,2,2,1] 다음 실험,
    2026-09). CompetitiveMultiHeadAttentionGate와 차이:

    기존 CompetitiveMultiHeadAttentionGate:
        theta_x/phi_g/psi_shared/psi_router를 head끼리 공유 -> 하나의 combined(x,g)에서
        s(p)와 r_h(p)를 전부 계산. head마다 다른 건 raw x를 채널 축으로 잘라 쓰는 것뿐
        (x_splits = x.chunk(num_heads)) -- attention을 "계산"하는 파라미터 자체는 공유.

    이 게이트:
        F_h = P_h(x)                     -- head마다 독립된 1x1 conv로 x 전체(채널 자르기 없음)를
                                             각자의 feature subspace로 투영
        A_h = sigmoid(psi_h(relu(theta_x_h(F_h) + phi_g_h(g))))   -- head마다 완전히 독립된
                                             attention 서브네트워크(파라미터 공유 없음)
        Z_h = A_h * F_h
        r_h = softmax_h(psi_router(relu(theta_x_shared(x)+phi_g_shared(g))))  -- "어느 head를
                                             얼마나 신뢰할지"는 기존과 동일하게 경쟁시켜서 결정
                                             (population-level balance/entropy loss와 호환 유지)
        Z   = fusion(concat_h[r_h * Z_h])

    가설: 지금까지 head들이 "같은 shared feature에서 계산된 attention map"만 다르게 가져서
    raw Dice 개선이 제한적이었을 수 있다 -- attention을 계산하기 전부터 feature
    representation 자체를 head마다 분리하면(P_h) representation-level specialization이
    가능해지고, 그게 성능 개선으로 이어지는지 확인하는 것. 이건 검증된 결론이 아니라
    새로운 가설(H1/H2/H4/stage-adaptive 결과가 이 가설을 강하게 지지하는 건 아님 -- 그
    실험들은 "필요한 head 수는 적다"를 보여줬을 뿐, "왜 raw Dice가 안 오르는가"에 대한
    직접적 증거는 아니었음).

    주의: force_uniform_router 같은 ablation 스위치나 diagnose_*.py 스크립트들은 아직 이
    gate_type을 지원하지 않음(전부 competitive_moe 전용으로 하드코딩돼 있음) -- 이 실험은
    우선 raw Dice만으로 1차 스크리닝하고, 성능이 유의미하면 그때 진단 스크립트 확장을
    별도로 진행할 것."""

    def __init__(
        self, skip_channels: int, gating_channels: int, inter_channels: int,
        num_heads: int = 2, temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if inter_channels % num_heads != 0:
            raise ValueError(f"inter_channels({inter_channels})가 num_heads({num_heads})로 안 나눠떨어짐")
        self.num_heads = num_heads
        self.temperature = temperature
        inter_per_head = inter_channels // num_heads

        # 라우팅(r_h) 계산용 -- head끼리 공유, CompetitiveMultiHeadAttentionGate와 동일한 역할
        self.theta_x_shared = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi_g_shared = nn.Conv2d(gating_channels, inter_channels, 1, bias=False)
        self.psi_router = nn.Conv2d(inter_channels, num_heads, 1)

        # head마다 독립: projection P_h + 독립 attention 서브네트워크 A_h
        self.proj_heads = nn.ModuleList(
            [nn.Conv2d(skip_channels, skip_channels, 1) for _ in range(num_heads)]
        )
        self.theta_x_heads = nn.ModuleList(
            [nn.Conv2d(skip_channels, inter_per_head, 1, bias=False) for _ in range(num_heads)]
        )
        self.phi_g_heads = nn.ModuleList(
            [nn.Conv2d(gating_channels, inter_per_head, 1, bias=False) for _ in range(num_heads)]
        )
        self.psi_heads = nn.ModuleList([nn.Conv2d(inter_per_head, 1, 1) for _ in range(num_heads)])
        self.relu = nn.ReLU(inplace=True)
        self.fusion = nn.Conv2d(skip_channels * num_heads, skip_channels, 1, bias=False)

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # -- 라우팅(r_h): 기존 competitive_moe와 동일한 방식(shared theta_x/phi_g -> psi_router) --
        x_proj_shared = self.theta_x_shared(x)
        g_proj_shared = self.phi_g_shared(g)
        g_proj_shared = F.interpolate(
            g_proj_shared, size=x_proj_shared.shape[-2:], mode="bilinear", align_corners=False
        )
        combined_shared = self.relu(x_proj_shared + g_proj_shared)
        logits = self.psi_router(combined_shared) / self.temperature
        r = torch.softmax(logits, dim=1)  # (B,num_heads,H,W), head 축 경쟁, sum_h r=1

        self.last_r_for_loss = r
        self.last_r = r.detach()

        # -- head별 독립 projection + attention --
        gated_splits, z_maps = [], []
        for i in range(self.num_heads):
            f_h = self.proj_heads[i](x)                                      # P_h(x), 채널 안 자름
            x_proj_h = self.theta_x_heads[i](f_h)
            g_proj_h = self.phi_g_heads[i](g)
            g_proj_h = F.interpolate(g_proj_h, size=x_proj_h.shape[-2:], mode="bilinear", align_corners=False)
            combined_h = self.relu(x_proj_h + g_proj_h)
            a_h = torch.sigmoid(self.psi_heads[i](combined_h))               # (B,1,H,W)
            z_h = a_h * f_h                                                  # (B,skip_channels,H,W)
            r_h = r[:, i : i + 1, :, :]
            if active_mask is not None:
                r_h = r_h * active_mask[i]
            gated_splits.append(r_h * z_h)
            z_maps.append(a_h)  # 진단용(head별 attention map, alpha_all 대신 참고용)

        gated = self.fusion(torch.cat(gated_splits, dim=1))
        # alpha_all은 balance/entropy loss가 "r = alpha/sum(alpha)"로 정확히 복원할 수 있도록
        # r 자체를 그대로 반환한다(A_h를 곱해버리면 그 복원식이 깨짐 -- A_h는 head별 attention
        # map일 뿐 routing 분포가 아니라서, alpha에 섞으면 head_balance_loss/head_entropy_loss가
        # 엉뚱한 걸 재구성하게 됨).
        return gated, r


class ResNet34Encoder(nn.Module):
    def __init__(self, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if imagenet_pretrained else None
        backbone = resnet34(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class DecoderStage(nn.Module):
    def __init__(
        self,
        decoder_in: int,
        skip_channels: int,
        out_channels: int,
        gate_type: Literal["single", "multi", "multi_split", "competitive_moe", "head_proj_moe"],
        num_heads: int,
    ) -> None:
        super().__init__()
        inter_channels = max(skip_channels // 2, 16)
        self.gate_type = gate_type

        if gate_type == "single":
            self.gate = AttentionGate(skip_channels, decoder_in, inter_channels)
        elif gate_type == "multi":
            self.gate = MultiHeadAttentionGate(skip_channels, decoder_in, inter_channels, num_heads=num_heads)
        elif gate_type == "competitive_moe":
            self.gate = CompetitiveMultiHeadAttentionGate(
                skip_channels, decoder_in, inter_channels, num_heads=num_heads
            )
        elif gate_type == "head_proj_moe":
            self.gate = HeadProjectionMultiHeadAttentionGate(
                skip_channels, decoder_in, inter_channels, num_heads=num_heads
            )
        elif gate_type == "multi_split":
            self.gate = ChannelSplitMultiHeadAttentionGate(
                skip_channels, decoder_in, inter_channels, num_heads=num_heads
            )
        else:
            raise ValueError(
                f"알 수 없는 gate_type: {gate_type!r}. "
                f"'single'/'multi'/'multi_split'/'competitive_moe'/'head_proj_moe' 중 하나여야 함 -- "
                f"이 에러가 나온다는 건 이 파일(attention_unet_multitask.py)이 오래된 버전이거나 "
                f"오타가 있다는 뜻(예전엔 여기서 조용히 ChannelSplitMultiHeadAttentionGate로 "
                f"fallback됐었는데, 그게 seed42/43 competitive gate 실험이 실제로는 전부 구버전 "
                f"게이트로 돌아간 원인이었음 -- 이제는 무조건 에러로 드러나게 고침)."
            )

        self.conv = ConvBlock(decoder_in + skip_channels, out_channels)

    def forward(
        self, decoder_feature: torch.Tensor, skip: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # active_mask는 multi_split 게이트에서만 의미가 있음(adaptive head 실험용).
        # single/multi 게이트는 그냥 무시하고 기존과 동일하게 동작.
        if self.gate_type in ("multi_split", "competitive_moe") and active_mask is not None:
            gated_skip, alpha = self.gate(skip, decoder_feature, active_mask=active_mask)
        else:
            gated_skip, alpha = self.gate(skip, decoder_feature)
        decoder_feature = F.interpolate(decoder_feature, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        out = self.conv(torch.cat([decoder_feature, gated_skip], dim=1))
        return out, alpha


class AttentionUNetResNet34MultiTask(nn.Module):
    """범용 멀티태스크 버전. num_tasks만큼 출력 채널을 갖는다는 것 외에는
    단일 타겟 버전과 완전히 동일한 구조. 어떤 타겟인지는 이 클래스가 전혀 모름."""

    def __init__(
        self,
        num_tasks: int,
        gate_type: Literal["single", "multi", "multi_split", "competitive_moe", "head_proj_moe"] = "single",
        num_heads: int | list[int] = 4,
        imagenet_pretrained: bool = True,
        learnable_head_gate: bool = False,
        gate_init: float | list[float] = 0.9,
        gate_kind: Literal["sigmoid", "hard_concrete"] = "sigmoid",
    ) -> None:
        """learnable_head_gate=True면 head마다 학습 가능한 scalar gate
        g_h = sigmoid(s_h)를 두고(Adaptive/Learnable Head Selection 실험용),
        forward()에서 active_mask를 명시적으로 안 넘기면 이 gate 값을 자동으로
        active_mask 자리에 써서 alpha_h에 곱한다. gate_init(기본 0.9)은 학습
        시작 시점의 g_h 값 -- 처음부터 8개 head를 다 웬만큼 켠 상태로 주고
        segmentation loss + sparsity penalty가 알아서 불필요한 head를 끄도록
        유도하려는 목적. logit=0(g=0.5)에서 시작하면 학습 초반에 sparsity
        압박이 진짜 필요한 head까지 억제할 위험이 있어서 높게 잡는다.

        gate_init에 float 하나 대신 길이 num_heads짜리 리스트를 주면 head마다
        다른 초기값을 쓴다 -- symmetry-breaking 진단 실험용(모든 head가 완전히
        동일한 초기값/구조로 시작하면 gradient가 특정 head를 골라 억제할 비대칭
        신호가 애초에 없어서 다 같이 shrink하는 게 자연스러운 해일 수 있다는
        가설을 검증하려는 목적. 아주 작은 초기 차이를 주고 그게 학습 중 증폭되는지
        사라지는지 관찰).

        num_heads: int 하나를 주면 기존과 동일하게 dec1~dec4 전부 같은 head 수를
        쓴다(완전히 하위호환, 기존 H1/H2/H4/H8 체크포인트 로딩에 영향 없음).
        길이 4짜리 리스트/튜플을 주면 [dec1, dec2, dec3, dec4] 순서로 stage마다
        다른 head 수를 쓴다(stage-adaptive head allocation 실험용 -- H4의
        causal ablation에서 측정된 stage별 effective head 수를 그대로 nominal
        head 수로 주고 처음부터 다시 학습해서, "필요한 만큼만 준 head 배분이
        기존 uniform H2/H4와 성능이 유지되는가"를 검증하기 위함. 순서는 기존
        --lambda-ent-per-stage CLI 옵션과 동일하게 dec1이 먼저임에 주의 --
        내부적으로 dec4가 먼저 생성되므로 여기서 순서를 뒤집어 배정한다)."""
        super().__init__()
        self.num_tasks = num_tasks
        self.encoder = ResNet34Encoder(imagenet_pretrained=imagenet_pretrained)

        if isinstance(num_heads, int):
            heads_per_stage = [num_heads] * 4  # [dec1, dec2, dec3, dec4], 기존과 동일
        else:
            heads_per_stage = list(num_heads)
            if len(heads_per_stage) != 4:
                raise ValueError(
                    f"num_heads를 리스트/튜플로 줄 때는 길이 4([dec1,dec2,dec3,dec4] 순서)여야 함, "
                    f"받은 값: {heads_per_stage}"
                )
            if learnable_head_gate:
                raise ValueError(
                    "learnable_head_gate=True는 아직 stage별 다른 num_heads 조합을 지원하지 않음 "
                    "(gate_init 길이 검증이 단일 num_heads를 가정함) -- 필요하면 별도로 구현할 것."
                )
        # 저장해두면 학습 스크립트/체크포인트에서 실제 배정이 뭐였는지 나중에 참조 가능.
        self.heads_per_stage = heads_per_stage  # [dec1, dec2, dec3, dec4]
        h1, h2, h3, h4 = heads_per_stage

        self.dec4 = DecoderStage(512, 256, 256, gate_type, h4)
        self.dec3 = DecoderStage(256, 128, 128, gate_type, h3)
        self.dec2 = DecoderStage(128, 64, 64, gate_type, h2)
        self.dec1 = DecoderStage(64, 64, 32, gate_type, h1)

        # 파일 버전 불일치로 gate_type 문자열이 조용히 다른 게이트로 fallback되는 사고
        # (2026-09, competitive_moe -> ChannelSplitMultiHeadAttentionGate)가 재발하지
        # 않도록, 실제로 어느 게이트 클래스가 만들어졌는지 매 실행 로그 맨 앞에 무조건 찍는다.
        # heads_per_stage도 같이 찍어서 dec1/dec4 순서가 뒤바뀌는 실수를 즉시 알아챌 수 있게 함.
        print(
            f"[AttentionUNetResNet34MultiTask] gate_type={gate_type!r}, "
            f"heads_per_stage(dec1..dec4)={heads_per_stage} -> "
            f"실제 생성된 게이트 클래스/num_heads: "
            f"dec1={type(self.dec1.gate).__name__}(H={h1}), "
            f"dec2={type(self.dec2.gate).__name__}(H={h2}), "
            f"dec3={type(self.dec3.gate).__name__}(H={h3}), "
            f"dec4={type(self.dec4.gate).__name__}(H={h4})"
        )

        self.head = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, num_tasks, kernel_size=1),
        )

        self.learnable_head_gate = learnable_head_gate
        self.gate_kind = gate_kind
        if learnable_head_gate:
            if gate_type != "multi_split":
                raise ValueError("learnable_head_gate는 gate_type='multi_split'에서만 의미가 있음")
            if isinstance(gate_init, (int, float)):
                init_values = [float(gate_init)] * num_heads
            else:
                init_values = [float(v) for v in gate_init]
                if len(init_values) != num_heads:
                    raise ValueError(f"gate_init 리스트 길이({len(init_values)})가 num_heads({num_heads})와 안 맞음")

            if gate_kind == "sigmoid":
                init_logits = [math.log(v / (1 - v)) for v in init_values]
            elif gate_kind == "hard_concrete":
                # gate_init을 "초기 P(head가 0이 아님)"으로 해석해서 log_alpha를 역산
                # -- sigmoid 버전과 초기 활성도 의미를 동일하게 맞춰 비교 가능하게 함.
                beta_log_ratio = _HC_BETA * math.log(-_HC_GAMMA / _HC_ZETA)
                init_logits = [math.log(v / (1 - v)) + beta_log_ratio for v in init_values]
            else:
                raise ValueError(f"알 수 없는 gate_kind: {gate_kind}")
            self.head_gate_logits = nn.Parameter(torch.tensor(init_logits, dtype=torch.float32))
        else:
            self.head_gate_logits = None

    def head_gate_deterministic(self) -> torch.Tensor:
        """train/eval 모드와 무관하게 항상 결정적(샘플링 없는) gate 값을 반환한다.
        로깅/체크포인트/effective_heads 계산 등 "해석용" 값이 필요할 때 이걸 쓴다.
        hard_concrete면 실제로 정확히 0이 될 수 있는 값이고, sigmoid면 그냥
        sigmoid(logit)."""
        if self.head_gate_logits is None:
            raise RuntimeError("이 모델은 learnable_head_gate=False로 생성됨")
        if self.gate_kind == "sigmoid":
            return torch.sigmoid(self.head_gate_logits)
        s = torch.sigmoid(self.head_gate_logits)
        s_bar = s * (_HC_ZETA - _HC_GAMMA) + _HC_GAMMA
        return s_bar.clamp(0.0, 1.0)

    def head_gate_values(self) -> torch.Tensor:
        """forward()에서 실제로 alpha에 곱해질 gate 값. sigmoid면 항상
        sigmoid(logit)(결정적). hard_concrete면 학습 중(self.training=True)엔
        매 forward마다 Hard-Concrete 분포에서 새로 샘플링(stochastic -- 이 확률성
        자체가 학습 신호가 됨)하고, eval 모드에서는 head_gate_deterministic()과
        동일한 결정적 값을 쓴다."""
        if self.head_gate_logits is None:
            raise RuntimeError("이 모델은 learnable_head_gate=False로 생성됨")
        if self.gate_kind == "sigmoid":
            return torch.sigmoid(self.head_gate_logits)
        if not self.training:
            return self.head_gate_deterministic()
        u = torch.rand_like(self.head_gate_logits).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.head_gate_logits) / _HC_BETA)
        s_bar = s * (_HC_ZETA - _HC_GAMMA) + _HC_GAMMA
        return s_bar.clamp(0.0, 1.0)

    def head_gate_l0_penalty(self) -> torch.Tensor:
        """Hard-Concrete gate 전용 sparsity penalty: 각 head가 "완전히 0이 아닐
        확률"의 합 (닫힌 형태, 미분 가능, Louizos et al. 2017). sigmoid gate의
        L1 penalty(sum(g_h), "크기"에 비례)를 대체한다 -- 크기가 아니라 "켜져
        있을 확률"에 직접 비용을 매겨서, 불필요한 head들이 다 같이 애매하게
        (0.3, 0.4, ...) 줄어드는 대신 실제로 꺼지도록(z=0) 유도하는 게 목적."""
        if self.gate_kind != "hard_concrete":
            raise RuntimeError("head_gate_l0_penalty()는 gate_kind='hard_concrete'에서만 사용 가능")
        beta_log_ratio = _HC_BETA * math.log(-_HC_GAMMA / _HC_ZETA)
        return torch.sigmoid(self.head_gate_logits - beta_log_ratio).sum()

    def forward(
        self, x: torch.Tensor, active_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """active_mask: (num_heads,) 0~1 텐서. 명시적으로 넘기면(예: adaptive
        head 실험의 고정 0/1 마스크) 그걸 그대로 쓰고, None인데
        learnable_head_gate=True면 head_gate_values()를 자동으로 대신 쓴다.
        둘 다 아니면(기존 실험들) 기존과 동일하게 전 head 사용."""
        if active_mask is None and self.learnable_head_gate:
            active_mask = self.head_gate_values()

        input_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encoder(x)

        d4, a4 = self.dec4(x4, x3, active_mask=active_mask)
        d3, a3 = self.dec3(d4, x2, active_mask=active_mask)
        d2, a2 = self.dec2(d3, x1, active_mask=active_mask)
        d1, a1 = self.dec1(d2, x0, active_mask=active_mask)

        logits = self.head(d1)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits, [a1, a2, a3, a4]


# ============================================================
# Hard-Concrete gate (Louizos, Welling & Kingma, "Learning Sparse Neural
# Networks through L0 Regularization", ICLR 2018) -- sigmoid+L1 gate가
# "8개 head가 다 같이 조금씩 줄어드는" diffuse shrinkage로 수렴하고 discrete한
# selection이 안 되는 걸 관찰한 뒤, 이 문제를 정면으로 겨냥해서 설계된 표준
# 기법으로 교체하려는 것. sigmoid gate는 g_h가 이론적으로 정확히 0에 도달할 수
# 없어서(항상 (0,1) 개구간) "필요 없다"는 판단이 "아주 작은 양수"로만 표현되고,
# L1 penalty(sum(g_h))는 크기에 비례한 페널티라 8개가 조금씩 다 낮추는 것과
# 1~2개만 확 낮추는 것의 페널티 총합이 비슷하면 최적화가 굳이 후자를 선호할
# 이유가 없다. Hard-Concrete gate는 (a) stretch + hard-clip 구조 덕분에 실제로
# 정확한 0(과 1)에 양의 확률로 도달할 수 있고, (b) penalty를 "크기"가 아니라
# "0이 아닐 확률"(L0-norm의 기댓값, 닫힌 형태로 미분 가능)에 매겨서, 불필요한
# head를 "작게" 만드는 게 아니라 "꺼지게" 만드는 쪽으로 직접 유도한다.
# ============================================================

_HC_BETA = 2.0 / 3.0   # temperature (논문 권장값, 별도 annealing 불필요 -- stretch+hard-clip 자체가 discrete화를 담당)
_HC_GAMMA = -0.1       # stretch 하한
_HC_ZETA = 1.1         # stretch 상한


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Loss / metrics — 전부 task_names/채널 인덱스 기준으로 범용 동작
# ============================================================

_POS_WEIGHTS: list[float] = []


def set_pos_weights(weights: list[float]) -> None:
    global _POS_WEIGHTS
    _POS_WEIGHTS = list(weights)


def estimate_pixel_pos_weights(
    dataset: "MultiTaskCsvDataset", max_samples: int = 3000, clamp_max: float = 200.0
) -> list[float]:
    """train 마스크 일부를 샘플링해 태스크별 픽셀 단위 배경:전경 비율을 추정한다.
    학습 시작 전 한 번만 계산해서 고정값으로 쓴다. 반환 순서는 dataset.task_names와 동일."""
    n = len(dataset.data)
    sample_n = min(max_samples, n)
    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=sample_n, replace=False)

    pos_pixels = [0] * dataset.num_tasks
    total_pixels = [0] * dataset.num_tasks

    for idx in indices:
        row = dataset.data.iloc[int(idx)]
        for t, mask_col in enumerate(dataset.mask_cols):
            mask = np.array(Image.open(row[mask_col]).convert("L")) > 0
            pos_pixels[t] += int(mask.sum())
            total_pixels[t] += mask.size

    weights = []
    for t in range(dataset.num_tasks):
        pos = max(pos_pixels[t], 1)
        neg = max(total_pixels[t] - pos, 1)
        weights.append(float(min(neg / pos, clamp_max)))
    return weights


def _positive_dice_loss_1ch(probs_c: torch.Tensor, targets_c: torch.Tensor) -> torch.Tensor:
    """(B,H,W) 단일 채널에 대해, 양성 샘플에서만 계산하는 Dice loss."""
    dims = (1, 2)
    positive = targets_c.sum(dims) > 0
    if not positive.any():
        return probs_c.sum() * 0.0

    p = probs_c[positive]
    t = targets_c[positive]
    intersection = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def combined_loss_multitask(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits/targets: (B, num_tasks, H, W). 채널별 고정 pos_weight BCE +
    채널별 positive-only Dice를 전부 더한다. _POS_WEIGHTS 길이가 채널 수와
    맞아야 함 (set_pos_weights로 미리 설정)."""
    num_tasks = logits.shape[1]
    if len(_POS_WEIGHTS) != num_tasks:
        raise RuntimeError(
            f"_POS_WEIGHTS 길이({len(_POS_WEIGHTS)})가 채널 수({num_tasks})랑 안 맞음. "
            "set_pos_weights()를 먼저 호출했는지 확인하세요."
        )

    pos_weight = torch.tensor(_POS_WEIGHTS, device=logits.device, dtype=logits.dtype).view(1, num_tasks, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    probs = torch.sigmoid(logits)
    dice_total = probs.sum() * 0.0
    for c in range(num_tasks):
        dice_total = dice_total + _positive_dice_loss_1ch(probs[:, c], targets[:, c])

    return bce + dice_total


@torch.no_grad()
def positive_segmentation_metrics_multitask(
    logits: torch.Tensor, targets: torch.Tensor, task_names: list[str]
) -> dict[str, tuple[float | None, float | None, int]]:
    """태스크별(task_names 순서)로 양성 샘플만 골라 Dice/IoU 계산.
    반환: {task_name: (dice, iou, n_pos)}"""
    pred = (torch.sigmoid(logits) >= 0.5).float()

    result: dict[str, tuple[float | None, float | None, int]] = {}
    for c, name in enumerate(task_names):
        pred_c = pred[:, c]
        target_c = targets[:, c]
        target_sum = target_c.sum((1, 2))
        positive_mask = target_sum > 0
        n_pos = int(positive_mask.sum().item())

        if n_pos == 0:
            result[name] = (None, None, 0)
            continue

        intersection = (pred_c * target_c).sum((1, 2))
        dice = (2.0 * intersection + 1e-7) / (pred_c.sum((1, 2)) + target_c.sum((1, 2)) + 1e-7)
        union = pred_c.sum((1, 2)) + target_c.sum((1, 2)) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)

        result[name] = (dice[positive_mask].mean().item(), iou[positive_mask].mean().item(), n_pos)

    return result


@dataclass
class MultiTaskEpochResult:
    loss: float
    task_dice: dict[str, float] = field(default_factory=dict)
    task_iou: dict[str, float] = field(default_factory=dict)


# ============================================================
# Head-diversity 정규화 (선택적) -- "무엇을 보라"가 아니라 "서로 다르게 보라"만
# 압박한다. compare_attention_unet_independent_heads.py(LiTS 단일 태스크)에
# 있던 것과 동일한 구현을 그대로 가져온 것 -- attention map 자체만 보고 계산하는
# 함수라 task_names/num_tasks와 무관하게 이 범용 모듈에 둬도 된다.
# ============================================================

def head_diversity_penalty(alphas: list[torch.Tensor]) -> torch.Tensor:
    """디코더 각 스테이지에서 나온 alpha(멀티헤드 attention map, (B, num_heads, H, W))
    들에 대해, head 쌍끼리 평균 코사인 유사도를 구해 전체 스테이지 평균을 반환.
    학습 loss에 이 값을 더하면 head들이 서로 달라지도록(코사인 유사도를 낮추도록)
    유도한다 -- 단, "뭘 봐야 하는지"는 전혀 지정하지 않으므로 여전히 관찰 대상.
    head가 1개뿐인 스테이지는 계산할 pair가 없어 건너뛴다."""
    stage_sims = []
    for alpha in alphas:
        num_heads = alpha.shape[1]
        if num_heads < 2:
            continue
        flat = alpha.flatten(start_dim=2)  # (B, num_heads, H*W)
        norm = flat / (flat.norm(dim=2, keepdim=True) + 1e-8)
        sim_matrix = torch.matmul(norm, norm.transpose(1, 2))  # (B, num_heads, num_heads)
        iu = torch.triu_indices(num_heads, num_heads, offset=1)
        pair_sims = sim_matrix[:, iu[0], iu[1]]  # (B, num_pairs)
        stage_sims.append(pair_sims.mean())

    if not stage_sims:
        return torch.zeros((), device=alphas[0].device if alphas else "cpu")
    return torch.stack(stage_sims).mean()


def hard_organ_routing_loss(
    model: nn.Module,
    masks: torch.Tensor,
    task_names: list[str],
    organ_names: list[str],
    stage_names: list[str],
) -> torch.Tensor:
    """Hard-Organ Routing Confidence Loss.

    작은/어려운 장기(organ_names, 예: gall_bladder/pancreas/left_adrenal_gland/
    right_adrenal_gland)의 GT 마스크 영역 "안에서만", 지정한 decoder stage
    (stage_names, 예: ["dec3"])의 competitive router가 만든 r_h(p)의 정규화
    entropy를 최소화한다.

        L_hard = (1/|stage_names|) * sum_stage [
                     (1/|organ_names|) * sum_organ [ 그 장기 영역 내 r의 평균 정규화 entropy ]
                 ]

    설계 의도: "이 영역에서는 여러 head를 애매하게(예: [.30,.25,.25,.20]) 조금씩
    쓰지 말고 필요한 head를 더 명확하게(예: [.05,.80,.10,.05]) 선택해라"만 요구하고,
    *어느* head를 선택할지는 지정하지 않는다 -- seed42/43에서 dec별 head-organ 배정
    자체가 서로 다르게 symmetry-breaking됐던 것과 정합적인 설계(고정 head 지정은
    두 seed 중 하나에서는 반드시 틀린 head를 지정하는 꼴이 됨).

    적용 범위를 organ_names/stage_names에 명시한 것으로만 좁힌다 -- 다른 장기/stage의
    routing은 지금처럼 segmentation loss + balance loss만 받고 이 loss의 영향을 전혀
    안 받는다(routing 의존도가 실제로 큰 곳에만 추가 신호를 준다는 uniform-router
    ablation 결과에 기반한 설계: 특히 small organ이 uniform-router ablation에서 가장
    크게 떨어졌음).

    반드시 gate.last_r_for_loss(미분 가능)를 쓴다 -- gate.last_r은 진단용으로
    detach돼 있어서 이걸로 loss를 만들면 gradient가 psi_router까지 전혀 안 들어감.
    alpha_h/sum(alpha_h)로 r을 재구성하는 방식도 쓰지 않는다(s≈0 근처 수치 불안정
    이력, head-organ 분석 v2/v3에서 확인됨) -- softmax 출력을 그대로 사용.

    나중에 stage_names에 dec2/dec4 등을 추가하면(예: ["dec2","dec3"]) 별도 코드 수정
    없이 그대로 확장 가능."""
    organ_indices = []
    for organ in organ_names:
        if organ not in task_names:
            raise ValueError(f"hard_organ_routing_loss: organ_names의 '{organ}'이 task_names에 없음")
        organ_indices.append(task_names.index(organ))

    stage_losses = []
    for stage_name in stage_names:
        stage = getattr(model, stage_name, None)
        if stage is None:
            raise ValueError(f"hard_organ_routing_loss: 모델에 '{stage_name}' 속성이 없음")
        gate = stage.gate
        r = getattr(gate, "last_r_for_loss", None)
        if r is None:
            raise RuntimeError(
                f"{stage_name}.gate에 last_r_for_loss가 없음 -- CompetitiveMultiHeadAttentionGate가 "
                f"아니거나(ChannelSplit 등 다른 gate_type), model(images)를 먼저 호출하지 않은 상태. "
                f"hard_organ_routing_loss는 competitive_moe 게이트 전용이고, 반드시 forward pass 이후에 "
                f"호출해야 함."
            )
        num_heads = r.shape[1]
        if num_heads < 2:
            continue
        h, w = r.shape[-2:]
        entropy_map = -(r * torch.log(r.clamp_min(1e-8))).sum(dim=1, keepdim=True) / math.log(num_heads)  # (B,1,h,w), 0~1

        organ_losses = []
        for organ_idx in organ_indices:
            organ_mask = masks[:, organ_idx : organ_idx + 1, :, :]  # (B,1,H,W), 원본 해상도
            valid = organ_mask.sum(dim=(1, 2, 3)) > 0  # 원본 해상도 기준 양성 샘플만(다운샘플로 인한 손실 방지)
            if not valid.any():
                continue
            mask_resized = F.interpolate(organ_mask, size=(h, w), mode="area")  # (B,1,h,w), 0~1 fractional coverage
            mask_sum = mask_resized.sum(dim=(2, 3)).clamp_min(1e-6)  # (B,1)
            per_image_entropy = (entropy_map * mask_resized).sum(dim=(2, 3)) / mask_sum  # (B,1)
            organ_losses.append(per_image_entropy.squeeze(1)[valid].mean())

        if organ_losses:
            stage_losses.append(torch.stack(organ_losses).mean())  # 장기별 동일 가중치(픽셀 수 무관)

    if not stage_losses:
        return torch.zeros((), device=masks.device)
    return torch.stack(stage_losses).mean()  # stage별 동일 가중치


def dilate_mask(mask: torch.Tensor, px: int) -> torch.Tensor:
    """(B,1,H,W) 이진/실수 마스크를 max-pool로 px 픽셀만큼 팽창시킨다. ROI를
    타겟 픽셀에 딱 맞게 좁게 잡으면 coarse decoder 스테이지(예: 14x14)에서 아예
    사라져버릴 수 있어서, 약간 여유를 주기 위한 용도."""
    if px <= 0:
        return mask
    k = 2 * px + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=px)


def roi_aware_head_diversity_penalty(
    alphas: list[torch.Tensor],
    roi_mask: torch.Tensor,
    min_roi_frac: float = 0.001,
) -> torch.Tensor:
    """head_diversity_penalty의 ROI 제한 버전. 전체 이미지에서 "서로 달라지라"고
    하면 모델이 이미지 구석/고정된 점처럼 segmentation과 무관한 위치로 도망가는
    trivial한 방법으로 만족시켜버릴 수 있다(실제로 disc/cup H8+diversity 실험에서
    관찰됨: cosine sim 0.9917->0.0032로 확실히 갈라졌지만 attention이 disc/cup이
    아니라 이미지 모서리·고정된 점으로 쏠렸고 dice는 그대로였음). 그래서 관심
    영역(roi_mask, 보통 disc 마스크 -- 필요하면 dilate_mask로 약간 팽창) 안에서만
    코사인 유사도를 계산해서, "관련 영역 안에서 서로 다른 걸 보라"고 좁혀서
    지시한다.

    roi_mask: (B, 1, H_orig, W_orig), 0~1. GT 마스크는 학습(이 penalty 계산)
    에만 쓰고 추론 시점에는 전혀 필요 없다 -- forward pass 자체는 그대로 image만
    입력받아 alpha를 만들고, 이 함수가 그 alpha를 ROI로 마스킹해서 loss만 계산.

    각 디코더 스테이지 해상도로 roi_mask를 area-보간(부분적으로 겹쳐도 완전히
    0으로 사라지지 않도록) 리사이즈하고, ROI 커버리지가 너무 작은 샘플(이미지)은
    그 스테이지에서 제외한다(min_roi_frac).

    주의(중요한 편법 하나): 이 penalty만 단독으로 쓰면, 어떤 head가 ROI 안에서
    attention을 거의 0으로 꺼버려도 코사인 유사도가 낮게(=페널티가 작게) 나올
    수 있다 -- "관련 영역 안에서 서로 다르게 보라"가 아니라 "관련 영역을 아예
    안 보면 유사도가 낮아진다"는 편법. F.normalize(eps 아주 작게)로 epsilon이
    벡터를 인위적으로 죽이는 문제는 줄였지만, 진짜로 attention이 거의 0인
    경우까지 막아주진 않는다 -- 그래서 별도로 roi_attention_relevance_penalty()
    를 두었다(필요하면 같이 쓸 것). 두 개를 한꺼번에 켜지 말고, 먼저 이 penalty만
    켜서 실제로 이 편법이 관찰되는지 확인한 뒤 relevance를 추가하는 순서를
    권장한다(head별 ROI 내부 평균 attention을 학습/분석 때 같이 로그로 남겨서
    판단)."""
    stage_sims = []
    for alpha in alphas:
        num_heads = alpha.shape[1]
        if num_heads < 2:
            continue
        h, w = alpha.shape[-2:]
        roi_resized = F.interpolate(roi_mask, size=(h, w), mode="area")  # (B,1,h,w), 0~1 커버리지
        roi_frac = roi_resized.mean(dim=(1, 2, 3))  # (B,)
        valid = roi_frac > min_roi_frac
        if valid.sum() == 0:
            continue

        masked_alpha = alpha * roi_resized  # (B, num_heads, h, w), 브로드캐스트
        flat = masked_alpha.flatten(start_dim=2)  # (B, num_heads, h*w)
        norm = F.normalize(flat, p=2, dim=2, eps=1e-12)
        sim_matrix = torch.matmul(norm, norm.transpose(1, 2))  # (B, num_heads, num_heads)
        iu = torch.triu_indices(num_heads, num_heads, offset=1)
        pair_sims = sim_matrix[:, iu[0], iu[1]]  # (B, num_pairs)

        pair_sims = pair_sims[valid]
        if pair_sims.numel() == 0:
            continue
        stage_sims.append(pair_sims.mean())

    if not stage_sims:
        return torch.zeros((), device=alphas[0].device if alphas else "cpu")
    return torch.stack(stage_sims).mean()


def gated_head_diversity_penalty(
    alphas: list[torch.Tensor],
    roi_mask: torch.Tensor,
    gate_values: torch.Tensor,
    min_roi_frac: float = 0.001,
) -> torch.Tensor:
    """roi_aware_head_diversity_penalty의 학습형 gate 버전 (Learnable Head Selection,
    g_h=sigmoid(s_h) 실험용). head pair (i,j)의 ROI 내부 코사인 유사도에
    g_i*g_j를 곱해서 가중 평균한다.

    왜 필요한가: g_h가 0에 가까워지도록(=그 head를 끄도록) sparsity penalty가
    유도하는데, 그러면 해당 head의 alpha는 거의 0벡터가 된다. F.normalize가
    이런 near-zero 벡터를 정규화하면 "방향"이 수치 노이즈에 좌우돼서 코사인
    유사도가 의미 없이 튈 수 있다(ROI 다양성 실험 초기에 겪었던 것과 같은 부류의
    문제). g_i*g_j로 가중하면 이미 꺼지기로 한 head가 낀 pair는 diversity loss에
    거의 기여를 안 하게 돼서, 이 노이즈가 gradient에 실제 신호처럼 섞여 들어가는
    걸 막는다 -- "꺼진 head끼리 우연히 비슷/다르게 보이는 것"이 학습에 영향을
    주지 않게 하려는 목적.

    relevance penalty(roi_attention_relevance_penalty)는 이 실험에서는 같이 쓰지
    않는다 -- "모든 head가 최소한 이 정도는 봐라"라는 강제가 "필요 없으면 꺼라"는
    sparsity penalty(lambda_head * sum(g_h))와 정확히 반대 방향으로 당기기
    때문이다. g_h -> 0은 이 실험에서는 실패가 아니라 의도된 선택 결과다."""
    num_heads = gate_values.shape[0]
    iu = torch.triu_indices(num_heads, num_heads, offset=1)
    pair_gate_weight = gate_values[iu[0]] * gate_values[iu[1]]  # (num_pairs,)
    weight_sum = pair_gate_weight.sum().clamp_min(1e-8)

    stage_sims = []
    for alpha in alphas:
        h, w = alpha.shape[-2:]
        roi_resized = F.interpolate(roi_mask, size=(h, w), mode="area")
        roi_frac = roi_resized.mean(dim=(1, 2, 3))
        valid = roi_frac > min_roi_frac
        if valid.sum() == 0:
            continue

        masked_alpha = alpha * roi_resized
        flat = masked_alpha.flatten(start_dim=2)
        norm = F.normalize(flat, p=2, dim=2, eps=1e-12)
        sim_matrix = torch.matmul(norm, norm.transpose(1, 2))
        pair_sims = sim_matrix[:, iu[0], iu[1]]  # (B, num_pairs)
        pair_sims = pair_sims[valid]
        if pair_sims.numel() == 0:
            continue

        weighted = (pair_sims * pair_gate_weight.unsqueeze(0)).sum(dim=1) / weight_sum
        stage_sims.append(weighted.mean())

    if not stage_sims:
        return torch.zeros((), device=alphas[0].device if alphas else "cpu")
    return torch.stack(stage_sims).mean()


def gated_relevance_penalty(
    alphas: list[torch.Tensor],
    roi_mask: torch.Tensor,
    gate_values: torch.Tensor,
    min_attention: float = 0.2,
) -> torch.Tensor:
    """roi_attention_relevance_penalty의 gate-가중 버전 (Learnable Head Selection,
    gated_head_diversity_penalty와 세트로 쓰는 용도). head별 ROI 내부 평균
    attention이 min_attention보다 낮으면 그 부족분만큼 페널티를 매기되, 그
    부족분에 해당 head 자신의 gate 값 g_h를 곱한다.

    relevance penalty를 gate 없이 그대로 걸면 "필요 없어서 gate가 낮아진 head"
    까지 억지로 최소 attention을 유지하라고 강제하게 되어 sparsity penalty
    (lambda_head * sum(g_h))와 정반대로 충돌한다("꺼져도 된다" vs "그래도 최소한은
    봐라"). g_h로 가중하면 이미 gate가 낮은(=쓸모없다고 판단된) head는 relevance
    요구도 자동으로 같이 낮아져서 sparsity와 충돌하지 않는다.

    반대로 gate가 아직 높은(="쓰겠다"고 판단된) head가, gate는 안 낮추면서 ROI
    내부 attention만 죽여서 diversity loss를 값싸게 회피하는 건 이 penalty가
    막아준다 -- gated_head_diversity_penalty를 relevance 없이 단독으로 썼을 때
    실제로 관찰된 실패 모드(gate는 0.88~0.90대로 거의 안 움직이는데 ROI 코사인은
    몇 epoch 만에 0.007까지 붕괴)에 대한 직접적인 대응책.

    참고: alpha는 이미 forward에서 g_h가 곱해진 상태로 들어오기 때문에(모델의
    active_mask=head_gate_values() 배선), roi_attention 자체가 이미 대략 g_h에
    비례해서 낮게 나온다 -- 그래서 gate가 낮은 head는 "부족분(shortfall)"이 이미
    반쯤은 자연스럽게 발생하고, 거기에 g_h를 한 번 더 곱해서 최종 페널티를
    확실하게 작게 만든다."""
    penalties = []
    for alpha in alphas:
        h, w = alpha.shape[-2:]
        roi_resized = F.interpolate(roi_mask, size=(h, w), mode="area")
        roi_sum = roi_resized.sum(dim=(2, 3)).clamp_min(1e-6)
        roi_attention = (alpha * roi_resized).sum(dim=(2, 3)) / roi_sum  # (B, num_heads)
        shortfall = F.relu(min_attention - roi_attention)  # (B, num_heads)
        gated_shortfall = shortfall * gate_values.unsqueeze(0)  # (B, num_heads), 브로드캐스트
        penalties.append(gated_shortfall.mean())
    return torch.stack(penalties).mean()


def roi_attention_relevance_penalty(
    alphas: list[torch.Tensor],
    roi_mask: torch.Tensor,
    min_attention: float = 0.2,
) -> torch.Tensor:
    """roi_aware_head_diversity_penalty와 세트로 쓰는 보조 penalty. head가 ROI
    안에서 attention을 아예 꺼버려서(=거의 0으로 만들어서) 코사인 유사도를 낮추는
    편법을 막기 위한 것 -- "관련 영역은 최소한 어느 정도 봐라"만 강제하고, 그
    안에서 뭘 어떻게 나눠 보는지는 지정하지 않는다(여전히 대칭/관찰 목적 유지).

    head별로 ROI 내부 평균 attention이 min_attention보다 낮으면 그 부족분만큼
    페널티. min_attention은 예시값이라 논문의 핵심 숫자로 삼지 말 것 -- 처음엔
    diversity penalty만 켜서 이 편법이 실제로 관찰되는지 먼저 확인하고, 관찰되면
    그때 이 함수를 추가하는 순서를 권장(한 번에 여러 개 바꾸면 뭐가 효과를
    냈는지 알 수 없다)."""
    penalties = []
    for alpha in alphas:
        h, w = alpha.shape[-2:]
        roi_resized = F.interpolate(roi_mask, size=(h, w), mode="area")  # (B,1,h,w)
        roi_sum = roi_resized.sum(dim=(2, 3)).clamp_min(1e-6)  # (B,1)
        roi_attention = (alpha * roi_resized).sum(dim=(2, 3)) / roi_sum  # (B, num_heads)
        penalty = F.relu(min_attention - roi_attention).mean()
        penalties.append(penalty)
    return torch.stack(penalties).mean()


@torch.no_grad()
def active_head_cosine_similarity(
    alphas: list[torch.Tensor], active_head_idx: list[int]
) -> float | None:
    """head_diversity_penalty의 "활성 head만" 버전. Adaptive Progressive Multi-Head
    실험에서 현재 켜져 있는 head들끼리만 평균 코사인 유사도를 재려는 용도 -- 꺼진
    head는 alpha가 이미 (active_mask로) 0이라, 8개 채널을 그냥 다 넣고 코사인을
    재면 "꺼진 head들끼리는 완전히 똑같이 0"이라는 사실 때문에 유사도가 왜곡된다.
    그래서 active_head_idx로 지정한 채널만 골라서 계산한다.

    redundancy 트리거(현재 활성 head들이 서로 너무 비슷한 걸 보고 있는가) 판단에
    이 값을 쓴다. head가 1개 이하로 활성화된 경우 pair가 없어 None을 반환한다."""
    if len(active_head_idx) < 2:
        return None
    idx = torch.tensor(active_head_idx, device=alphas[0].device)
    stage_sims = []
    for alpha in alphas:
        sub = alpha.index_select(1, idx)  # (B, n_active, H, W)
        flat = sub.flatten(start_dim=2)
        norm = F.normalize(flat, p=2, dim=2, eps=1e-12)
        sim_matrix = torch.matmul(norm, norm.transpose(1, 2))
        n = len(active_head_idx)
        iu = torch.triu_indices(n, n, offset=1)
        pair_sims = sim_matrix[:, iu[0], iu[1]]
        stage_sims.append(pair_sims.mean())
    return torch.stack(stage_sims).mean().item()


@torch.no_grad()
def roi_attention_diagnostics(alphas: list[torch.Tensor], roi_mask: torch.Tensor) -> list[float]:
    """디버깅/분석용: 디코더 스테이지 전체 평균으로, head별 ROI 내부 평균
    attention 크기를 리스트로 반환(head 순서대로). 학습 루프 로그나 분석
    스크립트에서 "이 head가 ROI를 아예 꺼버렸는지"를 바로 확인할 때 쓴다."""
    stage_means = []
    for alpha in alphas:
        h, w = alpha.shape[-2:]
        roi_resized = F.interpolate(roi_mask, size=(h, w), mode="area")
        roi_sum = roi_resized.sum(dim=(2, 3)).clamp_min(1e-6)
        roi_attention = (alpha * roi_resized).sum(dim=(2, 3)) / roi_sum  # (B, num_heads)
        stage_means.append(roi_attention.mean(dim=0))  # (num_heads,) -- 배치 평균
    return torch.stack(stage_means).mean(dim=0).tolist()  # 스테이지 평균 -> head별 리스트


# ============================================================
# Stage 1 SCHR (Shared-Competitive Head Routing) -- Learnable Head Selection
# 라인(sigmoid+L1, Hard-Concrete 둘 다) 전부 "8개 head 중 몇 개를 전역적으로
# 켜고 끌지"를 학습시켰는데, lambda_head를 5배(0.02->0.1)까지 올려도 8개 head가
# 거의 항상 다 같이 살아남는 uniform shrinkage로 수렴했다(diffuse, 균등한 수축 --
# 특정 head가 discrete하게 꺼지지 않음. 두 gate 메커니즘·두 lambda 모두 동일한
# 결론). "이 데이터셋에 몇 개의 head가 필요한가"라는 전역 스칼라 질문 자체가
# 이 구조에 안 맞는 질문이었을 수 있다 -- disc/cup 두 태스크가 서로 다른 head
# 조합을 선호할 수 있는데, 전역 gate 하나로는 "두 태스크 요구의 평균"만 나올 수
# 밖에 없어서 애초에 discrete하게 갈라질 이유가 없었을 수 있음.
#
# Stage 1은 그래서 "head를 켜고 끄는" 대신 "태스크마다 같은 head pool을 어떻게
# 섞어 쓰는지"를 학습시킨다(MMoE, Ma et al. 2018의 task-specific gating과 동일한
# 구조). head 자체(theta_x/phi_g/psi_heads)는 여전히 태스크 공용 "shared expert"로
# 남겨두고, 마지막 디코더 스테이지에서만 태스크별 라우팅 가중치
# R_{t,h}=softmax_h(S_{t,h}/tau)로 섞는다. 그 앞단(encoder, dec4/dec3/dec2)은
# Fixed H{num_heads} 베이스라인과 구조가 완전히 동일하게 둬서, 라우팅이 실제로
# 뭔가를 바꾸는지만 분리해서 관찰한다.
#
# 주의(수학적으로 짚고 넘어간 부분): "두 태스크가 head 하나를 놓고 진짜로
# 경쟁한다"는 걸 R을 head 축(column)으로 정규화해서 강제하는 안도 검토했지만
# (예: R_{t,h}=softmax_t(S_{t,h}/tau)), 이건 gate로서 의미가 이상해진다 -- 한
# 태스크가 특정 head를 많이 쓰는 게 다른 태스크가 그 head를 쓸 수 있는 양을
# 수학적으로 깎아먹어야 할 이유가 없다(같은 feature map을 두 태스크가 "나눠
# 갖는" 게 아니라 "각자 원하는 만큼 읽는" 구조이기 때문). 그래서 R은 표준 MoE
# gate처럼 태스크(row)별로 정규화하고, "쏠림"에 대한 비용은 아래
# load_balance_loss()(Shazeer et al. 2017 스타일)로 별도로 준다 -- 명시적
# 제약이 아니라 학습 신호로.
# ============================================================


class TaskAdaptiveHeadRouter(nn.Module):
    """공유 head pool 위에서 태스크별로 다른 혼합 가중치를 학습하는 라우팅
    레이어. head 자체의 파라미터(attention 계산)는 전혀 안 건드리고, 이미
    계산된 head별 결과를 태스크마다 다르게 가중합만 한다."""

    def __init__(self, num_tasks: int, num_heads: int, tau: float = 1.0) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.num_heads = num_heads
        self.tau = tau
        # 0으로 초기화 -> softmax 후 모든 head에 균등(1/num_heads) 가중치로 시작.
        # Fixed H{num_heads} 베이스라인과 최대한 비슷한 지점에서 출발시켜서,
        # 라우팅이 갈라진다면 그게 초기화 때문이 아니라 학습 신호 때문이라는 걸
        # 보장하려는 목적.
        self.routing_logits = nn.Parameter(torch.zeros(num_tasks, num_heads))

    def routing_weights(self) -> torch.Tensor:
        """(num_tasks, num_heads), 태스크(row)별로 합=1인 표준 MoE gate."""
        return F.softmax(self.routing_logits / self.tau, dim=1)

    def combine(self, head_splits: list[torch.Tensor]) -> list[torch.Tensor]:
        """head_splits: 이미 alpha로 gating된 head별 (B, C_h, h, w) 텐서 리스트
        (길이 num_heads). 태스크별로 R[t,:] 가중합해서 이어붙인(concat) 텐서를
        반환한다(길이 num_tasks 리스트, 각각 fusion conv 입력과 같은 채널 수).

        routing_weights()는 head 축으로 softmax(합=1)라서, uniform 초기화
        [1/H,...,1/H]를 그대로 곱하면 head별 기여가 처음부터 Fixed H{num_heads}
        (곱 없이 그대로 concat) 대비 1/H로 줄어든 채로 시작한다 -- concat이라서
        softmax 합=1이 "가중 평균"처럼 스케일을 보존해주지 않기 때문(weighted
        sum이었다면 문제없었을 것). 그래서 여기서만 num_heads를 곱해 uniform
        라우팅일 때 각 head가 x1로 그대로 들어가게 만든다(균등 분배 시
        Fixed H{num_heads}와 정확히 같은 스케일에서 학습을 시작하기 위함) --
        해석/로깅용 routing_weights()(합=1 분포, cosine·entropy 계산에 사용)는
        그대로 두고, forward 경로에서만 스케일을 보정한다."""
        R = self.routing_weights() * self.num_heads
        combined_per_task = []
        for t in range(self.num_tasks):
            weighted = [head_splits[h] * R[t, h] for h in range(self.num_heads)]
            combined_per_task.append(torch.cat(weighted, dim=1))
        return combined_per_task

    def load_balance_loss(self) -> torch.Tensor:
        """head별 총 사용량(importance_h = sum_t R[t,h])의 변동계수 제곱(CV^2).
        Shazeer et al. 2017(Outrageously Large Neural Networks)의 load-balancing
        loss와 동일한 형태 -- 모든 head가 태스크 전체에 걸쳐 고르게 쓰이면 0에
        가깝고, 소수 head에 라우팅이 쏠리면 커진다. "두 태스크가 서로 다른 head를
        쓰라"고 직접 강요하는 게 아니라 "일부 head가 완전히 방치되지는 말라"는
        훨씬 약한 제약 -- task-specialization 자체는 이 항이 전혀 막지 않는다."""
        R = self.routing_weights()
        importance = R.sum(dim=0)  # (num_heads,)
        mean = importance.mean()
        var = importance.var(unbiased=False)
        return var / (mean ** 2 + 1e-8)


class TaskRoutedDecoderStage(nn.Module):
    """마지막 디코더 스테이지 전용. ChannelSplitMultiHeadAttentionGate와 동일한
    head별 attention 계산(theta_x/phi_g/psi_heads, 전부 태스크 공유)을 그대로
    두되, head-split을 fusion conv에 넣기 직전 지점에서 TaskAdaptiveHeadRouter로
    태스크별로 다르게 섞는다. fusion conv와 그 뒤 ConvBlock도 태스크 간 완전히
    공유(가중치 공유) -- Stage 1에서는 "라우팅 가중치가 다르다"는 것 하나만
    태스크 간 차이로 두고, 그 효과만 분리해서 관찰한다(파라미터 자체를 태스크별로
    분기하는 shared/task-specific 분해는 여기서 하지 않음 -- 필요하면 Stage 2에서
    검토)."""

    def __init__(
        self,
        decoder_in: int,
        skip_channels: int,
        out_channels: int,
        num_heads: int,
        num_tasks: int,
        router_tau: float = 1.0,
    ) -> None:
        super().__init__()
        inter_channels = max(skip_channels // 2, 16)
        if skip_channels % num_heads != 0 or inter_channels % num_heads != 0:
            raise ValueError(
                f"skip_channels({skip_channels}) / inter_channels({inter_channels})가 "
                f"num_heads({num_heads})로 나눠떨어지지 않음"
            )
        self.num_heads = num_heads
        self.num_tasks = num_tasks
        self.theta_x = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi_g = nn.Conv2d(decoder_in, inter_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        inter_per_head = inter_channels // num_heads
        self.psi_heads = nn.ModuleList([nn.Conv2d(inter_per_head, 1, 1) for _ in range(num_heads)])
        self.fusion = nn.Conv2d(skip_channels, skip_channels, 1, bias=False)  # 태스크 공유
        self.router = TaskAdaptiveHeadRouter(num_tasks, num_heads, tau=router_tau)
        self.conv = ConvBlock(decoder_in + skip_channels, out_channels)  # 태스크 공유

    def forward(
        self, decoder_feature: torch.Tensor, skip: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        x_proj = self.theta_x(skip)
        g_proj = self.phi_g(decoder_feature)
        g_proj = F.interpolate(g_proj, size=x_proj.shape[-2:], mode="bilinear", align_corners=False)
        combined = self.relu(x_proj + g_proj)

        combined_splits = combined.chunk(self.num_heads, dim=1)
        x_splits = skip.chunk(self.num_heads, dim=1)

        gated_splits, alphas = [], []
        for i in range(self.num_heads):
            alpha_h = torch.sigmoid(self.psi_heads[i](combined_splits[i]))
            gated_splits.append(x_splits[i] * alpha_h)
            alphas.append(alpha_h)
        alpha_all = torch.cat(alphas, dim=1)  # (B, num_heads, h, w) -- 기존 diversity penalty와 호환

        combined_per_task = self.router.combine(gated_splits)  # list[num_tasks] of (B, skip_channels, h, w)

        decoder_feature_up = F.interpolate(
            decoder_feature, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )

        outs = []
        for t in range(self.num_tasks):
            fused_t = self.fusion(combined_per_task[t])
            out_t = self.conv(torch.cat([decoder_feature_up, fused_t], dim=1))
            outs.append(out_t)
        return outs, alpha_all


class AttentionUNetResNet34MultiTaskSCHR(nn.Module):
    """Stage 1 SCHR (Shared-Competitive Head Routing). encoder/dec4/dec3/dec2는
    Fixed H{num_heads} 베이스라인과 구조가 완전히 동일하고, 마지막 스테이지
    (dec1)에서만 TaskRoutedDecoderStage로 태스크별 head 라우팅을 학습한다.
    최종 1x1 conv도 태스크별로 분리하되(각 태스크 전용 채널 1개), 그 앞의
    ConvBlock(32,32) 정제 레이어는 태스크 간 공유 -- 기존
    self.head = ConvBlock(32,32)+Conv2d(32,num_tasks,1)와 실질적으로 동일한
    용량이고(Conv2d(32,num_tasks,1) 자체가 이미 태스크별로 독립적인 1x1 conv를
    이어붙인 것과 수학적으로 같음), 새로 추가되는 용량은 라우터 파라미터
    (num_tasks*num_heads, 아주 작음)뿐이다."""

    def __init__(
        self,
        task_names: list[str],
        num_heads: int = 8,
        imagenet_pretrained: bool = True,
        router_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.task_names = list(task_names)
        num_tasks = len(task_names)
        self.num_tasks = num_tasks
        self.encoder = ResNet34Encoder(imagenet_pretrained=imagenet_pretrained)

        self.dec4 = DecoderStage(512, 256, 256, "multi_split", num_heads)
        self.dec3 = DecoderStage(256, 128, 128, "multi_split", num_heads)
        self.dec2 = DecoderStage(128, 64, 64, "multi_split", num_heads)
        self.dec1 = TaskRoutedDecoderStage(
            64, 64, 32, num_heads=num_heads, num_tasks=num_tasks, router_tau=router_tau
        )

        self.shared_refine = ConvBlock(32, 32)  # 태스크 공유
        self.task_heads = nn.ModuleList([nn.Conv2d(32, 1, kernel_size=1) for _ in range(num_tasks)])

    def routing_weights(self) -> torch.Tensor:
        return self.dec1.router.routing_weights()

    def load_balance_loss(self) -> torch.Tensor:
        return self.dec1.router.load_balance_loss()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        input_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encoder(x)

        d4, a4 = self.dec4(x4, x3)
        d3, a3 = self.dec3(d4, x2)
        d2, a2 = self.dec2(d3, x1)
        d1_per_task, a1 = self.dec1(d2, x0)  # list[num_tasks], (B, num_heads, h, w)

        task_logits = []
        for t in range(self.num_tasks):
            refined_t = self.shared_refine(d1_per_task[t])
            logit_t = self.task_heads[t](refined_t)
            task_logits.append(logit_t)
        logits = torch.cat(task_logits, dim=1)  # (B, num_tasks, H, W)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits, [a1, a2, a3, a4]


@torch.no_grad()
def task_routing_cosine_similarity(routing_weights: torch.Tensor) -> float:
    """routing_weights: (num_tasks, num_heads). 태스크 쌍끼리 평균 코사인 유사도
    -- 기존 head_diversity_penalty와 정확히 대칭되는 지표지만, 이번엔 "head들이
    서로 다른가"가 아니라 "태스크들이 head를 서로 다르게 쓰는가"를 잰다. 1에
    가까우면 두 태스크가 사실상 같은 라우팅(=head selection 라인에서 봤던 것과
    동일하게 결국 하나로 수렴), 낮을수록 진짜 task-specialization이 생긴 것."""
    num_tasks = routing_weights.shape[0]
    if num_tasks < 2:
        raise ValueError("태스크가 2개 이상이어야 pairwise cosine을 계산할 수 있음")
    norm = F.normalize(routing_weights, p=2, dim=1, eps=1e-12)
    sim_matrix = torch.matmul(norm, norm.transpose(0, 1))
    iu = torch.triu_indices(num_tasks, num_tasks, offset=1)
    return sim_matrix[iu[0], iu[1]].mean().item()


@torch.no_grad()
def task_routing_entropy(routing_weights: torch.Tensor) -> torch.Tensor:
    """태스크별 routing entropy H(R_t) = -sum_h R_{t,h} log R_{t,h}. shape:
    (num_tasks,). routing_cosine만으로는 "두 태스크의 라우팅이 서로 다른가"만
    보이고 "각 태스크가 head를 얼마나 좁게/넓게 쓰는가"는 안 보인다 -- 우리가
    실제로 원하는 결과("task-specific specialization")는 cosine이 낮으면서
    동시에 entropy도 낮은 경우(각 태스크가 좁은, 그리고 서로 다른 head 집합에
    집중)다. cosine만 낮고 entropy가 높으면 그냥 분포 모양이 우연히 다른
    것뿐일 수 있어서, 두 지표를 같이 봐야 구분된다."""
    R = routing_weights.clamp_min(1e-12)
    return -(R * R.log()).sum(dim=1)


@torch.no_grad()
def task_routing_entropy_normalized(routing_weights: torch.Tensor) -> torch.Tensor:
    """task_routing_entropy를 log(num_heads)로 나눠 [0,1]로 정규화한 버전 --
    1=완전 균등(모든 head를 똑같이 씀), 0=한 head에 완전히 집중. head 개수와
    무관하게 해석 가능해서(log(num_heads) 최댓값 기준 상대화) 비교/발표에 더
    직관적."""
    num_heads = routing_weights.shape[1]
    entropy = task_routing_entropy(routing_weights)
    return entropy / math.log(num_heads)
