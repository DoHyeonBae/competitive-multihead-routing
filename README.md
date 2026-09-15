# Competitive Multi-Head Routing for Abdominal Multi-Organ Segmentation

복부 다중 장기 분할에서 Competitive Multi-Head Routing의 Global–Foreground 활용 괴리 분석

## Overview

Multi-head attention 구조에서 head 수를 늘리는 것이 실제로 head 간 기능적 분업(functional specialization)을 보장하는지를 검증하기 위한 연구 코드입니다. AMOS22 복부 CT 12-organ segmentation을 대상으로, 위치별 head 간 경쟁적 routing과 population-level load-balancing을 결합한 **Competitive Multi-Head Attention Gate**를 Attention U-Net에 적용하고, H=1~4 등 다양한 multi-head 구성에서의 성능 및 head 활용 양상을 분석했습니다.

## Key Findings

| 항목 | 결과 |
|---|---|
| Mean Dice (6개 구성) | 0.7827 ~ 0.7911 (독립 seed 재학습 변동폭과 비슷한 수준) |
| Global head utilization imbalance | 1.29× |
| Foreground head utilization imbalance (before balancing) | 111.7× |
| Foreground head utilization imbalance (after FG-aware balancing) | 1.67× |
| Dominant-head removal ΔDice | −0.2976 → −0.0298 (balancing 적용 후) |
| Overall Dice 변화 (balancing 적용 전후) | 0.7737 → 0.7749 |

즉, 전체 픽셀 기준으로는 head 활용이 균등해 보여도(1.29×) 실제 장기 영역(foreground)에서는 극도로 편중되어 있었고(111.7×), foreground-aware balancing으로 이 불균형을 완화할 수 있었지만(1.67×) segmentation 정확도(Dice) 자체는 거의 변하지 않았습니다. 즉 **utilization의 균형화가 head의 장기별 기능적 분화까지 보장하지는 않는다**는 것이 핵심 결론입니다.


## Repository Structure

### Training
- `train_amos_competitive_gate.py` — 경쟁형(competitive) multi-head gate 학습
- `train_amos_fixed_heads.py` — 고정 head 구성 baseline 학습
- `train_amos_head_projection_gate.py` — head projection gate 기반 학습
- `train_amos_stage_adaptive_heads.py` — decoder stage별 적응형 head 수 학습
- `attention_unet_multitask.py` — Attention U-Net 기반 멀티태스크 모델 정의
- `merge_seed43_pair.py` — 복수 seed 결과 병합

### Analysis
- `analyze_foreground_background_balance.py` — Global vs. Foreground head utilization 비교
- `analyze_difficulty_rigorous.py` / `analyze_per_case_difficulty.py` / `verify_difficulty_is_real.py` — 장기별/케이스별 난이도 분석 및 검증
- `analyze_correlation_and_joint_removal.py` — head 활용도와 head-removal ΔDice 간 상관관계 분석
- `compute_organ_identity_reproducibility.py` — multi-seed 재현성 계산
- `measure_efficiency.py` — 연산 효율성 측정

### Diagnostics
- `diagnose_head_organ_specialization.py` — head별 장기 특화 양상 진단
- `diagnose_ablation_uniform_and_head_removal.py` — balancing/ablation 및 head 제거 실험
- `diagnose_content_vs_position.py` — content 기반 vs. 위치 기반 routing 특성 진단
- `diagnose_flip_equivariance.py` — 대칭 불변성 진단

### Visualization
- `visualize_competitive_gate.py` — head별 attention/routing map 시각화

### Utilities
- `ablation_utils.py` — ablation 실험 공통 유틸리티

### Data
- `csv/` — 실험 결과 및 분석 출력값 저장 폴더

## Environment

- Python 3.x, PyTorch
- Backbone: Attention U-Net (ResNet-34)
- Dataset: [AMOS22](https://amos22.grand-challenge.org/) (Abdominal CT, 300 volumes, 8–12 organs)

> 정확한 패키지 버전은 `requirements.txt`로 별도 정리 예정입니다.

## Author

배도현 (Do-Hyeon Bae) — [ehgusel0521@gmail.com](mailto:ehgusel0521@gmail.com)
