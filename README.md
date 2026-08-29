# 2026 ETRI 자율주행 AI 챌린지 — E2E Driving (VAD + LAW)

6개 카메라 영상으로 3D 인지(객체/차선)와 ego 경로 계획을 함께 학습하는 End-to-End
자율주행 모델. VAD-tiny를 ETRI 도메인에 맞게 2단계로 학습하고, LAW(latent world
model)로 planning latent를 정제한다.

## 아키텍처

**Stage 1 (VAD-only).** 6-카메라 BEV 인코더 + 3D 검출/차선/타 에이전트 궤적
예측 헤드를 ETRI 데이터로 도메인 적응. `projects/mmdet3d_plugin/VAD/VAD.py`,
`VAD_head.py`.

**Stage 2 (VADLAW).** Stage1의 perception 가중치를 nuScenes-pretrained LAW
world model과 병합(`tools/merge_stage1_world_model.py`)한 뒤, ego-agent/
ego-map cross-attention과 world-model 기반 latent 예측 loss로 미세조정.
`projects/mmdet3d_plugin/LAW/VAD_LAW.py`.

두 단계 모두 ego future trajectory를 command(주행 의도)로 선택된 mode에 대해
회귀하는 multi-mode 구조 (`ego_fut_mode`개 후보 × `fut_ts`스텝 × (x,y)).

## 핵심 설계 요소

### 1. Causal, 규정 준수 ego-motion feature (10Hz)
`ego_lcf_feat`(속도/가속도/yaw rate)는 raw 10Hz 자차 pose로부터 직접 미분 계산.
대회 운영측 Q&A가 "과거 pose를 이용해 현재 ego status를 구하는 용도는
허용되며, 이를 계산하는데 사용된 과거 영상까지 입력할 필요는 없습니다"라고
명시한 데 근거해,
프레임별 fed-image 의무와 무관하게 조밀한 pose 샘플을 그대로 사용한다
(`tools/data_converter/etri_vad_converter_10hz.py`의 `robust_motion`/
`local_motion`).

### 2. STOP 커맨드 — 데이터에서 직접 파생
Raw `command.parquet`은 6종(LANE_KEEP/LANE_CHANGE_L/R/TURN_LEFT/RIGHT/U_TURN)만
제공하고 정지 상태를 구분하지 않는다. GT future trajectory(3.0s)의 총 변위가
임계값(`STOP_DISP_THRESH=0.5m`, 실데이터로 캘리브레이션) 미만이면 STOP으로
재라벨링해 7-way command vocab을 구성한다. 추론 시점엔 미래 GT가 없으므로,
target_point까지의 거리가 임계값 미만일 때만 STOP 모드를 선택하도록
`tools/etri_test_submit.py`에 반영했다.

### 3. target_point — 생성이 아닌 선택에만 사용
대회 운영국 Q&A(2026-08-26/27): "기준점을 기반으로 궤적을 새로 생성/보정하는
것이 아니라 선택에만 쓰이는 경우 허용됩니다." 이에 따라 target_point는 모델의
forward pass(생성 경로)에 전혀 관여하지 않는다 — attention/residual 주입 경로를
제거했다(`VAD_head.py`에 `ego_pos`는 항상 zero). 유일한 용도는 추론 시 이미
생성된 7개 궤적 후보 중 STOP 모드를 고를지 판단하는 selection 규칙
(`tools/etri_test_submit.py`)뿐이며, 이는 baseline이 이미 사용하는
"command로 출력 중 선택" 패턴과 동일하다.

### 4. PRISM — 특권 latent supervision (stage2 한정)
ETRI 데이터는 평가 구간(3s)을 넘어 5s까지 ego pose를 제공한다. 이 여분의
train-only 정보를 [PRISM (arXiv:2608.01201)](https://arxiv.org/abs/2608.01201)
스타일 CVAE로 활용: posterior encoder가 GT 3~5s future를 보고 얻은 latent로
prior(비전만 보는 네트워크)를 KL divergence로 정규화해, planning latent가
장기 주행 의도를 더 잘 담도록 유도한다. 추론 시엔 posterior를 전혀 쓰지 않고
prior의 결정론적 평균만 사용하므로 추가 latency/FLOPs가 없다
(`VAD_head.py`의 `prism_latent_supervision`, stage2 config에서만 활성화).

### 5. Qwen-VL teacher distillation (선택적 KD)
Qwen2.5-VL-3B를 ETRI 궤적 예측 텍스트 태스크로 파인튜닝해 teacher로 사용.
Teacher의 예측 궤적을 학습셋에 대해서만 오프라인으로 캐싱한 뒤
(`evodrive_etri_prep/generate_teacher_cache.py`), student(VAD+LAW) 학습 시
challenge-metric-정합 L2 distillation loss를 보조 항으로 추가한다
(`projects/mmdet3d_plugin/EvoKD/vad_head_kd.py`). Teacher는 forward graph에
전혀 포함되지 않으므로 추론 시 FLOPs/latency는 plain VADLAW와 동일.

## 저장소 구조

```
projects/
├── configs/VAD/                 # stage1/stage2/eval config
├── mmdet3d_plugin/
│   ├── VAD/                     # VAD.py, VAD_head.py (perception + planning head)
│   ├── LAW/                     # VAD_LAW.py (world-model 기반 stage2)
│   ├── EvoKD/                   # Qwen teacher distillation 연동
│   └── datasets/                # ETRI dataset, pipeline
tools/
├── data_converter/               # parquet → pkl 변환 (2Hz/10Hz variant)
├── cache/etri_geometry_cache.py  # undistort/crop 이미지 캐시
├── run_full_pipeline_split_301_75_10hz.sh   # stage1→merge→stage2→eval→submit 전체 파이프라인
├── eval_holdout_l2.py            # hold-out planning L2 평가
├── eval_holdout_l2_shortcut_ablation.py  # target_point/BEV-refine 의존도 진단
├── etri_test_submit.py           # 테스트셋 추론 → submission.json
├── measure_flops.py / measure_t_infer.py
evodrive_etri_prep/                # Qwen teacher 데이터 준비/캐시 생성
```

## 실행

```bash
# 1. pkl 생성 (10Hz ego-motion, 301/75 command-계층화 split)
python tools/data_converter/regenerate_causal_infos_10hz.py \
  --train-root data/train --test-root data/test \
  --output-dir data/etri/.causal_regen_split_301_75_10hz \
  --val-scenes val_scenes_301_75.txt

# 2. 전체 파이프라인 (stage1 48ep → merge → stage2 12ep → eval → submit → FLOPs/T_infer → 6조건 ablation)
bash tools/run_full_pipeline_split_301_75_10hz.sh
```

각 단계는 출력 파일 존재 여부로 완료 여부를 판단해 중단 후 재실행이 안전하다.

## 평가 지표

- **Planning L2**: 0.5/1.0/1.5/2.0/2.5/3.0초 ego 위치의 평균 오차 (m)
- **Error Score** = `L2 × (1 + max(0, T_infer − 100) / 200)`, T_infer는 클립
  하나(`reset_stream` → 마지막 프레임)의 forward pass 누적 시간
- **FLOPs cutoff**: baseline(2,351.0 GFLOPs) × 3 이하
