# 파이프라인 — 현재 최선 구성

측정 근거는 전부 `SHARED_CONTEXT.md` 2절에 있다. 이 문서는 "무엇을 왜 이렇게
설정했는가"와 "각 서버가 무엇을 돌리는가"만 담는다.

---

## 전체 구조

```
                    ckpts/law_pretrained_nus.pth  (nuScenes warm-start)
                                  │
            ┌─────────────────────┴─────────────────────┐
            │                                           │
   stage1 BEST nolcf  (48ep)                  stage1 BEST lcfon  (48ep)
   ego_lcf OFF, decoder 512폭                 ego_lcf ON, decoder 520폭
            │  3090                                     │  A5000
            ▼                                           ▼
   merge + 8칸 zero-pad → 520                  merge → 520 (surgery 불필요)
            │                                           │
            │                              stage2 TEACHER (12ep)
            │                              ego_lcf 직접 입력 → 제출 불가
            │                                           │
            │            ego_scene_feats(512) + ego_status_feats(8) 증류
            └──────────────────┬────────────────────────┘
                               ▼
                    stage2 STUDENT (12ep)  ← 제출 모델
                    ego_lcf 없음. 상태 슬롯은 BEV 에서 추정
                               │
                               ▼
              제출: --frame-offsets 0,-5,-10 --bev-only-history
```

---

## 지금 각 서버가 하는 일

| 서버 | 작업 | work_dir | 산출물 |
|---|---|---|---|
| **3090** | `VAD_etri_tiny_stage1_best_nolcf.py` | `stage1_best_nolcf` | student 도너 (512폭) |
| **A5000** | `VAD_etri_tiny_stage1_best_lcfon.py` | `stage1_best_lcfon` | teacher 도너 (520폭) |

두 lineage 가 따로 필요한 이유: teacher 는 ego status 를 `ego_feats` 에 넣으므로
decoder 가 `embed_dims*2 + 8 = 520` 폭이고, compliant student 는 512 다. 한쪽
폭으로 학습한 도너는 다른 쪽으로 전이되지 않는다.

```bash
# 3090
docker exec -it gyuz_split_3090 tail -f /workspace/VAD/work_dirs/stage1_best_nolcf/train.log

# A5000
ssh -p 22022 vcl@10.10.52.49 -t "docker exec -it gyuz_split2 tail -f /workspace/VAD/work_dirs/stage1_best_lcfon/train.log"
```

---

## 적용한 설정과 근거

### 운동 표현 (stage1·stage2 양쪽 동일)

| 설정 | 값 | 근거 |
|---|---|---|
| `aux_bev_motion_temporal` | True | **필수.** 없으면 descriptor 가 단일 프레임 global mean 으로 떨어지고, 그건 shift-invariant 라 자기운동이 안 보인다. 아래 두 설정도 같이 무효화된다 |
| `aux_bev_motion_frames` | 3 | 2 프레임은 속도까지만. 2차 차분이 곧 가속도 (`d1-d2 == a·dt²`, 폐곡선 검증). 오라클: 속도만 0.5965 → 가속도까지 **0.2708** |
| `aux_bev_motion_grid` | 8 | grid 4 는 한 칸 15.0m×7.5m 인데 0.5초 변위가 ~5m — 칸 안에서 평균화돼 사라진다. grid 8 은 7.5m×3.8m |
| `aux_bev_motion_idx` | (0,1,2,3,4,7) | 가속도(2,3) 추가. 3 프레임이 되어야 관측 가능해진 값 |
| `aux_bev_motion_norm` | 성분별 std | 정규화 전: vx 49.3% + speed 49.4% (거의 같은 값), **yaw_rate 0.1%** → 회전이 사실상 무감독 |
| `aux_bev_future_motion` | True | 현재를 아는 것의 상한이 0.2708 인데 1 등은 0.14. 3 초 뒤 속도가 현재와 평균 **0.80 m/s** 다르다 — 운동학 외삽이 버리는 부분 |

### stage1 planning 신호

| | 값 | 근거 |
|---|---|---|
| `kd_weight` | **0** | Qwen KD teacher: train 0.1798 / hold-out **0.3511** (95% 악화, 5.9σ). 암기 신호였고, stage1 에서 planner 를 학습시키는 **유일한** 신호였다 |
| `loss_plan_reg` | **1.0** | 정확한 GT 가 같은 파일에 있다. stage2 와 같은 가중치라 두 단계가 일관된 목적함수를 본다 |

`loss_plan_reg=0.0` 은 원 VAD 레시피("덜 익은 BEV 위에서 planner 를 학습시키지
말 것")를 물려받은 것인데, KD 를 넣는 순간 그 전제가 이미 깨졌다.
**미검증 반론**: 48ep GT 가 planner 를 과적합시키거나 인지 학습을 방해할 수 있다.

### stage2 teacher (제출 불가)

- `ego_lcf_feat_idx=[0..7]` — 실제 자기상태 직접 입력
- `ego_lcf_embed_dim=8` + **`ego_lcf_embed_residual=True`** — 도너의 ego 열은 raw
  물리값으로 학습됐다(mean|w| 0.0481, scene 열의 2.2배). zero-init residual 로
  step 0 에 raw 와 bit-identical. 없을 때 iter100 loss_plan_reg 0.3884 → **0.0068**
- `echo_cycle_weight=0` — plan 을 world-model 왕복 일관성 쪽으로 당기는데, 실제
  ego 를 아는 teacher 는 그 보상을 못 받는다
- `plan_reg_ts_weight_mode='cumulative'` — metric 이 델타를 cumsum 하므로 초기
  스텝 오차가 이후 위치를 전부 밀어낸다

### stage2 student (제출 모델)

- `ego_lcf_feat_idx=None` — 실제 자기상태가 planner 로 가지 않는다
- `ego_status_est_dim=8` — BEV descriptor 에서 추정. 2026-09-04 Q&A 가 허용한
  "신경망 자신이 vision 으로 추론한 ego status"
- `ego_status_est_dropout=0.3` — 추정치에 과의존 방지.
  `aux_bev_motion_feedback` 이 0.5635→0.6419 로 실패한 구조와 같은 계열이라 필요
- 분리 증류: scene 512 (w=0.3) + status 8 (w=0.5)
- `echo_cycle_weight=0.1` **유지** — teacher 와 반대. student 는 ego motion 단서가
  없어 cycle consistency 의 보상이 실제로 있다

### 평가·제출

```bash
--frame-offsets 0,-5,-10 --bev-only-history
```

`--bev-only-history` 는 history 프레임의 decoder 를 건너뛴다. BEV 는
bit-identical 이라 궤적이 안 바뀌는데 비용만 준다 (27.7ms vs 61.9ms, 실측).

| 프레임 | T_infer | 페널티 |
|---|---|---|
| 2 (플래그 없음) | 135ms | x1.176 |
| 2 (플래그 켬) | 89.6ms | **x1.0** |
| **3** | 117.3ms | **x1.087** |

점수 = `L2 × (1 + max(0, T−100)/200)`, 4090 기준 model forward 만 계산.

---

## 규정

| | 허용 | 우리 사용 |
|---|---|---|
| `ego_lcf` → planner 입력 | ❌ | student 는 안 씀 |
| `ego_lcf` → 학습 감독 타깃 | ✅ | `aux_bev_motion` 의 `ego_lcf_target` |
| vision 추정 ego status → planner | ✅ | student 의 8 차원 슬롯 |
| `can_bus` → BEV 인코더 | ✅ | 배포 베이스라인이 이미 이 상태 |
| `target_point` → 생성 | ❌ | 안 씀 |

`can_bus` 와 `ego_lcf` 의 차이: 배포 베이스라인이 `can_bus` 는 쓰고
`ego_lcf_feat_idx=None` 으로 막아뒀다. can_bus 는 이전 BEV 를 자차 이동만큼
회전·평행이동 보정하는 **기하 정합용**이고, 검출·맵·모션 head 가 전부 공유한다.
규정의 "간접 활용(공통 feature 개선)" 에 해당한다.

---

## 검증

학습 전에 반드시:

```bash
python tools/audit_pipeline.py <train_config> \
    --eval-config <eval_config> \
    --val-ann data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl
```

검사 항목 — 전부 **실제로 일어났고 그때 에러가 안 났던** 것들이다.
mmcv 는 `strict=False` 로 로드하므로 shape 불일치가 크래시가 아니라 로그 한 줄이다.

1. 도너 decoder shape (576 도너를 520 모델에 → planner 랜덤 초기화)
2. train/eval config 일치
3. 규정 (제출 모델에 ego_lcf 가 planner 로 가는지)
4. 조용한 no-op (`frames`/`grid` 를 켰는데 `temporal` 이 꺼진 경우 등)
5. teacher/student descriptor 대칭
6. train/val scene 겹침

---

## 일정

```
+2일   stage1 둘 다 완료 → merge
+3일   teacher (A5000)  |  student 는 teacher 필요해서 대기
+4일   student (3090)   → 최종 평가
```

---

## 목표와 한계

| | L2 |
|---|---|
| 1 등 | 0.14 *(같은 split 인지 미확인)* |
| 완벽한 v+a 운동학 오라클 | **0.2708** |
| 우리 teacher 최고 | 0.2328 |
| **우리 compliant 최고** | **0.4218** |
| compliant 베이스라인 | 0.4885 |

**0.2 미만은 현 구조로 불가능하다.** 증류는 student 가 teacher 를 못 넘고, 우리
teacher 가 0.23 이다. 1 등 0.14 는 완벽한 운동학 오라클마저 48% 앞서므로 미래
기동을 장면에서 예측해야 나오는 수치다. 현실적 목표는 **0.33~0.38**.
