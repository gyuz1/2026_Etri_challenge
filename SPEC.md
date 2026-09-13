# 스펙 — stage1 부터 추론까지 무엇을 넣었고 무엇으로 재는가

`PIPELINE.md` 는 구조와 일정, 이 문서는 **넣은 것 하나하나의 근거와 측정 방법**이다.
모든 수치는 split_301_75_10hz 에서 실측했다. 추측은 `[미측정]` 으로 표시한다.

---

## 0. 지금 기준선

| | L2 | |
|---|---|---|
| compliant 베이스라인 | 0.4885 | 주최측 배포 상태 |
| **A student v1** | **0.4218** | 현 최고 기록. 이번 재구축이 넘어야 할 값 |
| A teacher v1 | 0.2328 | 증류 천장 (제출 불가) |
| 완벽한 v+a 운동학 오라클 | 0.2708 | teacher 가 이걸 이미 앞선다 = 장면 맥락을 쓴다 |
| 1 등 | 0.14 | **같은 split·metric 인지 미확인** |

---

## 1. 어디서 지고 있는가 — 이게 모든 설계의 출발점

커맨드별 비교 (teacher B 0.2542 vs student v1 0.4218):

| | teacher | student | 격차 |
|---|---|---|---|
| STOP | 0.0129 | 0.0120 | 없음 |
| U_TURN | 0.6631 | 0.6753 | 거의 없음 |
| **LANE_KEEP** (오차의 85%) | **0.2560** | **0.4350** | **1.70배** |

student 는 STOP·U_TURN 에서 이미 teacher 를 따라잡았고 **LANE_KEEP 하나에서만 크게 진다.**
`--zero-ego-lcf` 대조군에서 LANE_KEEP 이 0.256 → 10.01 (39배) 로 무너진 것과 합치면,
**LANE_KEEP 오차 = 속도 추정 오차의 적분**이다.

### 속도 오차 예산 [측정] `tools/speed_error_to_l2.py`

실제 val GT 에 등가속 롤아웃 + 통제된 노이즈, 대회와 동일한 L2 windowing:

| 속도 RMSE | L2 | | 가속도 RMSE | L2 |
|---|---|---|---|---|
| 0.00 | 0.2708 | | 0.00 | 0.2708 |
| **0.10 m/s** | **0.3311** | | 0.10 | 0.3140 |
| 0.25 m/s | 0.5039 | | 0.25 | 0.4506 |
| 0.50 m/s | 0.8571 | | 0.46 (a 의 std) | 0.6866 |

평균 속도 10.57 m/s 에서 **상대오차 2.4% (0.25 m/s) 만으로 L2 가 0.50** 까지 간다.
민감도가 이 정도라 "vision 으로 v/a 를 얼마나 정확히 맞히나" 가 사실상 전부다.

**단, 이건 하한 직관이지 요구조건이 아니다.** teacher 는 실제 ego 를 알면서 0.2328 로
완벽한 운동학 오라클(0.2708)을 앞선다 — 망은 운동학 외삽기가 아니라 차선 형상·선행차 같은
장면 맥락으로 미래를 제약한다.

---

## 2. stage1 에 넣은 것

| 설정 | 값 | 근거 (전부 [측정]) |
|---|---|---|
| `kd_weight` | **0** | Qwen KD teacher: train 0.1798 / hold-out **0.3511** (95% 악화, 5.9σ). 암기였다 |
| `loss_plan_reg` | **1.0** | 정확한 GT 가 같은 파일에 있다. stage2 와 같은 가중치 |
| `aux_bev_motion_temporal` | True | 없으면 descriptor 가 단일 프레임 global mean 으로 떨어지고, 그건 **원리적으로 shift-invariant** (실측 Δ 4.9e-10) |
| `aux_bev_motion_frames` | 3 | 2 프레임은 속도까지. 2차 차분이 가속도. 오라클 0.5965 → **0.2708** |
| `aux_bev_motion_grid` | 8 | 실제 변위 5.4m 에서 descriptor 가 자기 scale 의 57.1% 변함. **grid 4 는 30.1% — grid 8 이 1.90배 민감** |
| `aux_bev_motion_idx` | (0,1,2,3,4,7) | 가속도(2,3) 추가. 3 프레임이라야 관측 가능 |
| `aux_bev_motion_norm` | 성분별 std | 정규화 전 vx+speed 가 L1 의 98.7%, **yaw_rate 0.1%** → 회전 무감독 |
| `aux_bev_future_motion` | True | 현재를 다 알아도 천장이 0.2708. 3초 뒤 속도가 현재와 **샘플별 0.842 m/s** 다르다 |
| `history_sampling` | **'fixed'** | 기본 'random' 은 67% 샘플에서 간격 불일치 → 2차차분에 `v·dt` 5.28m 혼입 (진짜 `a·dt²` 0.12m 의 **46배**) |
| `EMAHook` | momentum 2e-4 | 체크포인트 **일반 슬롯 = EMA**, `ema_*` = raw (실측: epoch 간 이동량 165 vs 375) |

---

## 3. stage2 teacher (제출 불가)

| | 값 | 근거 |
|---|---|---|
| `ego_lcf_feat_idx` | [0..7] | 실제 자기상태 직접 입력 — 그래서 제출 불가 |
| `ego_lcf_embed_dim` | 8 | 64 면 576 폭이라 어떤 stage1 도 도너가 못 된다 |
| `ego_lcf_embed_residual` | **True** | zero-init residual. 없으면 iter100 `loss_plan_reg` 0.3884, 있으면 **0.0068** (57배) |
| `echo_cycle_weight` | 0.0 | 실제 ego 를 아는 teacher 는 cycle 보상이 없다 |
| `plan_reg_ts_weight_mode` | 'cumulative' | 지표에 수치미분: 실측 대비 오차 **19.5% vs position 52.2%** |
| `ego_status_distill_idx` | (0,1,2,3,4,7) | 5,6번 열(길이/폭)은 std 정확히 0. 정지 샘플에서 타깃 norm 의 **99.7%** |
| `ego_status_decode` | **True** ★신규 | 아래 5절 |

---

## 4. stage2 student (제출 모델)

| | 값 | 근거 |
|---|---|---|
| `ego_lcf_feat_idx` | **None** | 실제 자기상태가 planner 로 안 간다 |
| `ego_status_est_dim` | 8 | BEV descriptor 에서 추정. 2026-09-04 Q&A 가 허용한 vision 추론값 |
| `ego_status_est_dropout` | 0.3 | 추정치 과의존 방지 |
| 분리 증류 | scene 512 (w 0.3) + status 8 (w 0.5) | 융합형은 status 기여가 hidden 전체에 번져 따로 가중할 수 없다 |
| `echo_cycle_weight` | 0.1 | teacher 와 반대. **[측정] 실제로는 작다** — v1 에서 0.00508→0.00079, `plan_reg` 의 6.5% |
| `ego_status_decode` | **True** ★신규 | 아래 |

---

## 5. ★ 이번에 새로 넣은 것 — status slot 물리 디코딩

**문제**: 같은 descriptor 를 읽는 head 가 둘인데 **플래너에 닿는 건 하나뿐**이다.

```
                    ┌─ aux_bev_motion_head ─► bev_pred ─► GT ego_lcf 와 L1
BEV descriptor ─────┤                                      (플래너로 안 감)
   (6144)           │
                    └─ ego_status_est_net ──► 슬롯(8) ──► ego_feats ──► 플래너
                                                 ▲
                                          teacher 와 cosine 뿐
```

물리량 감독은 `aux_bev_motion_head` 에 있고 그 출력은 어디로도 안 간다
(`aux_bev_motion_feedback` 은 꺼져 있다 — 0.5635→0.6419 로 실패한 경로).
플래너가 먹는 슬롯은 **cosine 하나로만** 학습된다. 그런데 cosine 은 **스케일 불변**이라
10.5 m/s 와 5.2 m/s 를 구분하지 못하는데, 1절 예산표대로 L2 는 그 절대값에 잔인하게 민감하다.

**조치**: 슬롯에서 물리량을 되읽는 `Linear(8 → 6)` 를 붙이고 GT `ego_lcf` 로 감독한다.

- 학습 전용. 슬롯 폭·추론 경로·decoder 입력 **전부 불변**
- GT 를 **타깃**으로 쓰는 것이라 규정 위반 아님 (`aux_bev_motion` 이 이미 같은 방식)
- `aux_bev_motion_feedback` 실패와 다르다: 그건 **예측값을 플래너 입력에 밀어넣은** 것,
  이건 이미 플래너가 먹는 슬롯에 **감독을 추가**하는 것. 입력 경로 변경 없음
- 단일 Linear 인 이유: 더 깊으면 플래너의 첫 Linear 가 못 쓰는 형태로 상태를 숨길 수 있다
- 성분별 std 정규화 (2절과 동일). 안 하면 여기서도 yaw_rate 가 무감독
- teacher 에도 켠다 — 그 슬롯은 `raw + embed(raw)` 라 거의 공짜지만, 임베딩이 물리적으로
  읽히는 상태를 유지하게 묶어둔다. 그게 student 가 재현해야 할 성질이다

**[측정] 배선 검증**: 손실 계산됨, `d loss / d slot` 평균 9.98e-2 (0 아님 = 슬롯에 압력이 간다),
8차원 슬롯이 6개 물리량을 표현 가능 (400 step 최적화 후 정규화 L1 0.087).

**[미측정] 리스크**: 슬롯이 물리량 재현에 용량을 뺏겨 teacher 표현 정합이 나빠질 수 있다.
`ego_status_decode_weight=0.5` 로 조절 가능하고, `nodistill` 쪽에서 증류와 분리해 잴 수 있다.

---

## 6. 무엇으로 재는가

### 지표
```
L2 = mean( mean|Δ|₁ₛ , mean|Δ|₂ₛ , mean|Δ|₃ₛ )      Δ = cumsum(pred) − cumsum(gt)
점수 = L2 × (1 + max(0, T_median − 100)/200)         T_infer 는 model forward 만, 4090 기준
```
스텝 6개(0.5s 간격), 창은 `dist[:2] / dist[:4] / dist[:6]`.

### 추론 설정
```bash
--frame-offsets 0,-5,-10 --bev-only-history --fp16
```
`--bev-only-history` 는 history 프레임의 decoder 를 건너뛴다. BEV 가 bit-identical 이라
궤적은 안 바뀌고 비용만 준다 (27.7ms vs 61.9ms 실측).

| 프레임 | T_infer | 페널티 |
|---|---|---|
| 2 (플래그 없음) | 135ms | ×1.176 |
| 2 (플래그 켬) | 89.6ms | ×1.0 |
| **3** | 117.3ms | **×1.087** |

3 프레임 본전 문턱은 8.7% 인데, 가속도 레버는 오라클 기준 55% 다.

**`eval_l2.sh` 는 config 에서 `aux_bev_motion_frames` 를 읽어 창을 정하고, 모자라면 거부한다.**
3 프레임 모델을 2 프레임 창으로 재면 `prev_bev2` 가 끝까지 None 이라 가속도 블록이 0 인 채
채점된다 — 모델에서 고친 결함을 플래그 기본값이 되살리는 형태였고, 실제로 그랬다.

### 학습 전 필수 검사
```bash
python tools/audit_pipeline.py <config> [--eval-config <cfg>]   # 도너·정합·규정·no-op·누수
python tools/check_accel_block_live.py <config>                 # 3프레임 경로(학습·추론)
```
`run_stage2_best.sh` 가 둘을 자동으로 돌리고 실패하면 학습을 안 띄운다.

### 학습 중 검사
```bash
python tools/check_accel_block_trained.py <ckpt> --config <cfg>  # raw weight 로 결과 확인
```
`ema_*` 가 있으면 **raw 를 우선해서 읽는다**. 일반 슬롯은 EMA 라 초기 체크포인트에서
학습이 잘 돼도 초기값 근처에 머문다 — 그걸 읽고 "학습이 안 된다" 고 오판할 뻔했다.

### 운영 감시
양 머신 로그의 `Traceback|Error|CUDA out of memory|nan|Killed|assert` + **프로세스 소실**.
로그에 아무것도 안 남기고 죽는 경우를 잡으려면 프로세스 확인이 따로 필요하다.

---

## 7. 검증했고 문제 없던 것 (다시 의심하지 말 것)

| | 결과 |
|---|---|
| descriptor shift 민감도 | 실제 변위에서 57.1%, global mean 4.9e-10 |
| `FUT_TS_INTERVAL_S` | 0.5 vs 실측 **0.5001** |
| `aux_bev_motion_norm` 6개 | 실제 std 와 **정확히 일치**, 분담 16.7% 균등 |
| 미래속도 valid flag | **100%** 유효 |
| `refine_ego_trajs_with_bev` | no-op 6.6e-07 (fp32 한계), grid_sample 축 규약 정상 |
| PRISM posterior | 학습 전용 게이트, 추론은 prior mean 결정론적 |
| teacher 격리 | 비등록 리스트 + `eval()` + `requires_grad=False`, `prev_bev` clone |
| train/eval 기하 | **`lidar2img` bit-identical**, 픽셀만 3.5% (JPEG 디코드, 현 설정이 최선) |
| merge / surgery | 정확한 합집합, scene 512열 bit-identical + 8열 정확히 0 |

---

## 8. 열린 문제

- **1 등 0.14 가 같은 split·metric 인가** — 3회 이상 제기, 여전히 미확인.
  "0.2 미만 불가능" 이라는 결론이 이 가정 위에 있다
- `grad_norm: nan` — A5000 epoch 22·23 에 각 1회 (0.41%). 21 epoch 동안 0 이었다.
  fp16 GradScaler 의 정상 동작 범위지만 추세는 지켜본다
- 2절의 "옛 stage1 = nuScenes 의 weight decay (cosine 1.0000)" 분석이 EMA 슬롯을
  읽은 것 — raw 로 재확인 안 됨
- `ego_status_decode` 의 실제 효과 — 미측정. `nodistill` 이 증류와 분리해 잰다
