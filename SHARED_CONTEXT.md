# SHARED_CONTEXT

ETRI 2026 자율주행 챌린지 · LAW_split 트랙 공유 작업 기록.
Claude와 Codex가 공유한다. 최종 갱신: 2026-09-09.

이 파일과 `CLAUDE.md` / `AGENTS.md`는 **저장소 안(`LAW_split/`)에 있고**,
상위 `challenge/`에는 심볼릭 링크만 있다. 상위 디렉터리는 git 저장소가 아니라
버전 관리도 머신 간 동기화도 안 되기 때문. 어느 쪽 경로로 열어도 같은 파일이다.

표기 규칙
- **[사용자]** 사용자가 직접 말한 목표·제약·지시
- **[Claude 제안]** Claude가 제안했으나 아직 확정되지 않은 것
- **[Codex 의견]** Codex가 남긴 의견 (보존 대상, 삭제 금지)
- **[확정]** 합의되어 실행에 들어간 결정
- **[측정]** 실제로 실행해서 얻은 수치 (추측 아님)

---

## 1. 목표와 요구사항 [사용자]

- **최우선 목표는 compliant 모델의 planning L2를 낮추는 것.** 현재 최고 기록 0.4885m.
- 규정: ego 상태(`ego_lcf`)와 목표점(`target_point`)을 planner **생성 단계**에 직접 넣을 수 없다.
  간접 활용(공통 feature 개선)은 주최측 Q&A로 허용됨.
- **시간이 binding constraint.** 마감이 임박했고 재작업 여유가 없다.
  "학습 시간보다 정확하고 올바른 설계로 L2를 낮추는 게 목적."
- **버그 재발 방지가 최우선 요구사항.** 조용히 틀린 수치가 나오는 일이 반복되면 안 된다.
  (실제로 버그 하나로 22시간 학습을 통째로 날린 전례가 있다. 6절 참조)
- 3090은 사용자가 직접 관리하는 경우가 있음. 지시 없이 건드리지 말 것.

---

## 2. 확정된 사실 [측정]

동일 split(301/75), 10Hz, 2-frame hold-out eval 기준.

| 항목 | 값 | 비고 |
|---|---|---|
| compliant 최고 | **0.4885m** | KD stage1 계보 |
| `privileged_distill` (궤적 증류) | 0.4897m | 무변화. 학습 중엔 privileged head가 35% 잘 맞춤(0.0084 vs 0.0130) |
| `aux_bev_motion_feedback` | 0.5635 → **0.6419** | 악화. LANE_KEEP 0.582→0.690, 회전은 개선 |
| can_bus[7:16] 제로화 | 0.593 → **10.4m** | 모델이 can_bus에 절대적으로 의존 |
| ego_lcf-ON 기록 (금지 전) | **0.2166m** | 같은 split. 다른 stage1 계보 + 구버전 코드 (아래 상세) |
| surgical 계보 vs KD 계보 | 0.5635 vs **0.4885** | KD 승 (13%) |
| TP shortcut 제거 시 (구코드) | 0.114 → 5.45m | 47배 붕괴 |

**커맨드별 오차 기여도** (베이스라인 0.4885 분해)

| 커맨드 | 샘플 | L2 | 오차 기여 |
|---|---|---|---|
| **LANE_KEEP** | 3321 (82.1%) | 0.5079 | **85.4%** |
| LANE_CHANGE_L | 138 | 0.6918 | 4.8% |
| TURN_LEFT | 142 | 0.5559 | 4.0% |
| TURN_RIGHT | 118 | 0.4995 | 3.0% |
| LANE_CHANGE_R | 58 | 0.6168 | 1.8% |
| U_TURN | 26 | 0.6520 | 0.9% |
| STOP | 242 | 0.0133 | 0.16% |

→ **헤드룸의 85%가 LANE_KEEP에 있다.** 회전을 전부 오차 0으로 만들어도 0.4885→0.4178이 한계.
LANE_KEEP 20% 개선이 그보다 크다(→0.4051).

→ LANE_KEEP 오차는 시간에 선형 증가(0.290/0.497/0.736) = **속도 추정 오차의 적분**.
약 0.245 m/s, 평균 주행속도 13 m/s 대비 1.9%.

### Qwen KD teacher 자체 성능 [측정 2026-09-10] — "왜 증류가 안 넘어오나" 조사

`work_dirs/teacher_cache/etri_train_teacher_cache.json` (17157 항목, valid 99.9%)의
`teacher_l2`를 집계:

| | L2 |
|---|---|
| **Qwen KD teacher** (train split, 파인튜닝한 데이터) | **0.1798m** — ⚠️ 암기 포함, hold-out에선 0.3511 |
| 옛 최고 기록 (ego_lcf ON) | 0.2166m |
| A teacher v1 (ego_lcf ON) | 0.2328m |
| B teacher (ego_lcf ON) | 0.2542m |
| compliant 최고 | 0.4885m |

**미래 프레임 유출은 아니다.** [build_etri_teacher_data.py:157-168](../evodrive_etri_prep/build_etri_teacher_data.py)에
따르면 예전엔 `+1s/+2s/+3s` 미래 front 이미지를 teacher에 넣었다가 "규정 위반"으로 이미 수정했고,
지금은 `sample['teacher_images'] = student_images`로 student와 **동일한 인과적 입력**을 받는다.

**[측정] teacher 우위의 커맨드별 분해 — 속도 지식의 지문**

| 커맨드 | teacher (train) | compliant (val) | 배율 |
|---|---|---|---|
| **LANE_KEEP** | 0.1680 | 0.5079 | **3.02배** |
| LANE_CHANGE_L | 0.2718 | 0.6918 | 2.54배 |
| LANE_CHANGE_R | 0.2554 | 0.6168 | 2.42배 |
| TURN_LEFT | 0.3230 | 0.5559 | 1.72배 |
| TURN_RIGHT | 0.3707 | 0.4995 | 1.35배 |
| U_TURN | 0.4761 | 0.6520 | 1.37배 |
| **STOP** | 0.0138 | 0.0133 | **1.0배 (동일)** |

정지(STOP)에서 우위가 정확히 0이고, 순수 속도적분(LANE_KEEP)에서 최대다.
→ teacher 강점의 상당 부분이 **ego history(속도)** 에서 온다는 뜻. vision-only student가
원천적으로 접근 못 하는 정보라 궤적 증류로는 안 넘어온다.

**이미 같은 실패가 측정돼 있다**: `privileged_distill`(궤적 수준 특권 증류) = 0.4897 (무변화).
당시 진단도 "privileged head가 실제로 더 잘 맞췄지만(0.0084 vs 0.0130) 출력 12개 숫자로
짜내는 과정에서 살아남지 못했다"였다. **궤적 수준 증류가 두 번 다 실패한 셈.**
Scheme A/B의 feature distillation이 바로 이 한계를 겨냥한 것.

### ★ [측정 2026-09-10] **Qwen KD teacher는 상당 부분 암기다** — hold-out 검증 완료

teacher는 `run_etri_teacher.sh`의 `train_data=Drive_KD_train_his_ego_future.json`,
즉 **train split 301 scene으로 3 epoch 파인튜닝**됐는데, 지금까지 인용해온 0.1754/0.1798은
**바로 그 train split에서 잰 값**이었다. 같은 teacher를 hold-out val 75 scene(파인튜닝에
쓰인 적 없음, scene 겹침 0개 확인)에서 돌려 같은 코드/같은 L2 windowing으로 재측정:

| | n | mean | median |
|---|---|---|---|
| **TRAIN 301 scene** (파인튜닝함) | 17138 | **0.1798** | 0.1166 |
| **VAL 75 scene** (안 본 데이터) | 399 | **0.3511** | 0.2013 |

→ **악화폭 0.1712m (95%), 5.9 시그마.** 재현 스크립트:
`evodrive_etri_prep/run_val_teacher_cache_holdout.sh` +
`evodrive_etri_prep/compare_teacher_train_vs_val.py`

**함의 (중요):**
1. **teacher의 진짜 실력은 0.35m 수준**이다. compliant 최고(0.4885)보다 나은 건 맞지만,
   0.1754가 시사하던 "2.8배 우위"는 허수였고 실제로는 1.4배 정도다.
2. stage1의 `loss_plan_kd`는 **train scene에서 teacher가 외운 답을 따라하도록** decoder를
   학습시켜온 셈이다. 일반화되지 않는 신호를 48 epoch 동안 유일한 planning 신호로 쓴 것.
3. **VLM을 더 좋은 것(Alpamayo 등)으로 바꿔도 해결되지 않는다.** 모델 용량 문제가 아니라
   "파인튜닝 데이터 = 평가 데이터"라는 설계 문제이므로, 어떤 모델을 넣어도 같은 일이 벌어진다.
   → Alpamayo 교체 보류 결정의 근거가 됨.
4. 캐시에 val scene도 섞여 있다(`run_fulldata_teacher_cache_75extra.sh`가 fulldata 376 scene용으로
   나머지 75개를 추가했는데 그게 정확히 val scene이다). 큰 캐시 속 val scene은 n=4272, mean 0.2737 —
   train(0.1798)과 hold-out(0.3511) 사이다. **stage2 학습 시 `data.train.ann_file`은 301 train만
   쓰므로 누수는 아니지만, 캐시 전체 평균을 teacher 성능으로 인용하면 안 된다.**

**남은 확인거리**: KD 신호가 그래도 도움이 됐다는 측정(KD 계보 0.4885 vs surgical 0.5635)은
계보가 같이 바뀐 교란된 비교다. 암기 신호였다면 그 13%가 어디서 왔는지 재설명이 필요하다.

**입력 검증** [측정] `tools/verify_teacher_inputs.py`로 확인:
`ego_target_point`(8/8 non-zero, 샘플별 상이, reshape 후 폭 2), `ego_lcf_feat[0..7]`(8/8 non-zero),
`can_bus[7:16]`(8/8 non-zero) 전부 실제 값이 들어옴.

---

## 3. A안 — 분리 증류 [사용자 제안 → 확정, 3090에서 진행]

### 내용
```
teacher: ego_feats = cat([agent(256), map(256), MLP(8→64)])   = 576
student: ego_feats = cat([agent(256), map(256), vision추정(64)]) = 576
증류: 512 ↔ 512,  64 ↔ 64   (따로, 각자 가중치)
```

### 선택 이유 [사용자]
- 합쳐서 하나의 cosine으로 재면 576차원 중 512(89%)가 loss를 지배해 64 블록이 학습되지 않는다.
- ego_lcf를 raw 8칸으로 옆에 붙여놓고 512만 증류하는 건 의미가 없다.
  ego_lcf를 분리하거나 의미있게 사용해야 한다.

### Claude가 인정한 A안의 장점
- 분업이 명시적: 512=장면, 64=자기운동. 각자 감독 신호를 받는다.
- teacher/student 폭이 동일해져 활성값이 직접 비교 가능.
- 문헌 뒷받침: [AdaptiveAD (arXiv 2511.13079)](https://arxiv.org/abs/2511.13079)가
  "BEV 인코더 단계의 ego status 조기 융합"을 shortcut의 근원으로 지목하고
  scene/ego 이중 분기 분리를 해법으로 제시. 구조적으로 동형.

### A안의 알려진 리스크
`aux_bev_motion_feedback`이 실패한 구조와 동일 계열 — vision 추정치를 **decoder 입력**으로 넣는다.
추정 오차가 추론 시에도 그대로 주입되고, 피해가 LANE_KEEP(오차의 85%)에 집중됐던 전례가 있다.

**[측정 2026-09-10] 임베딩 64차원 자체가 콜드스타트를 B안보다 심하게 만듦.**
`ego_lcf_embed_dim=64`를 고른 근거가 코드 어디에도 없음 (임의 선택으로 보임). B안(raw 8칸
zero-pad)과 같은 12epoch을 받았는데 도달한 scene 대비 비가 A는 0.147, B는 0.292 — **A가 절반
수준.** 이유: B는 decoder 입력 8칸만 새로 자라면 되는데, A는 ① decoder 입력 64칸 ②그걸
만드는 `ego_lcf_embed_net`(도너에 없어 랜덤초기화되는 새 2-layer MLP) 둘 다 12epoch 안에
학습해야 함 — 학습할 파라미터가 더 많은데 시간은 동일. A student 결과가 기대에 못 미치면
"임베딩 설계가 나쁘다"보다 "64가 12epoch엔 너무 커서 안 여물었다"를 먼저 의심할 것.
개선책(미실행, 시간 부족으로 보류): 임베딩 크기를 8~16으로 축소하거나, B처럼
embed_net까지 포함해 stage1부터 학습시키는 lineage로 전환.

### 완화책 [구현 완료 2026-09-09]
**Modality dropout (p=0.3).** 학습 중 64 슬롯을 무작위로 드롭하면 디코더가 그것에만 의존할 수 없다.
문헌 근거: ["dropout precludes over-reliance on the easiest modality"](https://www.emergentmind.com/topics/modality-dropout),
[CVPR 2024 — teacher latent를 앵커로 dropout 편향 방지](https://openaccess.thecvf.com/content/CVPR2024/papers/Dai_A_Study_of_Dropout-Induced_Modality_Bias_on_Robustness_to_Missing_CVPR_2024_paper.pdf).
이 코드베이스에 이미 `prev_bev_dropout`이라는 같은 패턴의 선례가 있다.

### 구현 상세 [확정 2026-09-09]
- student의 64칸 출처 = **BEV 시간차 descriptor** `cat([현재, 현재-이전])` (1024-d).
  `aux_bev_motion`이 이미 이 descriptor에서 (vx, vy, yaw_rate, speed)를 회귀하고 있다는 것이
  "여기 자기운동 정보가 실제로 있다"는 근거다. 따라서 `aux_bev_motion_temporal=True`가 **필수**이고,
  코드에서 강제한다 (단일 프레임 global mean은 shift-invariant라 자기운동 신호가 거의 없음).
- dropout은 증류용 벡터를 **캡처한 뒤** 적용한다. 드롭된 스텝에서도 `loss_status_distill`은
  추정망을 계속 학습시키고, 디코더의 시야만 가려진다. (합성 텐서로 gradient 흐름 확인함)
- 가중치: scene 0.3 / status 0.5. status가 더 높은 이유는 추정망의 다른 감독 경로
  (디코더를 통한 planning loss)가 30% 스텝에서 꺼지기 때문.
- **`aux_bev_motion_feedback` 실패와 다른 점**: 그건 4개 스칼라를 real ego_lcf에 대한 L1로만
  학습시켰다. 이건 teacher의 디코더가 **실제로 소비한 벡터**를 목표로 한다. 목표가 물리량이 아니라
  표현이다. 다만 구조적 베팅은 같으므로 dropout이 필요하다.

---

## 4. B안 — 융합 증류 [Claude 제안 → 확정, A5000에서 진행]

### 내용
```
teacher: ego_feats = cat([agent(256), map(256), lcf_raw(8)]) = 520
         → Linear(520→512) → ReLU → ego_plan_hidden(512)   ← 여기를 읽음
student: ego_feats = cat([agent(256), map(256)])            = 512
         → Linear(512→512) → ReLU → ego_plan_hidden(512)
증류: hidden ↔ hidden  (cosine, 1개)
```

### 근거 [Claude 제안]
- 첫 Linear가 8칸을 512칸과 섞은 **직후** 지점이라, student는 그 기여분을 vision만으로
  재구성해야 한다. `ego_feats`(융합 전)를 맞추면 반대로 "속도는 512에 안 담아도 된다"를 학습할 위험.
- student가 운동 추정치를 **벡터로 만들지 않는다.** 실패해도 손실이 안 내려갈 뿐,
  디코더 입력은 순수 vision 512 그대로.
- teacher 구조 변경 불필요 → 지금 돌고 있는 teacher를 그대로 사용.

---

## 5. 검토 후 배제한 대안

| 대안 | 배제 이유 |
|---|---|
| **BEV 레벨 증류** (AdaptiveAD 방식) | AdaptiveAD는 두 분기의 BEV가 구조적으로 다름(모션 보정 유/무). 우리는 teacher/student 둘 다 `use_can_bus=True`로 **동일하게** 보정됨 → 옮길 실체가 없음. student의 BEV는 detection/map도 공유해 교란 위험만 큼 |
| **ResAD 잔차 참조** ([arXiv 2510.08562](https://arxiv.org/abs/2510.08562)) | inertial reference 대비 잔차 예측. ego status를 생성에 직접 쓰는 셈이라 규정 위험. **normalization 아이디어만** 차용 (7절) |
| **stage1을 ego_lcf ON으로 재학습** | 60시간 소요. stage1은 `loss_plan_reg=0.0`이라 신호가 매우 약함(col/dir/bound만 활성, 0.004 수준). 게다가 lcf가 있으면 상류 vision feature가 덜 발달해 **distillation 소스로는 오히려 불리**. 기존 관측(0.5635 vs 0.4885)도 같은 방향 |
| **ego_lcf OFF 모델을 teacher로** | 특권 정보가 없어 옮길 것이 없음. self-distillation이 되는데 우리에겐 더 나은 compliant 모델도 없음 |
| **64 슬롯을 raw 8칸으로 대체** | teacher 재학습은 피할 수 있으나, raw 8칸 증류 = ego_lcf 회귀 = `aux_bev_motion`과 동일. 정보가 늘지 않음 |

---

## 6. 버그 이력 — 반드시 읽을 것

| 버그 | 증상 | 원인 | 대가 |
|---|---|---|---|
| **`ego_target_point` 미전달** | teacher eval 35.7m | `VAD_LAW.forward_train`이 인자로 받고도 `pts_bbox_head`에 안 넘김. 학습 내내 goal=0. test 경로는 정상 전달 → train/test 불일치 | **22시간 학습 폐기** |
| **eval config 불일치** | `prism_posterior_net.0.weight` size mismatch | eval config가 학습 config의 `prism_posterior_lcf_idx`를 안 맞춤 | eval 1회 재실행 |
| **`prev_bev` in-place 이중 회전** | (사전 발견, 피해 없음) | 인코더가 `VAD_transformer.py:268`에서 caller 텐서를 in-place 회전. student가 먼저 돌린 걸 teacher에 그대로 넘기면 이중 회전 | 사전 차단 |
| **MSE 스케일 폭주** | `loss_feature_distill: 292` (다른 loss는 0.01~5) | 독립 학습된 두 망은 임베딩 스케일이 다름 | cosine으로 전환 |
| **`--cfg-options`로 리스트 오버라이드** | `cfg must be a dict, got str` | `log_config.hooks="[{...}]"` 파싱 실패 | 별도 override config 사용 |
| **컨테이너 shm 부족** | DataLoader bus error | 임시 컨테이너에 `--shm-size` 미지정 | `workers_per_gpu=0` |

**교훈**: 크래시 없이 조용히 틀리는 유형이 가장 위험하다.
학습 시작 전 (a) config diff로 의도한 차이만 있는지, (b) 데이터가 실제로 들어오는지,
(c) eval config가 학습 config와 구조적으로 일치하는지 반드시 확인한다.

---

## 7. 변경한 파일과 상태

### `projects/mmdet3d_plugin/VAD/VAD_head.py`
- **`ego_plan_hidden`** outs에 추가 — `ego_fut_decoder[:2](ego_feats)`, PRISM 주입 이전 결정론적 값. B안 증류 대상. [확정, 미검증]
- **`plan_reg_ts_weight_mode`** (`'position'` 기본 / `'cumulative'`) — 지표가 델타를 cumsum하므로
  초기 델타 오차가 이후 모든 위치를 밀어냄. 실효 가중치 `[1.0, .69, .39, .25, .11, .056]` vs
  기존 `[.31, .31, .14, .14, .06, .06]`. [확정, 미검증]
- **`ego_lcf_embed_dim`** — A안 teacher용. raw 8칸 대신 `Linear(8→64)→ReLU→Linear(64→64)`.
  `ego_fut_dec_in_dim` 512→576. [확정, 미검증]

### `projects/mmdet3d_plugin/LAW/VAD_LAW.py`
- **`ego_target_point=ego_target_point`** 를 현재 프레임 `pts_bbox_head` 호출에 추가 — 6절 버그 수정 [확정, 검증됨]
- **`_teacher_plan_hidden`** — frozen teacher를 별도 forward, `ego_plan_hidden` 반환.
  teacher는 `requires_grad=False` + `.eval()` + `torch.no_grad()` 3중 차단
- **`teacher_prev_bev`** 를 dropout 이전에 `.clone()`, metas는 `deepcopy` — in-place 오염 차단
- **`loss_feature_distill`** = `weight * (1 - cosine(student_hidden, teacher_hidden.detach()))`

### `tools/eval_holdout_l2_and_tinfer.py`
- `--zero-target-point`, `--zero-ego-lcf` 진단 플래그 추가.
  특권 입력이 실제로 쓰이는지 검증용 (6절 버그 재발 방지)

### `tools/verify_teacher_inputs.py` (신규)
- CPU만으로 TP/ego_lcf/can_bus가 실제 값으로 들어오는지 검증

### configs (신규)
| 파일 | 용도 |
|---|---|
| `VADLAW_etri_tiny_kd_lcfon_diag.py` | **B안 teacher** (ego_lcf ON, TP-free, KD stage1) — A5000 학습 중 |
| `VADLAW_etri_tiny_fast_eval_kd_lcfon_diag.py` | 위 teacher의 eval config (구조 일치 검증 완료) |
| `VADLAW_etri_tiny_kd_lcfemb_teacher.py` | **A안 teacher** (ego_lcf → 64d 임베딩) — 3090 학습 중 |
| `VADLAW_etri_tiny_fast_eval_kd_lcfemb_teacher.py` | 위 teacher의 eval config (구조 일치 검증 통과) |
| `VADLAW_etri_tiny_kd_nolcf_split_distill.py` | **A안 student** (scene 0.3 / status 0.5, dropout 0.3) — 대기 |
| `VADLAW_etri_tiny_fast_eval_split_distill.py` | 위 student의 eval config (구조 일치 검증 통과) |
| `VADLAW_etri_tiny_kd_nolcf_feature_distill.py` | **B안 student** (weight 0.3, `aux_bev_motion_temporal=True`) — 대기 |

### `tools/diff_eval_config.py` (신규)
학습 config와 eval config가 **같은 망을 만드는지** 두 모델을 실제로 build해서
state_dict shape로 대조한다. config dict 비교로는 부족하다 — 학습 전용 옵션이
모듈 폭을 바꾸는 경우가 있기 때문. 네 쌍(A/B × teacher/student) 전부 통과 확인.

```bash
python tools/diff_eval_config.py <train_config> <eval_config> [ckpt]
```

**이 도구가 즉시 잡아낸 것 두 가지** (둘 다 크래시 없이 조용히 틀렸을 것):
1. A student config에 `ego_fut_dec_hidden_dim`을 안 적어서 576으로 기본값이 잡혔다.
   도너는 512폭 → mmcv는 `strict=False`로 로드하므로 **디코더 전체가 랜덤 초기화된 채
   22시간 학습**될 뻔했다.
2. eval 체인의 base가 `aux_bev_motion=False`다. B안은 loss 전용이라 무해했지만,
   A안 student는 추정망이 그 descriptor를 읽으므로 **추론 경로 요구사항**이다.

---

## 8. 실행 상태와 체크포인트 경로

### ★ [확정 2026-09-12] Qwen KD 라인 중단, GT 직행으로 전환

**결정**: stage1에서 Qwen KD를 끄고 실제 GT 궤적 손실을 켠다.

**근거 (전부 측정)**:
1. Qwen teacher hold-out 성능 = **0.3511m** (파인튜닝한 train split에선 0.1798m).
   그동안 인용해온 0.1798은 자기가 학습한 데이터에서 잰 허수였다.
2. teacher 우위가 **속도 지식의 지문**을 그린다 — STOP 1.0배(우위 0), LANE_KEEP 3.02배.
   프롬프트에 `vel_x=7.48, acc_x=-0.97`이 그대로 들어가 있었다. student는 접근 불가.
3. `loss_plan_reg=0.0`의 근거를 추적하니 규정이 아니라 *"matching the original VAD stage1
   recipe"* ([VAD_etri_tiny_stage1.py:304](projects/configs/VAD/VAD_etri_tiny_stage1.py#L304)).
   그 레시피 전제("stage1에선 planner를 안 건드림")는 KD를 넣는 순간 이미 깨졌다.
4. 궤적 수준 증류는 이미 두 번 실패했다 — `privileged_distill` 0.4897(무변화),
   `loss_plan_kd` 13%(교란된 비교).

**중단한 작업**: ego state 제거 재파인튜닝(35h 예정). 그 측정은 "teacher 품질"이라는
*대리 지표*인데, 결국 stage1+stage2를 돌려야 실제 효과를 알 수 있다. GT 직행이 더 싸고
더 직접적이다. (프롬프트에서 ego state 빼는 코드 자체는 `--no-ego-state` 옵션으로 남겨둠)

**남은 미검증 반론**: GT를 48 epoch 걸면 planner 과적합/인지 방해 우려가 있고, 부정확한
KD가 약한 정규화 역할을 했을 수도 있다. 이번 A/B가 그걸 판정한다.

**실행 중 잡은 문제 3개**:
- `kd_weight=0`은 손실만 끄고 **데이터 로딩은 안 끈다** — `LoadTeacherWaypoints`가 dataset
  생성 시점에 캐시 파일을 열어서 크래시. mmcv는 리스트를 통째로 교체하므로 파이프라인 전체를
  다시 정의해 제거해야 했다.
- 3090 컨테이너에 `law_pretrained_nus.pth`가 없고 `law_pretrained_nus_nolcf.pth`라는 다른
  변형만 있었다. 호스트 원본을 복사(A5000과 크기 대조 489227785로 동일 확인).
- `scripts/_common.sh`의 `require_gpu_free`가 `grep -c` 0건일 때 `"0\n0"`을 반환해
  "integer expression expected"로 죽던 버그 수정.

### 현재 실행 중
- **A5000** (`ssh 10.10.52.49`, docker `gyuz_split2`): B안 teacher — **학습 완료 (2026-09-10 00:02, epoch_12)**
  - work_dir `work_dirs/stage2_kd_lcfon_diag`
  - [측정] iter 2500에서 `loss_plan_reg` 0.0231 (compliant 베이스라인은 iter 3000에서 0.0268) → ego_lcf가 실제로 기여 중
  - **[측정 2026-09-10] hold-out L2 = 0.2542m** (`./scripts/eval_l2.sh B teacher` 상당,
    raw-image 마운트 문제로 임시 컨테이너 `gyuz_split2_evaltmp`에서 실행 — 아래 인프라 메모 참조)
    - L2@1s 0.1072 / L2@2s 0.2347 / L2@3s 0.4206
    - LANE_KEEP 0.2560 (3321/3321, 전체의 대부분), U_TURN 0.6631 (26개, 가장 나쁨), STOP 0.0129 (가장 좋음)
    - 판정: 0.20 이하("정상")에는 못 미치지만 0.25~0.30 "진행 가치 있음" 구간. compliant 베이스라인
      0.4885m 대비 약 48% 낮음.
  - **[측정 2026-09-10] `--zero-ego-lcf` 대조군 = 8.886m** (정상 0.2542m 대비 **35배 붕괴**)
    - LANE_KEEP 0.256 → **10.01** (39배). LANE_CHANGE_L 0.303 → 10.63.
    - STOP 0.0129 → **0.0131 (사실상 무변화)**. 정지 상태는 속도 입력 없이 시각만으로 판단 가능.
    - 판정: teacher가 ego_lcf를 실제로, 매우 강하게 쓰고 있음이 확인됨. distillation target으로 유효.
    - LANE_KEEP이 가장 심하게 무너지는 것은 기존 측정("LANE_KEEP 오차 = 속도 추정 오차의 적분")과 일치.
    - **[측정 2026-09-10] 0.2166 계보를 텐서 단위로 해부한 결과 — 원인 1순위는 zero-pad**

      0.2166을 낸 stage1(`work_dirs/stage1_etri_v2/`)의 실제 config가 디스크에 남아 있어 확인함.
      파일명이 `VAD_etri_tiny_stage1_cached.py` — **`_kd` 없음. Qwen KD가 없었다.**
      게다가 `loss_plan_reg/bound/col/dir` 전부 `loss_weight=0.0`. 즉 stage1에서
      `ego_fut_decoder`를 학습시키는 신호가 **하나도 없었다.**

      그럼 그 decoder 값은 어디서 왔나 — nuScenes LAW pretrain과 직접 대조:
      | | layer0 norm비 | 방향 cosine |
      |---|---|---|
      | 옛 stage1 epoch_48 vs nuScenes | **0.8870** | **1.0000** |

      두 층의 축소 비율이 동일하고 cosine이 정확히 1.0 → **학습이 아니라 순수 weight decay**.
      즉 그 stage1의 기여는 오직 **초기값 전달**이었다.

      결정적 차이는 `ego_fut_decoder.0.weight`의 마지막 8열(ego_lcf 입력):
      | | scene 512열 평균\|w\| | **ego_lcf 8열 평균\|w\|** |
      |---|---|---|
      | nuScenes LAW pretrain | 0.0216 | **0.0336** |
      | 옛 stage1 (0.2166 계보) | 0.0191 | **0.0298** |
      | `kd_lcfon_diag` (0.2542) | (KD로 학습됨) | **정확히 0** (zero-pad) |

      0.2166은 ego_lcf 칸이 **scene 열보다 1.5배 큰 사전학습 가중치**로 시작했고, 현재
      B teacher는 **0에서 시작**해 stage2 12epoch만에 쓸모를 만들어야 했다.
      merge 메타도 확인: `merge_world_model_keys: 41` → nuScenes에서 가져온 건 `bev_world_model`뿐,
      `ego_fut_decoder`는 stage1것이 520폭 그대로 전이됨 (zero-pad 없음).

      부수 발견: 옛 decoder 최종층이 `(72, 520)` = **6 모드**. 지금은 84 = 7 모드(STOP 추가 후).
      0.2166은 STOP 클래스가 생기기 전 모델이다. (현재 B teacher에서 STOP은 L2 0.0129로 최저 구간)

    - **[Claude 제안, 미측정] 나중에 추가된 4개 메서드가 teacher에 해로운가**
      깨끗한 on/off ablation은 **하나도 없다.** 기존 수치(0.4885 prismlcf / 0.5635 surgical /
      0.6419 aux_bev_motion_**feedback**)는 전부 계보가 같이 바뀐 교란된 비교이고, 전부
      compliant 모델 대상 — ego_lcf-ON teacher에서 이 넷을 잰 적은 없다. 아래는 코드 구조 추론.
      | 메서드 | 판단 | 근거 |
      |---|---|---|
      | `echo_cycle_weight=0.1` | **해롭다고 봄 → 뺀다** | gradient가 waypoints를 통해 planner에 직접 도달([VAD_LAW.py:735-750](projects/mmdet3d_plugin/LAW/VAD_LAW.py#L735)). "GT에 가까운 궤적"이 아니라 "world model이 왕복 가능한 궤적"으로 당김. compliant엔 보상(motion 일관성)이 있었지만 teacher는 ego_lcf를 정확히 알아 보상이 없음 |
      | `bev_refine_steps=3` | 해롭기 어려움 → 유지 | 모든 refine MLP 마지막 층이 zero-init([VAD_head.py:824,855](projects/mmdet3d_plugin/VAD/VAD_head.py#L824)) → **정확히 no-op에서 시작**. 게다가 decoder **이후** 단계라 `ego_plan_hidden` forward 경로 밖 |
      | `aux_bev_motion` (w=0.5) | B안 유지 / A안은 무방 | stage2에서 크기가 작음(0.0675 vs 총 loss ~5). **B안**: teacher의 vision feature가 motion을 인코딩하게 해 target을 vision으로 도달 가능하게 만듦 → 유지. **A안**: 64-d target이 `ego_lcf_embed_net(raw 8)`, 즉 ego_lcf만의 순수 함수라 이 메서드와 무관 → 유지 근거 약함 |
      | `prism_posterior_lcf_idx` | 유지 | teacher는 prior도 ego status를 봄(`prism_prior_net(ego_feats)`, [VAD_head.py:1693](projects/mmdet3d_plugin/VAD/VAD_head.py#L1693)) → prior/posterior 정보 격차가 student보다 작음. `prism_z_proj`도 zero-init |

      **규모 감각**: 전체 격차가 0.2542 − 0.2166 = **0.038m**. echo_cycle이 원인의 전부여도 그게
      상한이고, student에게 전달되는 몫은 더 작다.
- **A5000** (docker `gyuz_split2`): **새 stage1 — ego_lcf ON + Qwen KD** (2026-09-10 01:2x 시작,
  02:0x 버그 수정 후 재시작)
  - config [VAD_etri_tiny_stage1_cached_kd_lcfon.py](projects/configs/VAD/VAD_etri_tiny_stage1_cached_kd_lcfon.py),
    work_dir `work_dirs/stage1_etri_split_301_75_10hz_kd_lcfon`, 48 epoch, ETA 약 1~2일
  - 실행: `./scripts/run_B_stage1_lcfon.sh`
  - 목적: zero-pad 문제를 근본 해결. KD가 **520폭 decoder를 ETRI에서 직접 학습**시키므로
    nuScenes 물려받기(0.2166 계보)보다도 나을 여지가 있음. **B안·A안(8차원 v2) teacher 둘 다** 여기서 갈라짐
  - **[버그, 1epoch 시점에 발견·수정]** 최초 launch에서 `ego_fut_dec_hidden_dim`을 안 적어
    기본값이 520(=in_dim)이 됨. teacher v2 config는 hidden=512를 기대하므로 layer 0/2/4
    **세 층 전부** shape 불일치 — mmcv strict=False라 에러 없이 decoder가 통째로 랜덤
    재초기화될 뻔함 (48시간 낭비). config에 `ego_fut_dec_hidden_dim=512` 명시 후 재시작.
    부작용: hidden이 520→512로 바뀌어 **nuScenes pretrain의 decoder도 더 이상 상속 불가**
    (그전엔 이미 hidden 520이라 nuScenes에서 왔었음, 비 1.62). 지금은 랜덤+KD로 처음부터.
  - [측정] `loss_plan_kd` 0.8744(iter100) → 0.0811(iter800, 버그판) / 재시작 후 0.0520(iter1100)
    → 0.0435(epoch5). 살아있음 확인 — stage1에서 decoder를 학습시키는 유일한 신호
  - **[측정] ego_lcf 8열이 랜덤 초기화에서 실제로 자라는 추세 (scene 대비 비)**:
    | epoch | 비 |
    |---|---|
    | 1 | 1.074 |
    | 2 | 1.239 |
    | 3 | 1.413 |
    | 4 | **1.564** (= 옛 0.2166 계보의 stage2 시작값과 동일 수준) |

    B teacher(zero-pad, 12epoch 후 0.29)·A teacher v1(zero-pad, 11epoch 후 0.147) 대비 압도적으로
    빠르게 자람. "랜덤 시작이라 12epoch 안엔 못 자란다"는 우려 해소. 44epoch 더 남음.
  - nolcf stage1과의 diff 검증: `ego_fut_decoder`의 Linear 3개만 512→520,
    나머지 모듈 전부 동일 (의도한 변수 하나만 격리됨)
  - 완료 후: `tools/merge_stage1_world_model.py` → `stage2_init_merged.pth`,
    그 다음 [VADLAW_etri_tiny_kd_lcfon_v2.py](projects/configs/VAD/VADLAW_etri_tiny_kd_lcfon_v2.py)(B안)
    또는 [VADLAW_etri_tiny_kd_lcfemb8_teacher.py](projects/configs/VAD/VADLAW_etri_tiny_kd_lcfemb8_teacher.py)(A안 v2)로 stage2.
    **zero-pad surgery 불필요** (이미 520폭)
  - stage1은 `type='VAD'`(VADLAW 아님)라 `echo_cycle_weight`가 애초에 없음 — 이번 변경과 무관

- **3090** (로컬, docker `gyuz_split_3090`): A안 teacher v1(64차원) — **학습 완료 (2026-09-10 04:01, epoch_12)**
  - work_dir `work_dirs/stage2_kd_lcfemb_teacher`
  - [측정] `stage2_init_merged_lcfemb64.pth` size mismatch 없이 로드, iter 100에서 `loss_plan_reg` 0.0198
  - **[측정 2026-09-10] hold-out L2 = 0.2328m** — B teacher(0.2542)보다 좋음.
    **단 교란된 비교임**: A teacher v1은 `plan_reg_ts_weight_mode='cumulative'`가 켜져 있고
    B teacher v1은 기본값('position')이었다 — 임베딩 방식 말고 손실 가중치도 동시에 다름.
    v2끼리는 이 변수를 통일해 임베딩 방식만 남김 (아래 새 stage1 섹션 참조)
    - L2@1s 0.0937 / L2@2s 0.2124 / L2@3s 0.3922
    - LANE_KEEP 0.2315, U_TURN 0.6157(가장 나쁨), STOP 0.0089(가장 좋음)
  - **[측정] `--zero-ego-lcf` 대조군 = 8.307m (35.7배 붕괴)** — B teacher(35배)와 비슷한 강도로
    ego_lcf에 의존. teacher로서 유효함 확인
  - eval config 정합성은 `tools/diff_eval_config.py`로 통과 확인 후 측정

- **3090**: A안 **student v1(64차원)** — 2026-09-10 04:1x 착수, ETA 약 1일
  - work_dir `work_dirs/stage2_kd_nolcf_split_distill`
  - teacher = 위 A teacher v1 epoch_12. eval config 정합성 `diff_eval_config.py` 통과 확인 후 착수
  - [측정] iter 100: `loss_plan_reg` 0.0763, `loss_scene_distill` 0.0407(raw≈0.136),
    `loss_status_distill` 0.3991(raw≈0.798). `aux_bev_motion_head` size mismatch는 예상된 것
    (`aux_bev_motion_temporal=True`로 입력이 256→1024로 넓어져 도너에 없는 모듈이 새로 생김 — B student
    검증 때도 동일 패턴)
  - **목적**: Scheme A(분리 증류)가 실제로 전달되는지 첫 측정. 판정 기준: compliant 베이스라인
    0.4885 대비 — 0.44 이하면 전달됨, 0.48 근처면 전달 안 됨

### 주요 체크포인트
| 경로 | 내용 |
|---|---|
| `work_dirs/stage1_etri_split_301_75_10hz_kd_nolcf/stage2_init_merged.pth` | KD stage1 → compliant stage2 초기값 (512폭) |
| `.../stage2_init_merged_lcfon8.pth` | 위 + 8칸 zero-pad (B안 teacher 초기값, 520폭) |
| `.../stage2_init_merged_lcfemb64.pth` | 위 + 64칸 zero-pad (576폭). **A안 teacher와 student가 공용**한다 — 둘 다 디코더 첫 Linear가 576→512이고 마지막 64칸이 0이다. student용 별도 surgery 불필요 (확인함) |
| `work_dirs/stage2_lcfon_tpshortcut_teacher_v2/epoch_10.pth` | 중단된 TP teacher (TP 오염 있음, fallback) |
| `work_dirs/EvoDriveVLA_code/result/etri_teacher/teacher_model_fulldata_3B_3epoch/checkpoint-3900` | **Qwen fulldata finetune, step 3900에서 중단·보존.** 재개하려면 여기서 resume |

### 인프라 메모
- 3090 = `10.10.52.54:7777`, A5000 = `10.10.52.49:22022` (SSH config에 등록됨)
- 학습 산출물은 컨테이너 안에서 root 소유 → 호스트에서 `rm` 불가. **`docker exec`로 지워야 함**
- A5000 컨테이너는 `data/train` 심볼릭 링크(→ `/media/vcl/SSD-DATA/pys/ETRI_e2e_dataset/train`)가
  컨테이너 밖을 가리켜 **원본 이미지 접근 불가**. 학습은 geometry cache를 쓰므로 무관하나, raw
  이미지가 필요한 eval은 마운트 추가한 임시 컨테이너 필요. `scripts/eval_l2.sh`의 `in_container`는
  본 컨테이너(`gyuz_split2`)에 고정돼 있어 이 경우엔 못 쓰고 수동으로 돌려야 함:
  ```bash
  ssh -p 22022 vcl@10.10.52.49 "docker run -d --name gyuz_split2_evaltmp --gpus all \
    -v /media/vcl/SSD-DATA/gyuz/LAW_split:/workspace/VAD \
    -v /media/vcl/SSD-DATA/pys/ETRI_e2e_dataset/train:/workspace/VAD/data/train \
    --shm-size=16g etri-vad:cu128 sleep infinity"
  # 이후 docker exec gyuz_split2_evaltmp ... 로 eval_holdout_l2_and_tinfer.py 직접 실행
  # 끝나면 docker rm -f gyuz_split2_evaltmp 로 정리 (본 컨테이너는 안 건드림)
  ```
- 디스크 정리 완료: 로컬 23G + A5000 59G 확보 (구 실험 체크포인트 삭제)

### RTK (토큰 절감 CLI) — 제한적으로만 사용 [측정 2026-09-09]
`~/.local/bin/rtk` v0.48.0 설치됨. **전역 훅(`rtk init -g`)은 의도적으로 설치하지 않았다.**

측정 결과 (이 저장소 기준):
| 명령 | 원본 | rtk | 판정 |
|---|---|---|---|
| `tree projects/mmdet3d_plugin` | 12030 B | 5252 B | 쓸 만함 |
| `ls -la projects/.../VAD/` | 689 B | 217 B | 쓸 만함 |
| `git status` | 878 B | 528 B | 쓸 만함 |
| `read train.log` (기본 `-l none`) | 131409 B | 131409 B | 절감 0 |
| `read train.log -l minimal` | 131409 B | 134385 B | **더 커짐** |
| `read train.log -l aggressive` | 131409 B | 717 B | **위험 — 아래 참조** |

**`rtk read -l aggressive`를 학습 로그에 쓰지 말 것.** 131409 B → 717 B로 줄이면서
`loss_plan_reg`가 있는 12줄을 **전부** 버렸다. 학습 로그의 진단 내용은 loss 값 그 자체이므로
남는 게 없다. 이 프로젝트의 주된 실패 유형이 "조용히 불완전한 정보"이고(6절 버그 이력),
eval config 버그를 잡은 결정적 단서가 `size mismatch for prism_posterior_net.0.weight`
한 줄이었다. 출력 필터를 기본값으로 켜면 그런 단서를 놓친다.

정리: `rtk tree` / `rtk ls` / `rtk git status`만 선택적으로 쓴다.
로그와 체크포인트 로딩 출력은 **반드시 필터 없이** 본다.

---

## 9. 미확정 사항과 다음 할 일

### 즉시
- [x] `stage2_init_merged_lcfemb64.pth` 생성
- [x] 3090에서 A안 teacher 학습 시작 + 첫 iteration 검증
- [x] A안 student 구현: vision→64 추정, modality dropout, 512/64 이중 증류 loss
- [x] eval config 4쌍 전부 `tools/diff_eval_config.py`로 구조 일치 검증
- [ ] **A안 student는 아직 forward를 실제로 돌려본 적이 없다.** config build와 합성 텐서
      단위 테스트만 통과했다. `run_A_student.sh`의 `verify_start`가 iteration 100 또는
      Traceback까지 기다리므로 실패는 즉시 드러나지만, teacher가 끝난 뒤에야 확인된다

### teacher 완료 후 (약 22시간 뒤)
- [ ] 두 teacher의 2-frame L2 측정 + `--zero-ego-lcf` 대조군
- [ ] **분기 판정**: teacher가 0.2166 수준을 재현하는가?
  - 0.20 이하 → 정상, 격차 0.29, 진행
  - 0.25~0.30 → 격차 0.2, 진행 가치 있음
  - **0.35 이상 → 뭔가 잘못됨.** 같은 정보에 기법을 더 얹었는데 나쁘다면 원인부터 찾을 것
- [ ] student 학습 (A/B 병렬)

### 미검증 가정 [Claude 제안 수준]
- **stage1의 `loss_plan_reg=0.0` (GT 궤적 손실 끔)** — 근거를 추적해보니 규정이 아니라
  [VAD_etri_tiny_stage1.py:304](projects/configs/VAD/VAD_etri_tiny_stage1.py#L304)의
  *"matching the original VAD stage1 recipe"*, 즉 **원 논문 관례를 따른 것**이다.
  원 레시피의 논리는 "덜 익은 BEV 위에서 planner를 학습시키면 잘못된 특징에 의존하게 된다"
  (커리큘럼 학습). **그런데 KD를 넣으면서 그 전제가 이미 깨졌다** — `loss_plan_kd`가
  stage1에서 planner를 학습시키는 유일한 신호라고 config docstring에 명시돼 있다.
  planner를 학습시키기로 한 이상 남는 질문은 "무엇으로"뿐인데, 지금은 **정확한 GT를 끄고
  부정확한 Qwen 예측(held-out 0.3511, ego state 제거 후엔 더 나쁠 전망)을 목표로 삼고 있다.**
  반론: GT를 48 epoch 걸면 planner 과적합/인지 방해 우려가 있고, 부정확한 KD가 약한 정규화
  역할을 했을 수도 있다. **양쪽 다 측정된 적 없음.**
  실험 후보 (stage1 1회 = 24h라 전부는 무리):
  | | kd_weight | loss_plan_reg |
  |---|---|---|
  | (가) 현재 | 0.2 | 0.0 |
  | (나) GT만 | 0.0 | 1.0 |
  | (다) 둘 다 | 0.2 | 1.0 |

  → **분기 조건**: 진행 중인 no-ego-state 재파인튜닝의 held-out L2가 우리 student(0.4218)보다
  나쁘면 (가)는 자동 탈락 — teacher가 student보다 못한 답을 가르치는 셈이므로 (나)로 간다.
- `feature_distill_weight=0.3` — cosine loss가 0으로 안 내려가므로 후반에 궤적 손실을 압도하지 않게 낮춘 값. 근거는 논리뿐, 튜닝 안 됨
- `plan_reg_ts_weight_mode='cumulative'` — [측정 2026-09-10, 단 A안과 교란됨] A teacher v1(0.2328,
  cumulative 켜짐)이 B teacher v1(0.2542, 꺼짐/기본값)보다 좋았음. 최소 불안정하진 않다는 증거는
  됨. 단 **A/B teacher v1의 유일한 다른 차이가 아니었다** — A는 임베딩(64d)도 동시에 다름.
  0.2328 vs 0.2542는 "cumulative 효과"와 "임베딩 효과"가 섞인 교란된 비교였다.
  → **[확정] B teacher v2에도 `plan_reg_ts_weight_mode='cumulative'` 추가함** (전엔 안 들어있었음).
  이제 A teacher v2 vs B teacher v2는 임베딩 방식(8d 학습 vs raw 8칸)만 다르다 — 깨끗한 비교 가능.
  절대 효과 크기는 여전히 미분리.
- `aux_bev_motion_temporal=True` — loss-only 모드에서는 한 번도 측정된 적 없음
  (temporal+feedback 조합은 실패했으나 그건 feedback 탓으로 진단됨)
- **student가 도달 가능한 L2** — Claude 예측 0.44~0.46 (격차의 20~40% 전달 가정).
  사용자는 0.2대를 목표로 함. **이 간극은 미해결.**
  Claude의 논거: target_point는 이미지에 없는 외부 정보라 vision으로 복원 불가.
  단 B안은 TP-free teacher를 쓰므로 이 논거의 적용 범위는 ego_lcf 부분에 한정됨

### 열린 쟁점
- **can_bus 경로의 규정 해석.** `can_bus[7:16]`(가속도/각속도/속도)이 `can_bus_mlp`를 통해
  BEV 쿼리에 임베딩된다. 베이스라인 기본값이고 planner 직접 입력은 아니지만,
  `d(loss_plan_reg)/d(ego_lcf_feat)==0` 컴플라이언스 증명이 이 경로는 다루지 않는다.
  제거 시 10.4m로 붕괴하므로 성능 기여는 절대적. **감사 시 쟁점이 될 수 있음**

---

## 10. Codex 의견

*(아직 없음. Codex가 남긴 내용은 이 절에 보존하고 삭제하지 않는다.)*
