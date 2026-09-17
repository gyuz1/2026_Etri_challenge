# SHARED_CONTEXT

ETRI 2026 자율주행 챌린지 · LAW_split 트랙 공유 작업 기록.
Claude와 Codex가 공유한다. 최종 갱신: 2026-09-12.

**지금 무엇이 돌고 있는지는 8절 맨 앞**, 파이프라인 전체 설계와 근거는 [PIPELINE.md](PIPELINE.md).

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

- **[사용자 2026-09-15] GT 는 커맨드처럼, 규정을 위반하지 않는 선에서만 쓴다.**
  규정상 추론 입력으로 허용된 GT 는 **커맨드뿐**이다. 나머지(target_point, ego_lcf)는
  **학습 라벨로만** 쓰고, 추론 때는 망 안에서도, 망 출력 중 하나를 고르는 후처리에서도
  읽지 않는다 (Q&A A1: "신경망에 입력되지 않은 정보를 활용한 후처리는 금지").
  `tools/audit_pipeline.py` 가 제출·평가 스크립트의 target_point 읽기를 FAIL 로 잡는다.

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
| **`ego_fut_dec_hidden_dim` 미지정** | (사전 발견) | 기본값 576이 잡혀 512폭 도너와 불일치 → `strict=False`라 디코더 전체 랜덤 초기화 | `diff_eval_config.py`가 차단 |
| **576 도너 ↔ 520 모델** | (사전 발견) | 8-d 임베딩으로 바꿨는데 `load_from`이 64-d 시절 도너 그대로 | `audit_pipeline.py`가 차단 |
| **임베딩 의미 불일치** | `loss_plan_reg` 0.3884 (정상 0.0198) | 폭은 520으로 **정확히 일치**해 아무도 경고 안 함. 도너 ego 열은 raw 물리값으로 학습됐는데 랜덤 MLP 출력을 먹였다 | zero-init residual로 0.0068 (**57배**) |
| **`aux_bev_motion_temporal` 미상속** | (사전 발견) | `frames=3, grid=8`만 켜면 **둘 다 조용히 무시**되고 descriptor가 단일 프레임 global mean으로 떨어진다. 그건 shift-invariant라 자기운동이 원리적으로 안 보인다 | `audit_pipeline.py` 검사 3b |
| **`kd_weight=0`이 로더를 안 끔** | dataset 생성 시 크래시 | `LoadTeacherWaypoints`는 손실 가중치와 무관하게 캐시 파일을 연다. mmcv는 리스트 필드를 통째로 교체 | 파이프라인 전체 재정의 |
| **`reset_stream()` KeyError** | eval/제출 스크립트 크래시 | 두 스크립트가 `prev_frame_info`를 재구성하면서 새 키 `prev_bev2`를 빠뜨림. `VAD.py`는 무조건 읽음 | 양쪽 수정 |
| **감사 도구 자체가 no-op** | (사전 발견) | `_common.sh`가 `--cfg-options`로 ann_file을 덮어쓰므로 **config에 적힌 경로는 학습에 안 쓰인다.** 그 경로(`.causal_regen_split_301_75`, `_10hz` 없음)는 두 머신 어디에도 없어서 누수 검사가 매번 조용히 자기를 건너뛰었다 | `--ann-dir`로 실제 경로 해석 + 없으면 FAIL. 첫 실측: train 301 / val 75, 겹침 0 |
| **stage2 도너가 구 계보를 가리킴** | (사전 발견) | 재구축 후에도 `load_from`이 옛 stage1 work_dir 그대로. 그 merge가 디스크에 있고 **decoder 폭도 같아서** 모든 shape 검사를 통과하며 아무 경고도 안 낸다. 실제로는 2프레임 grid4 BEV 인코더를 싣는다 | 새 work_dir로 재지정. 파일이 아직 없는 상태가 **정상**이다 |
| ★ **가속도 블록이 학습에서 죽어 있었다** | (2026-09-12 발견, 재시작으로 해결) | `VAD.obtain_history_bev`가 history BEV를 **하나만** 반환하고 `forward_pts_train`에 `prev_bev2` 인자가 아예 없었다. 헤드는 항상 None 분기를 타 6144차원 descriptor의 **마지막 1/3을 0으로** 채웠다. 두 aux head의 4096:6144 열은 gradient를 못 받아 초기값 그대로, BEV 인코더는 가속도를 인코딩하라는 압력을 전혀 안 받았다 | **stage1 2시간 폐기** (조기 발견으로 48시간이 아니라 2시간) |
| ★ **제출 모델 추론에서도 가속도 블록이 0** | (2026-09-12 발견) | `VADLAW.forward_test`가 `prev_bev2`를 **전혀 다루지 않았다** — scene reset·`simple_test` 전달·shift 셋 다 없음. VADLAW는 `VAD.forward_test`를 의도적으로 우회하므로 부모 쪽 스트림을 고쳐도 **제출 모델엔 아무 효과가 없다.** 학습은 3프레임(`obtain_history_prediction`은 정상), 평가는 2프레임 | 사전 발견 |
| ★ **학습 프레임 간격이 무작위** | (2026-09-12 발견) | `prepare_train_data`가 history 후보 3개 중 하나를 무작위로 버린다(upstream VAD 증강). 그래서 **67% 샘플에서 두 간격이 불일치**하고, 2차차분에 `v·dt`가 섞인다 — 실측 5.28m 대 진짜 `a·dt²` 0.12m, **46배**, 부호는 버려진 후보에 따라 뒤집힘. 평가는 항상 (5,5) 고정이라 train/eval 불일치이기도 | `history_sampling='fixed'`. 실제 dataset으로 100% 균등 확인 (이전 33%) |
| ★ **`prev_bev2`가 회전된 텐서** | (2026-09-12 발견) | 인코더가 `prev_bev`를 **in-place로 yaw 정렬**하는데, 모든 스트림이 그 텐서를 다음 스텝 `prev_bev2`로 넘겼다. `prev_bev`의 descriptor는 회전 전에 뜨므로 `d1`과 `d2`가 **다른 연산자**가 되고 `d1−d2`는 가속도가 아니라 회전 잔차가 된다. 실측: 전형적 0.5s yaw(1.57°)가 평행이동의 14.4%, 선회(3~5°)는 28~44% | 회전 전 사본(`prev_bev_pristine`)을 따로 유지. 학습·추론 4개 경로 전부 |
| **평가 기본창이 2프레임** | (2026-09-12 발견) | `eval_l2.sh` 기본 `--frame-offsets 0,-5`. 3프레임 모델을 그걸로 채점하면 `prev_bev2`가 끝까지 None이라 가속도 블록이 0 — **모델 코드에서 고친 결함을 플래그 기본값이 되살린다** | config에서 `aux_bev_motion_frames`를 읽어 창을 정하고, 모자라면 거부. 제출 스크립트도 동일 |
| **teacher 체크포인트 무검증 로드** | (사전 발견) | `load_checkpoint`도 `strict=False`. teacher config와 체크포인트가 어긋나면 **랜덤 초기화된 모듈이 증류 타깃을 만든다**. 증류 손실은 멀쩡해 보인다 | 감사에 teacher 로드 검증 추가. 제출 스크립트에도 동일 가드 (v1 ckpt + v5 config로 동작 확인: 불일치 10, 누락 4 → 거부) |
| **증류 타깃에 상수열** | (2026-09-12 발견) | `ego_lcf` 5,6번 열(ego_length 4.635 / ego_width 1.89)은 std가 **정확히 0**인데 cosine 증류 타깃 안에 있었다. 제곱노름의 평균 21.5%, **정지 샘플에선 99.7%** → student가 상수 둘만 내놓아도 cosine이 거의 맞는다 | `ego_status_distill_idx=(0,1,2,3,4,7)`. `ego_feats`는 8열 유지라 decoder 520폭·도너 전이 영향 없음 |
| ★★ **dropout 이 만든 train/eval 속도 편향** | nodistill 0.5018 (자기 도너 0.4807 보다 나쁨) | 인코더·디코더 dropout(p=0.1)이 켜진 특징으로 속도 read-out 과 플래너 속도가 보정됨. 추론(dropout 끔)에서 속도 +0.52 m/s(+5%), 계획 +0.29 m/s. 크래시·경고 없음, shape·config·데이터 전부 정상 | dropout 끈 fine-tune 로 검증 중. 진단 도구 2개 |
| ★ **stage2 eval config 두 개가 조용히 틀려 있었다** | (2026-09-15 발견, 평가 전 차단) | (a) `..._fast_eval_split_distill8_3f_fut_g8.py`와 `..._fast_eval_kd_lcfemb8_teacher_best.py`가 **최상위 `model = dict(...)`를 두 번** 썼다. config는 파이썬이라 두 번째가 첫 번째를 **통째로 대체** → student/nodistill eval은 grid 4(1536폭)로 떨어져 `ego_status_est_net`(플래너 슬롯 입력)이 `strict=False`로 **랜덤 초기화**될 뻔했고, teacher eval은 frames=3·grid=8·future motion을 잃었다. (b) teacher eval config에 `ego_lcf_embed_residual=True`가 **없었다** — shape는 안 바뀌고 raw ego 열 덧셈만 빠진다. **원인 공통: `run_stage2_best.sh`가 감사에 `--eval-config`를 안 넘겨 parity 검사가 한 번도 안 돌았다** | 감사에 `--eval-config` 전달, 감사에 **동작 플래그 parity**(shape 무관, loss/dropout 제외 전 설정 일치) 추가 — 누락 flag로 음성 대조해 FAIL 확인. eval 도구가 shape 불일치/누락 가중치면 **채점 거부**. A5000 코드 사본 동기화(VAD_head/VAD.py가 goal-grid 이전 버전이었음, 추가분은 기본값에서 no-op이라 teacher 학습엔 영향 없음 — diff 확인) |

**교훈**: 크래시 없이 조용히 틀리는 유형이 가장 위험하다.
학습 시작 전 (a) config diff로 의도한 차이만 있는지, (b) 데이터가 실제로 들어오는지,
(c) eval config가 학습 config와 구조적으로 일치하는지 반드시 확인한다.

**위 표의 "조용한" 항목들이 `tools/audit_pipeline.py` 한 번으로 전부 걸러진다.**
긴 학습 전에 반드시 돌린다 (7절 참조).

**단, shape 검사로는 원리적으로 못 잡는 유형이 있다.** 위 ★ 두 건이 그것이다 —
descriptor는 1/3이 0이든 아니든 6144차원 그대로다. 그래서 검사 도구를 두 개 더 만들었다:
`tools/check_accel_block_live.py`(호출 경로를 읽는다, 학습·추론 양쪽 + `reset_stream` 키 일치),
`tools/check_accel_block_trained.py`(체크포인트 weight로 결과를 읽는다),
`tools/check_descriptor_sensitivity.py`(descriptor가 실제로 shift-sensitive한지 측정),
`tools/verify_motion_targets.py`(손으로 적은 상수를 데이터와 대조).
**교훈: "config에 켰다"와 "실제로 값이 흐른다"는 별개이고, 후자는 값을 봐야만 안다.**

---

## 7. 변경한 파일과 상태

### `projects/mmdet3d_plugin/VAD/VAD_head.py`
- **`ego_plan_hidden`** outs에 추가 — `ego_fut_decoder[:2](ego_feats)`, PRISM 주입 이전 결정론적 값. B안 증류 대상. [확정, 미검증]
- **`plan_reg_ts_weight_mode`** (`'position'` 기본 / `'cumulative'`) — 지표가 델타를 cumsum하므로
  초기 델타 오차가 이후 모든 위치를 밀어냄. 실효 가중치 `[1.0, .69, .39, .25, .11, .056]` vs
  기존 `[.31, .31, .14, .14, .06, .06]`. [확정, 미검증]
- **`ego_lcf_embed_dim`** — A안 teacher용. raw 8칸 대신 `Linear(8→h)→ReLU→Linear(h→dim)`.
  **64 → 8로 축소** (576폭은 어떤 stage1도 도너가 될 수 없어 그 열이 0에서 출발했다).
  8이면 `ego_fut_dec_in_dim`이 520 — stage1 도너와 정확히 일치. [확정, 검증됨]
- **`ego_lcf_embed_residual`** — `ego_status = raw + embed_net(raw)`, 마지막 층 zero-init.
  step 0에서 raw와 bit-identical. 없을 때 `loss_plan_reg` 0.3884 → 있을 때 **0.0068**. [측정]
- **`aux_bev_motion_temporal` / `_frames` / `_grid`** — BEV descriptor를 3프레임 × 8² 격자로.
  `cat([cur, d1, d1−d2])`, 2차 차분이 곧 가속도(`d1−d2 == a·dt²` 폐곡선 검증). 입력 6144차원.
  **셋은 세트다** — `temporal`이 꺼지면 나머지 둘이 조용히 무시된다. [확정, 검증됨]
- **`aux_bev_motion_norm`** — 성분별 std로 L1 정규화. 정규화 전엔 vx 49.3% + speed 49.4%가
  L1을 먹고 yaw_rate는 0.1% → 회전이 사실상 무감독이었다. 이후 20.1/20.1/19.6/17.1/13.4/9.7. [측정]
- **`aux_bev_future_motion`** — BEV descriptor에서 **미래 3초 속도 프로파일**을 회귀.
  `speed_gt = |Δ_i| / FUT_TS_INTERVAL_S` (GT가 per-step 델타라 0.5로 나눈다). 현재 상태만
  완벽히 알아도 천장이 0.2708인데 3초 뒤 속도가 현재와 평균 0.80 m/s 다르다. [확정, 미검증]
- **`ego_status_est_dim` / `_dropout`** — student가 BEV descriptor에서 자기상태를 추정해
  decoder에 넣는다. 2026-09-04 Q&A가 허용한 "신경망 자신이 vision으로 추론한 값". [확정]

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

### configs

**현재 계보 (2026-09-12 재구축)** — 이것만 쓴다:

| 파일 | 용도 | 상태 |
|---|---|---|
| `VAD_etri_tiny_stage1_best_nolcf.py` | stage1, student 도너 (512폭) | **3090 학습 중** |
| `VAD_etri_tiny_stage1_best_lcfon.py` | stage1, teacher 도너 (520폭) | **A5000 학습 중** |
| `VADLAW_etri_tiny_kd_lcfemb8_teacher_best.py` | stage2 teacher (ego_lcf → 8d 임베딩 + residual) | stage1 대기 |
| `VADLAW_etri_tiny_kd_nolcf_split_distill8_3f_fut_g8.py` | **stage2 student = 제출 모델** | teacher 대기 |
| 위 둘의 `..._fast_eval_...` 짝 | eval config | 감사 통과 |

**이전 계보** (2프레임 grid4 descriptor 시절, 기록용):
`VADLAW_etri_tiny_kd_lcfon_diag.py` (B teacher, 0.2542) ·
`VADLAW_etri_tiny_kd_lcfemb_teacher.py` (A teacher v1 64d, 0.2328) ·
`VADLAW_etri_tiny_kd_nolcf_split_distill.py` (A student v1, **0.4218 — 현 최고 compliant 기록**)

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

### `tools/audit_pipeline.py` (신규 2026-09-12) — 긴 학습 전 필수

`diff_eval_config.py`의 상위 집합. **6절 버그 표의 "조용한" 항목을 전부 검사한다.**
검사 항목은 하나도 가정이 아니다 — 전부 실제로 일어났고 그때 에러가 안 났던 것들이다.

```bash
python tools/audit_pipeline.py <train_config> \
    --eval-config <eval_config> \
    --val-ann data/etri/.causal_regen_split_301_75_10hz/vad_etri_infos_temporal_val_split.pkl
```

| # | 검사 | 걸러내는 것 |
|---|---|---|
| 1 | 도너 decoder shape | 576 도너를 520 모델에 → planner 랜덤 초기화 |
| 2 | train/eval 망 일치 | 학습과 다른 망을 채점 |
| 3 | 규정 | 제출 모델에서 `ego_lcf_feat_idx`/`target_point_shortcut` |
| 3b | **조용한 no-op** | `frames`/`grid`만 켜고 `temporal`이 꺼진 경우, `future_motion`만 켜고 `motion`이 꺼진 경우, `idx`에 정규화 없음, `norm` 길이 불일치 |
| 4 | teacher/student descriptor 대칭 | student가 teacher가 인코딩한 적 없는 구조를 재현하게 됨 |
| 5 | 데이터 누수 | train/val scene 겹침, Qwen KD 캐시 사용 여부 |

**검사 5는 config의 `ann_file`을 믿지 않는다.** `_common.sh`의 `launch_train`이
`--cfg-options`로 세 필드를 전부 덮어쓰기 때문 — config에 적힌 경로는 학습에 쓰이지 않고,
두 머신 어디에도 존재하지 않는다. `--ann-dir`(기본 `_10hz`)로 launcher와 같은 방식으로
해석하며, **`_common.sh`의 `ANN_DIR`을 바꾸면 이 기본값도 같이 바꿔야 한다.**

검사 1은 **nuScenes warm-start를 예외 처리**한다 — stage1은 `ego_fut_mode`가 달라 decoder가
원리적으로 전이 불가이고, 거기서 재초기화되는 게 정상이다. 도너 파일명에 `law_pretrained_nus`가
있으면 [OK]로 통과시킨다. (그 구분이 없어서 stage1 감사가 오탐을 냈었다)

### `tools/check_accel_block_live.py` (신규 2026-09-12) — shape로 못 잡는 것

3프레임 가속도 블록이 **학습과 추론 양쪽에서** 실제로 채워지는지 호출 경로를 읽어 검사한다.
`obtain_history_bev`가 두 프레임을 반환하는가 / `forward_pts_train`이 `prev_bev2`를
넘기는가 / `forward_test`가 스트림을 유지하고 **overwrite 전에 shift** 하는가.

마지막 항목이 중요하다 — shift가 overwrite 뒤에 오면 `prev_bev2`가 `prev_bev`를
alias 해서 2차 차분이 **항등적으로 0**이 된다. 역시 크래시 없음.

`type(model).forward_test`를 읽으므로 **VADLAW가 override한 버전을 검사한다.**
부모 것만 고치고 넘어가는 게 실제로 일어난 실수다.

### `tools/check_accel_block_trained.py` (신규 2026-09-12)
체크포인트에서 결과를 확인한다. gradient를 못 받은 블록은 초기 분포를 그대로 유지하므로,
방금 build한 모델의 해당 열 및 확실히 학습된 속도 블록과 통계를 비교한다.
**epoch 1 체크포인트가 나오면 반드시 한 번 돌릴 것.**

### `tools/verify_motion_targets.py` (신규 2026-09-12)
손으로 적어 넣은 상수 두 개를 데이터로 검증한다. 둘 다 [측정] 통과:
- `FUT_TS_INTERVAL_S = 0.5` vs 실측 **0.5001s** (움직이는 샘플 82314개)
- `aux_bev_motion_norm` 6개 전부 실제 std와 **정확히 일치**, 정규화 후 L1 분담 16.7% 균등

### `tools/kinematic_oracle_ceiling.py` (신규 2026-09-12)
등속/등가속 오라클을 대회와 동일한 L2 windowing으로 측정. 2절의 0.5965 / 0.2708 출처.

### `PIPELINE.md` (신규 2026-09-12)
전체 구조도, 서버별 작업, 설정 하나하나의 측정 근거, 규정표, 감사 절차, 일정.
"무엇을 왜 이렇게 설정했는가"를 한 곳에 모은 문서.

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

### ★ [측정 2026-09-12] A student v1 = **0.4218m** — feature distillation 은 전달된다

compliant 베이스라인 0.4885 대비 **−13.7%**. 궤적 수준 증류가 두 번 실패했던 것과 대조적.

| 커맨드 | A student | 베이스라인 | 변화 |
|---|---|---|---|
| **LANE_KEEP** (오차의 85%) | **0.4350** | 0.5079 | **−14.4%** |
| LANE_CHANGE_L | 0.6125 | 0.6918 | −11.5% |
| TURN_RIGHT | 0.4439 | 0.4995 | −11.1% |
| TURN_LEFT | 0.5021 | 0.5559 | −9.7% |
| STOP | 0.0120 | 0.0133 | −9.8% |
| U_TURN (26개) | 0.6753 | 0.6520 | **+3.6%** ⚠️ |

가장 큰 개선이 LANE_KEEP에 왔다는 게 핵심 — 헤드룸이 거기 85% 있다.

### ★ [측정 2026-09-12] 운동학 오라클 — 우리 천장이 어디인지

`tools/kinematic_oracle_ceiling.py`, val split, 대회와 동일한 L2 windowing:

| 오라클 | L2 |
|---|---|
| 정지 가정 | 12.5175 |
| **완벽한 속도만** | **0.5965** |
| **완벽한 속도 + 가속도** | **0.2708** |

커맨드별로 LANE_KEEP은 0.5543 → **0.2524**.

**함의 3가지**:
1. **가속도가 속도보다 큰 레버다** (0.5965 → 0.2708, 55%). 그런데 `aux_bev_motion_idx`가
   ax/ay(2,3)를 **아예 제외**하고 있었다.
2. 우리 student(0.4218)는 두 오라클 **사이**에 있다 = "속도는 알고 가속도는 모르는" 상태.
3. **1등 0.14는 완벽한 운동학 오라클(0.2708)마저 48% 앞선다.** 운동학 외삽으로 설명 불가 —
   미래 기동 자체를 장면에서 예측한다는 뜻. 우리 아키텍처로 도달 가능한 영역이 아니다.
   → **0.2 미만은 현 구조로 불가능.** 증류 천장이 teacher(0.2328)인데 목표가 그보다 낮다.

### ★ [측정 2026-09-12] T_infer 페널티 재계산 — 3프레임이 생각보다 싸다

`etri_test_submit.py`에 실측치가 있었다: `--bev-only-history` 사용 시 history 프레임
**27.7ms**, 채점 프레임 **61.9ms** (3090, fp16). BEV가 bit-identical이라 궤적은 안 바뀜.

| 프레임 | T_infer | 페널티 |
|---|---|---|
| 2 (bev-only 적용) | 89.6ms | **x1.0 (없음)** |
| **3** | 117.3ms | **x1.087** |
| 4 | 145.0ms | x1.225 |

**우리 eval은 이 플래그를 안 쓰고 있었다** → 135ms, x1.176. 실효 점수가 0.4218 × 1.176
= 0.496이었던 것. **플래그만 켜면 모델 변경 없이 15% 개선.** `scripts/eval_l2.sh`에 추가함.

그리고 3프레임 본전 문턱이 **8%**뿐이다(가속도 레버는 오라클 기준 55%).

### ★ [측정 2026-09-12] 임베딩 zero-init residual — 폭이 맞아도 조용히 깨지던 전이

A teacher v2/v3 를 ego_lcf-ON stage1 위에 올리자 iter 100 에서 `loss_plan_reg` 0.3884,
`prev_frame_loss_waypoint_0` 2.77 이 나왔다. A teacher v1(0.0198 / 1.03)보다 20배 나쁘다.
**폭은 520 으로 정확히 맞아서 shape 검증도, mmcv 도 아무 경고를 안 냈다.**

원인: stage1 도너의 ego 8열은 **raw 물리값**(vx·speed 평균 10.6 m/s 등)으로 학습됐고,
그 열의 `mean|w| 0.0481` 은 scene 열의 `0.0223` 보다 **2.2배 크다** — 도너가 vision 보다
자기상태에 더 기대고 있었다는 뜻. 그런데 임베딩 teacher 는 그 자리에 랜덤 초기화된
`ego_lcf_embed_net` 출력(O(1))을 넣는다. 스케일 문제가 아니라 **의미가 다른 입력**이다.

수정: `ego_lcf_embed_residual=True` → `ego_status = raw + embed_net(raw)`, 마지막 층 zero-init.
step 0 에서 raw 와 **bit-identical**(최대 차이 0.0 확인).

| | loss_plan_reg | waypoint_0 |
|---|---|---|
| 수정 전 | 0.3884 | 2.77 |
| **수정 후** | **0.0068** | **0.1068** |
| (참고) A teacher v1 | 0.0198 | 1.03 |

**57배 개선**, v1 대비로도 2.9배 낮다. 이게 없었으면 22시간짜리 실험 둘이 모두 나쁜
출발점에서 돌 뻔했다. `plan_bev_refine_mlp`/`prism_z_proj` 와 같은 zero-init residual 관례.

### ★ [확정 2026-09-12] stage1부터 최선 구성으로 전면 재구축 — 진행 중이던 A/B 전부 중단

**[사용자]**: *"지금 A/B안 비교가 중요한게아니라 가장 최선의 방법을 적용하는게 중요해
시간 생각하지말고 가장 베스트 초이스로 돌려"* → *"아니 다시해야되면 그것부터 다시하라니까"*
→ *"지금 너가생각한 최선의 파이프라인을 시간아깝다고 버리지말고 그것부터 차근히하라고"*

**중단한 것**: A teacher v2(A5000) / v3(3090). 둘은 `aux_bev_motion_frames`의 통제된
A/B였는데, 사용자가 원한 건 비교가 아니라 최선 구성이었다. 게다가 **둘 다 도너 stage1이
구식**이었다 — 2프레임 grid4 descriptor 위에서 학습된 stage1에 3프레임 grid8 stage2를
올리는 구조라, stage2가 stage1이 인코딩한 적 없는 표현을 12 epoch 만에 만들어내야 했다.

**새로 돌리는 것** (둘 다 48 epoch, 2026-09-12 08:0x 시작, ETA ~2일):

| 서버 | config | work_dir | 산출물 |
|---|---|---|---|
| 3090 | `VAD_etri_tiny_stage1_best_nolcf.py` | `stage1_best_nolcf` | student 도너 (decoder 512폭) |
| A5000 | `VAD_etri_tiny_stage1_best_lcfon.py` | `stage1_best_lcfon` | teacher 도너 (decoder 520폭) |

두 lineage가 필요한 이유: teacher는 ego status를 `ego_feats`에 넣어 decoder가
`embed_dims*2 + 8 = 520`폭이고 student는 512다. 한쪽 폭 도너는 다른 쪽에 전이 안 된다.
그리고 student가 **실제 ego status로 학습된 계보를 물려받으면 안 된다**(규정).

**stage1에 새로 들어간 것** (전부 기존 [측정] 근거):
- `kd_weight=0` + `loss_plan_reg=1.0` — Qwen KD hold-out 0.3511 (암기), GT 직행
- `aux_bev_motion_temporal=True`, `frames=3`, `grid=8` — 셋이 세트. temporal이 꺼져 있으면
  frames/grid가 **조용히 무시**된다(아래 버그 참조)
- `aux_bev_motion_idx=(0,1,2,3,4,7)` + 성분별 std 정규화 — 가속도 추가, yaw_rate 무감독 해소
- `aux_bev_future_motion=True` — 미래 3초 속도 프로파일. 현재 상태의 오라클 천장이 0.2708인데
  3초 뒤 속도가 현재와 평균 0.80 m/s 다르다

**launch 전 shape 직접 확인** (6144 = 3프레임 × 32 proj채널 × 8²):
```
best_nolcf   aux_head 입력 (256, 6144)   future head (256, 6144)   decoder (512, 512)
best_lcfon   aux_head 입력 (256, 6144)   future head (256, 6144)   decoder (512, 520)
```

### ★ [확정 2026-09-15] 제출 스크립트가 추론 때 GT target_point 를 읽고 있었다 → 제거

`etri_test_submit.py` 가 `|TP| < 0.5m` 이면 STOP 궤적을 골랐다. STOP 은 미래 궤적에서
만든 학습 전용 파생 라벨이라 테스트 커맨드에 절대 나오지 않고, 그걸 복구하려고 TP 를
읽은 것이다. 2026-09-04 에 "선택만이라 해결됨" 으로 적혔으나 **A1 원문과 맞지 않는다.**

**교체**: 망이 스스로 추정한 속도(`aux_bev_motion_head` 의 speed 열, 추론 때도 계산됨)
< 0.1 m/s 이면 STOP. 오히려 더 잘 잡는다 [측정, val STOP 라벨 기준]:

| 규칙 | 정밀도 | 재현율 |
|---|---|---|
| GT \|TP\| < 0.5 m (제거) | 1.000 | 0.871 |
| GT 현재 속도 < 0.1 m/s | 0.846 | **0.936** |

실제 제출은 GT 속도가 아니라 **추정** 속도를 읽는다. 추정 정확도는 v1 probe 에서 speed
R² 0.985 였으나 새 모델에서 STOP 판별 정확도는 **[미측정]** — 평가 때 확인할 것.

**평가 불일치도 같이 드러났다.** val 의 `gt_ego_fut_cmd` 에는 STOP 이 들어 있어서
`eval_holdout_l2_and_tinfer.py` 는 STOP 샘플(**6.4%**)에 정답 모드를 공짜로 줬다.
**지금까지의 모든 val L2 는 이 점에서 테스트보다 낙관적이다.** `--test-commands`
플래그로 테스트 조건(STOP→LANE_KEEP 대체 후 추정 속도로 STOP 선택)을 재현한다.
STOP 이전 원래 커맨드는 pkl 에 저장돼 있지 않아 LANE_KEEP 으로 근사한다.

### ★★ [측정 2026-09-14] `bev_residual_refine` 이 L2 를 21~25% 깎아먹고 있었다

재학습 없이 **지금 있는 모든 체크포인트에 즉시 적용되는** 개선이다.

`--disable-bev-refine` (모듈은 로드된 채 호출만 건너뜀, 같은 체크포인트·같은 데이터):

| stage1 epoch | refine 켬 | refine 끔 | 개선 |
|---|---|---|---|
| 24 | 0.6395 | **0.4799** | −25.0% |
| 36 | (중단) | **0.4886** | — |
| 48 | 0.6089 | **0.4807** | −21.1% |

**시간은 거의 안 준다** — 196.2ms → 194.9ms (1.3ms). 즉 지연-정확도 교환이 아니라
**그냥 궤적을 나쁘게 만들고 있었다.**

**가장 시사적인 관찰**: refine 끈 값이 epoch 24~48 내내 0.48 근처로 평평한데
(0.4799 / 0.4886 / 0.4807) 켠 값은 0.6395 → 0.6089 로 개선된다.
**거친 궤적은 epoch 24 에 이미 수렴했고, 나머지 24 epoch 은 refine 이 망친 것을
되돌리는 데 쓰였다.**

**[정정]** 2절 "나중에 추가된 4개 메서드" 표의 `bev_refine_steps=3` 항목:
"해롭기 어려움 → 유지, 근거: 모든 refine MLP 마지막 층이 zero-init 이라 정확히
no-op 에서 시작" — **틀렸다.** no-op 에서 시작한다는 것은 48 epoch 의 gradient 가
어디로 데려가는지에 대해 아무것도 말해주지 않는다. 그 판단은 측정이 아니라 추론이었다.

**[확정]** 평가·제출에서 끈다(재학습 0, 즉시 21~25%). 학습에서도 끄는 것이 더 나은지는
미측정 — `stage2_nodistill_best` 가 `bev_residual_refine=False` 로 학습 중이다.

**[미측정]** v1 student(0.4218) 에 이걸 적용하면 얼마가 되는가. v1 도 refine 을 켜고
학습·평가했으므로 같은 손해를 보고 있었을 가능성이 높다. GPU 가 나면 바로 잴 것.

### ★ [측정 2026-09-12] descriptor 가 실제로 자기운동을 본다 — 전제 검증

`tools/check_descriptor_sensitivity.py`. BEV 100x100 / pc_range x -30~30 → 셀당 0.600 m.
평균 속도 10.57 m/s × 0.5s = 5.29 m = 9 셀.

| | Δ/scale |
|---|---|
| 1 셀 (0.6m) | 0.0775 |
| 9 셀 (**실제 변위 5.4m**) | **0.5712** |
| 18 셀 (10.8m) | 0.7558 |
| **global mean pool, 9 셀** | **4.9e-10** |

global mean 은 원리적으로 shift-invariant다 — `aux_bev_motion_temporal=False`가 만드는
상태가 정확히 이것이고, 그 경우 `cur − prev`가 아무리 빨라도 0이다.
**grid 4 였다면 30.1% → grid 8 은 1.90배 민감.** grid 8 선택의 실측 근거.

### [측정 2026-09-12] 미래 속도 타깃 — 실제로 신호가 있다

`gt_ego_long_fut_trajs` 10 스텝 중 6 스텝(3초) 사용. **valid flag 100%** (마스킹 손실 없음).
현재 속도 평균 10.57 vs 3초 뒤 10.58 — 평균은 같지만 **샘플별 절대차 0.842 m/s**.
이게 운동학 외삽이 버리는 부분이고, future head 가 노리는 값이다.

### ★ [측정 2026-09-12] epoch 1 — 가속도 열이 **실제로** gradient 를 받았다

`tools/check_accel_block_trained.py`. gradient 를 못 받은 블록은 초기 분포를 그대로
유지하므로, 버그 상태였다면 가속도 열의 std 변화가 **정확히 0** 이어야 한다.

| | 속도 블록 std 변화 | 가속도 블록 std 변화 | 비 |
|---|---|---|---|
| 3090 `aux_bev_future_motion_head` | 0.000456 | **0.000612** | 1.34 |
| 3090 `aux_bev_motion_head` | 0.000066 | 0.000060 | 0.91 |
| A5000 `aux_bev_future_motion_head` | 0.000447 | **0.000601** | 1.34 |
| A5000 `aux_bev_motion_head` | 0.000067 | 0.000075 | 1.12 |

검사 FAIL 기준(속도의 2% 미만)을 크게 상회. **두 줄기 독립 증거 확보** —
호출 경로(`check_accel_block_live.py`)는 "흐를 수 있다", weight 는 "흘렀다".

두 계보 수치가 거의 동일한 것도 정상이다 (차이는 `ego_lcf` 뿐).

**~~관찰(미해결)~~ → 해소**: "`aux_bev_motion_head` 가 초기 대비 0.75% 밖에 안 움직였다"는
관찰은 **EMA 아티팩트였다**. 아래 항목 참조. raw 파라미터로 다시 재면 +22% 다.

### ★ [측정 2026-09-12] 체크포인트의 **일반 슬롯은 EMA**, `ema_*` 가 raw 다

이 저장소의 체크포인트를 읽는 모든 분석에 영향을 준다. 반드시 알고 읽을 것.

stage1 config 에 `EMAHook(momentum=0.0002)` 이 있고 **priority 가 HIGH** 다.
mmcv 의 `EMAHook.after_train_epoch` 이 `_swap_ema_parameters()` 를 호출하고,
`CheckpointHook` 은 NORMAL 이라 **swap 이 끝난 뒤에 저장**된다. 따라서:

- 체크포인트의 `pts_bbox_head.xxx` = **EMA 값**
- 체크포인트의 `ema_pts_bbox_head_xxx` = **raw 학습 파라미터**

추론이 아니라 측정으로 확정: epoch_1 → epoch_2 사이 이동량이
**일반 슬롯 165.1 / `ema_` 슬롯 375.0 — 일반 슬롯이 2.27배 덜 움직인다.**
(init 거리로만 보면 `img_backbone` 과 `pts_bbox_head` 가 반대로 나와 결론이 안 났다)

**왜 중요한가**: momentum 0.0002 는 유효 윈도우가 약 5000 iteration(~2.3 epoch)이라,
초기 체크포인트의 일반 슬롯은 학습이 아무리 잘 돼도 **초기값 근처에 머문다.**
그걸 읽고 "학습이 안 된다"고 판단하면 틀린다 — 실제로 그럴 뻔했다.

| epoch 2, 초기 대비 mean\|w\| 증가 | EMA 슬롯 | **raw 슬롯** |
|---|---|---|
| `aux_bev_motion_head` | 0.75% | **+22%** |
| `aux_bev_future_motion_head` | (epoch1) 4.9% | **+74%** |

`tools/check_accel_block_trained.py` 는 이제 `ema_*` 가 있으면 raw 를 우선해서 읽는다.

**부수 사실**: merge 는 일반 슬롯(=EMA)을 가져간다. 이게 stage2 초기값으로 옳은 선택이고
평가도 같은 슬롯을 쓴다. 바꿀 것 없음. 도너에 `ema_*` 612개가 따라오지만 로드 시 무시된다.

**[Claude 제안, 미검증]** 2절의 "옛 stage1 decoder = nuScenes 가중치의 weight decay
(cosine 1.0000)" 분석도 EMA 슬롯을 읽은 것이다. 48 epoch(103k iter)이면 EMA 가 충분히
수렴했을 것이라 결론은 유지될 가능성이 높지만, raw 로 다시 재본 적은 없다.

### [측정 2026-09-12] 미검사 구간 5곳 전수 점검 — 전부 통과

버그가 아니었지만 측정 전까지는 알 수 없던 것들. 앞으로 다시 의심하지 않기 위해 남긴다.

**1. `plan_reg_ts_weight_mode='cumulative'` 가중치** — 실제 지표(`dist[:2],dist[:4],dist[:6]`
평균의 평균)에 수치미분을 걸어 delta_j 민감도를 측정하고 코드가 계산하는 값과 대조:

| step | 수치미분(실측) | cumulative(코드) | position(기존) |
|---|---|---|---|
| 0 | 2.2394 | 2.4000 | 1.8333 |
| 1 | 1.6687 | 1.6667 | 1.8333 |
| 5 | 0.1593 | 0.1333 | 0.3333 |

최대 상대오차 **cumulative 19.5% vs position 52.2%**. 완전 일치가 아닌 건 L2 norm의
방향 의존성 때문이고, cumulative 가 옳은 방향임은 확인됨.

**2. `refine_ego_trajs_with_bev`** — 제출 궤적에 직접 들어간다(`bev_residual_refine=True`,
steps 3). zero-init no-op 최대차 **6.6e-07**(fp32 cumsum↔역차분 반올림, 0.00066mm).
`grid_sample` 축 규약: 셀(h=20,w=75) 스파이크를 (x=15.30m, y=-8.85m)에서 샘플 → **1.0**,
x/y 를 바꾸면 0.0. `flat[h*W+w] → bev_map[:,:,h,w]` reshape 규약도 일치.

**3. `echo_cycle_weight=0.1`** — [사용자 질문에 대한 답] 랜덤 BEV 탐침은 분포 밖이라
결론을 못 낸다(그 측정은 폐기). **실제 학습 로그** 기준: v1 student 에서
0.00508 → **0.00079** (−84%), 같은 시점 `loss_plan_reg` 0.0121 의 **6.5%**.
퇴화해서 해로운 수준은 아니고, plan 에 주는 제약도 그만큼 작다. 유지하되 효과는 미미.

**4·5. 데이터 파이프라인 / eval config 정합성** — 학습은 geometry cache, 평가는 실시간
계산이라 다른 경로다. 프레임을 메타(`sample_idx`)로 맞춰 비교:
- **`lidar2img` 최대차 0.000e+00 (bit-identical)**, `img_shape` 양쪽 (448,768,3)
- `crop_keep_top` 5개 카메라 동일, `scale` 0.4 동일, `PadMultiViewImage` 는 eval 의
  `MultiScaleFlipAug3D` **안에** 있어 누락 아님
- 픽셀은 평균 **3.5%** 차이 (JPEG 디코드 경로). `reduced_decode=1`(전체 디코드)로
  바꿔보니 오히려 **7.1%** 로 악화 — 캐시가 half-decode 로 만들어졌다는 뜻이라
  **현재 설정이 이미 최선**. 바꾸지 않는다.

주의: 처음 비교에서 115% 차이가 나왔던 건 (a) `PhotoMetricDistortion` 증강과
(b) mmdet 이 `prepare_train_data` 가 None 이면 **다른 인덱스를 재추첨**하는 것 때문이었다.
인덱스가 같다고 같은 프레임이 아니다 — 메타로 확인할 것.

### [측정 2026-09-12] 수술 도구 사전 검증

`surgical_ego_fut_decoder_transfer.py --ego-lcf-n 0 --pad-input-cols 8`을 기존 512폭
merge 에 미리 돌려 확인: (512,512) → (512,520), **scene 512열 bit-identical**,
추가 8열 정확히 0, 값이 바뀐 키는 그 하나뿐, 누락 키 0, 기존 lcfemb8 도너와 완전 동일.
stage1 이 끝난 뒤 막히지 않는다.

**[2026-09-12 09:0x] 가속도 버그로 둘 다 재시작.** 위 버그 표의 ★ 항목 — 3프레임
가속도 블록이 학습 내내 0이었다. 2시간 손실. 재시작 후 ETA 1일 17시간.

수정 후 반드시 확인할 것: epoch 1 체크포인트에
`tools/check_accel_block_trained.py`를 돌려 가속도 열이 **실제로 gradient를 받았는지**
weight로 확인한다. 호출 경로 검사(`check_accel_block_live.py`)는 이미 통과했지만,
그건 "값이 흐를 수 있다"까지고 "흘렀다"는 weight로만 증명된다.

**[측정 2026-09-12] 첫 launch 시 iter 200 건강성 확인** (가속도 버그 발견 전) — 둘 다 정상:

| | 3090 nolcf | A5000 lcfon |
|---|---|---|
| `loss_plan_reg` | 0.2237 → **0.1043** | 0.2200 → **0.0733** |
| `loss_aux_bev_future_motion` | 0.7222 → 0.3156 | 0.6946 → 0.3431 |
| `loss_aux_bev_motion` | 0.4925 → 0.4720 | 0.4989 → 0.4816 |
| `loss_plan_kd` | 0건 | 0건 |
| Traceback | 0 | 0 |
| ETA | 1일 23:07 | 1일 22:27 |

`loss_plan_reg`가 살아 내려가는 것 = GT가 KD를 실제로 대체했다는 확인.
`loss_aux_bev_future_motion`이 존재하는 것 = 새 head가 실제로 forward에 들어갔다는 확인
(config만 켜고 조용히 no-op 되던 과거 패턴이 아님).

**이후 일정**: stage1 완료 → merge → teacher(A5000, 12ep) → student(3090, 12ep) → 평가.
전체 파이프라인·근거·규정표는 [PIPELINE.md](PIPELINE.md).


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

### 진행 중 (stage1, ~2일)
- [x] 최선 구성 stage1 두 계보 launch + iter 200 건강성 확인 (8절)
- [ ] 48 epoch 완주 대기. 중간 점검 포인트: `loss_aux_bev_future_motion`이 계속 내려가는가
      (미래 속도가 실제로 학습 가능한 신호인지의 첫 증거)

### stage1 완료 후
- [ ] `tools/merge_stage1_world_model.py`로 양쪽 merge
- [ ] student 도너에 8칸 zero-pad (`surgical_ego_fut_decoder_transfer.py --ego-lcf-n 0
      --pad-input-cols 8`) → 520폭
- [ ] **감사 필수**: 두 stage2 config 전부 `tools/audit_pipeline.py` 통과 확인
- [ ] teacher 학습 (A5000, 12ep) → L2 측정 + `--zero-ego-lcf` 대조군
- [ ] **분기 판정**: teacher L2가
  - 0.20 이하 → 정상, 격차 0.29, 진행
  - 0.23 근처 → 이전 A teacher v1(0.2328)과 동급. 3프레임/가속도/미래속도가 teacher엔
    효과 없었다는 뜻이지만, **student 쪽 이득은 별개다** (teacher는 이미 ego_lcf를 알아
    운동 정보에서 얻을 게 없고, student는 그게 유일한 통로다) → 진행
  - **0.30 이상 → 뭔가 잘못됨.** 원인부터 찾을 것
- [ ] student 학습 (3090, 12ep) → 최종 평가 `--frame-offsets 0,-5,-10 --bev-only-history`
- [ ] **비교 기준은 0.4218** (A student v1). 이번 재구축이 그걸 못 넘으면 stage1 재구축이
      헛수고였다는 뜻이므로 v1 계보로 되돌아가 제출한다

### [확정 2026-09-15 15:12] stage2 종료 → 즉시 추론 체인 (`scripts/chain_stage2_finish.sh`)
[사용자] "끝나면 일단 바로 추론돌려". 로그: 3090 `work_dirs/chain_stage2_finish.log`
- A5000 teacher 끝(예상 ~16:05) → epoch_12 3090 복사(md5) → **A5000에 distilled student 시작**
  (A5000은 평가 불가라 GPU를 놀리지 않기 위함. 3090엔 student 도너만 있어 A5000으로 복사함, md5 일치)
- 3090 nodistill 끝(예상 ~17:07) → GPU0 `nodistill --test-commands` → `nodistill`(val 명령),
  GPU1 `teacher --test-commands` → `teacher --test-commands --zero-ego-lcf` → 가속도 블록 검사
- 두 평가가 같은 머신에서 동시에 돌므로 **T_infer는 CPU 경합으로 약간 부풀 수 있다** — 제출 판단 전 단독 재측정
- teacher 평가에서 이상(0.30 이상 또는 zero-lcf가 거의 안 변함)이 나오면 A5000 student 중단
- [측정 17:10] student A5000 정상 시작(도너 로드, size mismatch 0, Traceback 0, scene/status distill 손실 존재).
  nodistill 17:08 종료 → 체인 폴링 대기(최대 3분)를 기다리지 않고 `scripts/run_stage2_evals.sh`로 17:10 직접 시작.
  두 평가 모두 체크포인트 대조 통과(shape 불일치 0, 누락 0). 각 ~35분 예상.
- **버그**: `verify_start`가 A5000(ssh→docker 이중 인용)에서 셸 함수 정의가 깨져 coreutils `cut`이 불리고
  `until` 루프가 영원히 돌았다. 09-12부터 A5000 컨테이너에 6개 누적, 정상 시작을 "시작 실패"로 보고.
  → 로그를 가져와 호스트에서 판정하도록 재작성, 30분 타임아웃. 누적 루프 전부 kill.

### ★ [확정 2026-09-17] 새 stage2 planner 설정 — TP 예측 head 제거, 선택만 (구현 전)
[사용자] "그럼 우리 떼는 걸로 하고 지금 설정 픽스 박고 정리해서 말해줄래?"
- **제거**: goal_cls/goal_off/goal_embed/goal_follow/goal_select, argmax 궤적 loss, 10스텝 decoder 연장. 비전으로 TP 를 예측하는 경로 전부.
- **head**: 이동 5개(LK, LC_L, LC_R, TURN_L, TURN_R)는 command별 공유 head `776 → 512 → 512 → 12` (ReLU; 입력 = 기존 ego_feats 520 + cell PE 256).
  U_TURN·STOP 은 칸·PE 없는 전용 head `520 → 512 → 512 → 12`. 출력은 기존과 같은 6스텝 delta.
- **초기화**: donor `ego_fut_decoder`(520→512→512→84)에서 command별로 복사 — 첫 층 PE 열 0, 마지막 층은 해당 command 의 12행. 시작 출력 = donor.
- **칸** (train 이동 frame 만, 칸 안 3초 궤적 차이 최소화 exact DP, 칸당 ≥50 frame·≥3 scene·폭 ≥4m; val = 칸 안 궤적 L2):
  | command | 칸 | 전방 경계 (m) | 좌우 경계 (m, 좌측+) | val |
  |---|---|---|---|---|
  | LANE_KEEP | 15×1 | 1, 12, 22, 31, 38, 45, 51, 56, 61, 66, 72, 82, 94, 101, 106, 117 | −20, 17 | 0.807 |
  | LANE_CHANGE_L | 11×1 | 1, 20, 25, 36, 44, 51, 58, 66, 78, 94, 103, 116 | −16, 12 | 1.155 |
  | LANE_CHANGE_R | 14×1 | 1, 23, 30, 35, 41, 45, 49, 53, 59, 64, 74, 81, 89, 100, 115 | −12, 18 | 0.934 |
  | TURN_LEFT | 4×2 | 1, 16, 21, 30, 48 | −2, 10, 27 | 0.995 |
  | TURN_RIGHT | 6×2 | 1, 12, 16, 20, 26, 33, 43 | −26, −11.5, 2 | 0.790 |
  이동 칸 합 60. TURN_LEFT 7×2 는 제약(폭 4m·3 scene) 불만족 → 4×2 (5×2 val 0.998, 7×1 1.261). 외곽 범위는 요청서 값, 하한만 1m.
- **PE**: 고정 buffer, 칸 중심 [전방, 좌측], 축별 64 주파수 × sin/cos = 256, 파장 geomspace(4, 400 m).
- **선택 규칙 (학습·추론 동일, 비학습)**: TP 전방 < 1m → STOP head (command 무관). 아니면 U_TURN → U_TURN head, 이동 command → TP 를 포함하는 칸(내부 경계는 다음 칸, 범위 밖은 외곽 칸).
  command 는 입력 그대로, TP 로 command 를 바꾸지 않음(정지만 예외). LAW history frame 은 각 frame 의 command·TP 로 선택, world model·echo 에는 선택 궤적.
- **loss**: 선택 후보 하나에만 기존 loss_plan_reg(마스크·스텝가중 동일) + plan_bound/col/dir. aux(long_horizon, bev_motion, bev_future_motion, ego_status_decode), world model rec, echo, history waypoint 유지.
  STOP 라벨인데 TP ≥ 1m 인 frame(원래 command 불명, train 660·val 149)은 ego waypoint loss 에서 제외. STOP head 학습 frame train 5672 (STOP 라벨 5420 + 다른 command 252).
- **TP 경로**: feature 생성에 절대 안 들어감. 추론 시 모델은 전 후보(+마스크)·U_TURN·STOP 궤적만 출력, 선택은 평가·제출 도구에서.
- **검증**: TP 교란 시 후보 전부 동일 / TP 로 gradient 없음, PE·경계 비학습 / 각 칸 중심이 자기 칸 선택·패딩 미선택 / 혼합·단일·STOP-only batch forward·backward / 시작 출력 = donor / history·current·inference 좌표·loss 일치 / 감사 통과.
- **평가·제출**: val L2 전체 + TP 구간(<1m, 1–9m, ≥9m)별 + command별, T_infer. `etri_test_submit.py` 에 같은 선택 규칙. 비교군: 대조군 0.3339 에 "TP<1m → STOP 모드" 선택만 추가한 수치.

### [측정 2026-09-17] 외부 요청서 "command별 head + 고정 cell PE + TP containing-cell 선택" 검토
[사용자] 요청서 `stage2_planner_request.md` 제공: 이동 command 6개는 command별 공유 head(입력 = scene feature 520 + 고정 sin/cos cell PE 256 → 512 → 512 → 12),
STOP 전용 head, command별 비균일 전방 경계(총 64 cell), 학습·추론 모두 TP 는 containing-cell 선택에만, 선택 후보에만 waypoint loss, CE/TP 회귀 없음.
- 코드·데이터 대조: command ID·좌표(x 전방, y 좌측+) 일치, test 7875 frame 전부 TP 있음, history meta 에 frame별 TP 있음. 경계 밖 train ≤1.4%.
  cell 지원 부족: TURN_LEFT 빈 cell 1, LC_R 최소 28 frame/1 scene, U_TURN 최소 8 frame.
- 선형 대리(train→val 공식 L2, `goal_grid_value.py` 함수 재사용, cell별 독립 ridge라 작은 cell 에 불리):
  | | command만 | 요청 grid | 연속 TP |
  |---|---|---|---|
  | 전체 | 0.2234 | 0.1928 (−13.7%) | 0.1191 |
  | LANE_KEEP | 0.2226 | 0.1877 | 0.1156 |
  | LC_L / LC_R | 0.3106 / 0.2461 | 0.3187 / 0.2465 (이득 없음) | 0.1449 / 0.1328 |
  | TURN_L / TURN_R | 0.3648 / 0.3245 | 0.2935 / 0.2944 | 0.2105 / 0.1780 |
  | U_TURN | 0.6508 | 0.6957 (악화) | 0.7339 |
  LC 대안: 좌우 3칸(−3, 3) LC_L 0.2876 / LC_R **0.2102**, 좌우 2칸(0) 0.2837 / 0.2334 → 차선변경은 전방이 아니라 **좌우 분할**이 값이 있다. U_TURN 은 어떤 분할도 악화(134 frame).
- [사용자] "유턴이랑 stop 은 격자 안 나누고 cmd 로만 해도 될 것 같아" — 대리 측정과 일치.
- 참고: LC_L 의 TP y 중앙값 −1.8m(10/90% −5.2/+1.5) — 좌측 차선변경인데 좌측 이동이 작다. 라벨 시점 문제 가능성, 미조사.
- [측정] command별 전방/좌우 분할 비교 (같은 대리, 좌우 경계는 train 분위수):
  | command | 전방만 | 좌우만 | 전방×좌우 (모든 cell 에 train 데이터 있는 안) |
  |---|---|---|---|
  | LANE_KEEP | 15×1 −15.7% (분위수 경계 −18.4%) | ×3 −3.0% | 15×3 −19.7% 이지만 min 21 frame → **전방만** |
  | LC_L | +2.6% (악화) | ×2 **−11.5%** | 11×3 −15.8% (min 8) → **좌우만** |
  | LC_R | +0.2% | ×3 **−12.1%** | 14×2 −11.0% (빈 cell) → **좌우만** |
  | TURN_L | 7×1 −14.6% | ×3 −17.0% | 4×2 **−20.7%** (min 68); 요청 7×2 −19.5% (빈 cell 1) |
  | TURN_R | 6×1 −9.3% | ×3 −9.8% | 4×3 **−23.8%** (min 94), 5×3 −24.4% (min 53) |
  LANE_KEEP 은 요청서 경계보다 등분위 경계가 낫다(−15.7% vs −18.4%). TURN_L/TURN_R train frame 수가 2660 으로 정확히 같음 — 미조사.
- [측정] 요청 PE(Dpe 256, s=100m, base 10000): 실제 anchor 범위(전방 −6~117, 좌우 −26~27m)에서 값이 0.5 이상 변하는 차원 **51/256**,
  최단 파장 100m. 7.6m 떨어진 인접 anchor 간 PE 거리 0.95 (|PE| 11.3 의 8%). base 10 이면 186/256 차원이 변함, 파장 100~965m.
- [사용자] 분포 그림 "전방이 더 길게 분포되어 있는데 왜 좌우만 나누지?" → 재측정, **위 'LC 는 좌우만' 결론 정정**.
  - 현재 상태(속도·가속·yaw rate)로 5초 TP 를 선형 예측한 뒤 남는 std (val): LK 전방 3.6 / 좌우 1.9m, LC_L 3.4/2.4, LC_R 3.7/2.0, TURN_L 3.9/3.9, TURN_R 3.4/3.7.
    원래 전방 std(LK 23.6, LC 15~16m)의 대부분은 속도 차이라 이미 아는 정보. 남는 전방 불확실성 3.4~3.9m 도 좌우보다 크다.
  - 앞 대리는 cell 마다 독립 ridge 라 작은 cell 이 많은 전방 분할에 불리했다. 공유 모델(상태 × cell one-hot, head 공유에 가까움)로 재측정:
    | command | 전방만 | 좌우만 | 전방×좌우 (min train frame) |
    |---|---|---|---|
    | LK | 15×1 −17.9% | ×3 −2.8% | 15×2 −19.4% (25), 15×3 −21.1% (21) |
    | LC_L | 8×1 −3.1% | ×2 −11.8% | **8×2 −15.5% (138)** |
    | LC_R | 8×1 −10.1% | ×3 −11.0% | 8×2 −12.7% (67), 8×3 −24.2% (38, val n=368 이라 분산 큼) |
    | TURN_L | 8×1 −15.5% | ×3 −15.9% | **4×2 −25.0% (68)**, 6×2 −30.8% (17) |
    | TURN_R | 8×1 −14.5% | ×3 −10.0% | **4×3 −26.1% (94)**, 6×3 −27.8% (28) |
    거친 전방 분할(2~5칸)은 대부분 속도를 다시 적는 것이라 효과가 작고, 전방은 칸이 남는 불확실성 수준으로 촘촘해야 값이 있다. 공유 모델에서는 모든 이동 command 에서 전방×좌우 조합이 최선.
  - 사용자 그림은 TURN_LEFT 가 lateral 음수 쪽이라 **좌우 부호가 우리 데이터(y 좌측 +)와 반대**이고, 표본 수도 우리 val 과 다름(LC_L 798 vs 813, U_TURN 72 vs 131). 요청서 경계(TURN_LEFT [−2,27])는 좌측 + 기준이라 일치.
- [사용자 2026-09-17] **칸 기준: "나눈 칸 안에서 target point 에 따라 3초 waypoint 궤적이 같기만 하면 됨"**.
  [측정] 이 기준으로 직접 최적화: train 에서 칸 안 3초 궤적(12차원 누적 위치)의 칸 평균 대비 제곱오차 합을 최소화하는 경계를
  전방/좌우 교대 exact DP(전방 1m, 좌우 0.5m 격자, 칸당 train ≥50 frame)로 구함. 채점 = val frame 궤적 vs train 칸 평균 궤적의 공식 L2 (상태 입력 없음).
  | command | command만 | 요청서 | 같은 칸 수 최적 경계 | 대안 |
  |---|---|---|---|---|
  | LK | 4.648 | 15×1 0.836 | 15×1 **0.811** | 20×1 0.756, 15×2 0.795 (min 62) |
  | LC_L | 3.567 | 11×1 1.163 | 11×1 **1.155** | 11×2 1.191 (좌우 분할 악화), 16×1 1.079 |
  | LC_R | 5.464 | 14×1 **0.890** (min 28) | 14×1 0.903 (min 51) | 14×2 0.979 (악화) |
  | TURN_L | 2.047 | 7×2 1.064 (빈 칸) | 7×2 **0.982** (min 76) | 7×1 1.261, 5×2 0.994 |
  | TURN_R | 1.309 | 6×1 1.018 | 6×2 **0.790** | 6×3 0.764, 8×2 0.757 |
  결론: 이 기준에선 **LK·LC 는 전방만**(좌우 분할은 칸 안 궤적 일치도를 개선 못 함), **회전은 좌우 분할 필수**(TURN_R 6×1→6×2 −22%).
  요청서 설계가 LK·LC·TURN_L 에서 거의 최적이고, 약점은 TURN_R 좌우 미분할과 TURN_L 빈 칸. 분위수 경계는 LK 에서 최적보다 나쁨(15×1 0.912).
  앞의 "LC 는 좌우 분할" 권고(현재 상태를 입력으로 넣은 대리)는 이 기준에서는 성립하지 않아 철회.
  최적 경계: LK F=[−6,9,17,24,31,38,45,51,56,61,66,72,82,94,102,117]; LC_L F=[−6,20,25,36,44,51,58,66,78,94,103,116];
  LC_R F=[−6,23,30,35,43,47,52,59,64,74,81,87,93,100,115]; TURN_L F=[−3,16,19,21,24,27,30,48] L=[−2,10,27];
  TURN_R F=[−3,12,16,20,26,33,43] L=[−26,−11.5,2]. 스크립트는 scratchpad `cell_dp.py`(아직 tools 에 없음).
- `tools/plot_command_cells.py` (신규): command별 칸 그림(train 점 / val 원 / test 별, 칸 (행,열)+train 수, 0개 빨강·<30 노랑) + 칸별 train/val/test 수와 칸 안 3초 궤적 L2 표.
  `--layout chosen|request`, 출력 `reports/cell_plots/<layout>/`. test 는 clip 당 채점 frame(token `_0`) 1125개만.
  [측정] test TURN_LEFT 67개 = 사용자 그림과 일치. train/val 은 10Hz 연속 frame 전부라 사용자 그림(952/245)보다 많다 → `--frame-stride` 옵션.
  chosen 에서 LC_R val 은 64m 이상 칸에 0개(val LC_R 368개가 전방 64m 이하에만 있음), TURN_R test 3.1% 범위 밖.
- ★★ [측정 2026-09-17] **test 와 train/val 의 정지 표본 분포가 크게 다르다** (칸 그림에서 발견).
  - command 비율: STOP train 6.7% / val 6.4% / **test 0.0%**. TURN_L 2.9/3.5/**6.0**%, U_TURN 0.1/0.6/1.1%.
  - 원인: STOP 은 우리가 만든 라벨(`etri_vad_converter_10hz.py:441`, 미래 3초 이동 < 0.5m 이면 raw command 를 STOP 으로 덮어씀). test 는 raw command 그대로라 STOP 이 없고, 정지 clip 이 LK/TURN 명령을 달고 온다. 우리 pkl 에 raw command 필드는 없음.
  - command 안에서 TP 전방 <1m 비율: LK train 0.3% / val 0.2% / **test 24.8%**, TURN_L 1.2 / 0.0 / **52.2%**, TURN_R 0.3 / 0.0 / **28.1%**.
    test clip 중 TP 전방 <9m 가 344/1125 (30.6%), train 9.9%·val 9.3%. train 에서 TP<1m 5672 frame 중 5420 이 STOP 라벨.
  - 함의: (1) 칸 선택 설계에서 정지 test clip 은 LK/TURN 의 가장 가까운 칸으로 가는데, 그 칸의 train 데이터에는 정지가 거의 없다(STOP 으로 빠져 있어서).
    (2) val 점수는 test 보다 정지 비중이 1/3 이라 test 를 대표하지 못한다 — 0.3339 포함 모든 val 수치에 해당.
  - [Claude 제안] TP 전방 <1m 이면 command 와 무관하게 STOP head 로 선택(학습·추론 같은 규칙, 생성 후 선택이라 칸 선택과 같은 성격).
    val 은 TP 거리 구간별 L2 와 test 입력 분포(command × TP 구간)로 가중한 점수를 함께 보고.
- [측정] cell PE 주파수 비교 (chosen 칸, command 안에서): 요청서 PE(s=100m, base 10000)는 가장 가까운 두 칸의 PE 거리가 |PE| 의 **2.8%**,
  칸 사이에서 0.5 이상 변하는 차원 20~32/256. 파장을 4~400m 로그 간격(축별 64주파수 × sin/cos)으로 바꾸면 **67.5%**, 128~194/256.
  [Claude 제안] 고정·비학습 PE + concat + command별 공유 head 구조는 유지, 주파수만 4~400m 로그 간격으로. 칸 중심만 인코딩(칸 폭 불필요), PE 열 0 초기화로 시작 출력 = donor.
- [사용자 2026-09-17] "비전만으로 TP 예측하는 head 때문에 초반 loss 가 올라가는 것 같은데 떼는 게 더 좋지 않을까? 우리가 설정한 선택만 하고"
  근거 [측정]: goal_cls gradient 가 planner feature 에서 plan_reg 의 5~9배·직교, goal 모델 3개 모두 대조군 대비 plan_reg +10~14% (시작부터).
  새 설계는 TP 예측(goal_cls/off/follow, argmax 궤적 loss) 없이 칸 선택 + 선택 후보 loss 만 → 샘플당 감독 대상 1개로 대조군(명령 모드 1개)과 같은 구조.
  대가: 이동 command 는 TP 없이 궤적을 못 고름(test 1125 clip 전부 TP 있음). 규정상 TP 선택이 막히면 대안 없음 — 필요 시 feature 로 gradient 안 보내는(detach) 예측 head 를 fallback 으로 추가 가능.

### [확정 2026-09-17 01:50] 커맨드별 앵커 + TP 선택 손실 — lat1(A5000) / adaptive(3090) 학습 시작
[사용자] "구현하고 빨리 두 개 서버에 올려, 다른 두 개여야 되는데 가능성 있어 보이는 후보 두 개", "좌우는 많이 두는 것보다 2개 1개씩 조금만 나누는 게 더 좋은 거야?"
- 배경 [측정, 09-16]: 대조군 **0.3339**, goalpred(12구간) TP 미사용 **0.3557** / `--select-goal-by-tp` **0.3456** (후보 12개, T_infer 146ms 로 추가 비용 없음).
  TP 선택 이득이 0.010 뿐인 원인 [Claude 분석]: planning loss 는 argmax 후보만 학습, 선택되는 후보 경로는 아무도 학습 안 함.
- 구현 (`VAD_head.py`, Claude + Codex):
  - `goal_anchors`: 커맨드별 가변 길이 `[x_c, x_w, y_c, y_w]` 표, 패딩은 마스크(`goal_anchor_utils.py`, float32 연산). 라벨 = 폭 단위 최근접 앵커.
  - **`loss_goal_select`**: 커맨드 모드에서 예측 목표가 GT TP 에 가장 가까운 후보를 골라(no_grad), 그 후보 궤적에 `loss_plan_reg` 와 같은 타깃·마스크·스텝가중 L1(가중 1.0).
    TP 는 **어느 후보를 학습할지 고르기만** 하고 디코더 입력은 네트워크 자신의 예측 목표. GT 궤적이 `loss()` 에만 있어서 forward 가 `goal_sel_fut_preds` 를 넘기고 `loss_planning` 에서 계산.
  - Codex 수정 반영: TP 선택 인덱스를 커맨드 모드에만 적용(다른 모드는 앵커 수가 달라 패딩을 집을 수 있던 내 버그), 구 12구간 config/체크포인트 호환(`goal_select_weight` 기본 0), eval 에 `goal_cand_mask`.
- 앵커 (`tools/make_goal_anchor_pair.py`, **train 만**, 보고서 `reports/goal_anchor_pair_train.json`): 두 변형이 전방 구간·`goal_scale=(115,25)`·donor·seed 0·스케줄 동일, **좌우만 다름**.
  지원 부족(<50프레임 또는 <3장면) 구간은 병합, 좌우 분할은 2-means 이득 ≥20%·간격 ≥1m·분할 후 지원 충족일 때만.
  | 커맨드 | lat1 | adaptive | train 프레임/장면 |
  |---|---|---|---|
  | LANE_KEEP | 23 | 41 | 72681 / 300 |
  | LANE_CHANGE L/R | 10 | 13 | 6085 / 98 |
  | TURN L/R | 9 | 19 | 5320 / 72 |
  | U_TURN | 1 | 1 | **134 / 2** (지원 부족 → 앵커 1개, 오프셋 회귀만) |
  | STOP | 4 | 4 | 6080 / 74 |
- config: `VADLAW_etri_tiny_clean_goalanchors_{lat1,adaptive}.py`, eval `VADLAW_etri_tiny_fast_eval_clean_goalanchors_{lat1,adaptive}.py`(후보 노출).
  실행 `scripts/run_goal_anchor_pair.sh both --train`, work_dir `stage2_goalanchors_{lat1,adaptive}_v1`. A5000 동기화 `sync_goal_anchor_pair.sh --sync`(sha256 전 파일 일치).
  `_goal_anchor_pair.sh` 의 `rg` 가 호스트 bash 에 없어 `find` 로 교체.
- [측정] 사전 점검 두 역할 모두 통과: 감사(train/eval 앵커 sha256 동일, 3c 불일치 없음), accel 경로 1.9533, goal 실측 —
  초기 CE = 0.5·ln(앵커 수) 정확, 라벨 왕복 오차 ≤1.2e-7, 초기 3초 출력 donor 와 차 2.4e-7, eval 에서 TP 를 1234.5 로 바꿔도 궤적·후보·마스크 동일(라벨 호출 0), gradient 유한.
- 평가 계획: 두 run 각각 TP 미사용 / `--select-goal-by-tp` (`scripts/eval_goal_anchor_pair.sh`). 판정: TP 선택이 0.3339 를 넘으면 채택, 아니면 목표/선택 방향 중단하고 대조군 제출.
  후보 41개의 T_infer 는 **미측정**.
- [측정] iter 100 (두 run 모두 Traceback 0, size mismatch 0, missing keys 는 새 헤드만 — 구 goalpred run 과 동일 목록):
  | | goal_cls (초기 기대) | goal_off | goal_follow | goal_select | plan_reg | aux_bev_motion | s/iter |
  |---|---|---|---|---|---|---|---|
  | A5000 lat1 | 1.4369 | 0.0355 | 0.0027 | 0.0140 | 0.0140 | 0.0263 | 1.379 |
  | 3090 adaptive | 1.7012 | 0.0213 | 0.0032 | 0.0142 | 0.0142 | 0.0264 | 1.363 |
  초기엔 선택 후보 = argmax 후보(goal_embed zero-init)라 goal_select = plan_reg 가 정상. 공통 손실 거의 동일.
  iter 100 속도는 구 goalpred(1.435 → 이후 0.764 s/iter, 약 22h)와 같은 수준.
  wandb: lat1 `spnpi3zg`, adaptive `bx51jthe`. 커밋 `b1fb3b6` (push 는 사용자).
- [측정 2026-09-17 13:00] 학습 손실, 같은 step 구간 평균 (4k iter 창):
  | step | control plan_reg | 구 goalpred plan_reg | lat1 plan_reg / select | adaptive plan_reg / select |
  |---|---|---|---|---|
  | 24–28k | 0.00981 | 0.01125 | 0.01096 / 0.01079 | 0.01134 / 0.01116 |
  | 52–56k | 0.00781 | 0.00887 | 0.00857 / 0.00842 | (48–52k 0.00920 / 0.00902) |
  | 100–104k (최종) | 0.00589 | 0.00659 | — | — |
  goal 모델 plan_reg 는 대조군보다 10~14% 높다(구 goalpred 와 같은 패턴, 앵커 교체로 안 좁혀짐). lat1 은 구 goalpred 보다 약 3% 낮고 adaptive 는 구 goalpred 와 비슷.
  **goal_select 가 plan_reg 보다 1.7~2% 만 낮음** → train 에선 argmax 앵커가 대부분 TP 최근접 앵커와 같다. 선택 이득은 분류기가 틀리는 val 에서만 판단 가능. train 손실로는 결론 불가.
- [사용자 전달 리뷰 반영 2026-09-17] 패딩 앵커 버그 2건 확인:
  (1) 라벨 계산 0/0 NaN → 실행 중 코드는 이미 해결(`build_anchor_table` 패딩 폭 1, 마스크 선적용). CPU 재현: TP (0,0) 이 유효 앵커 선택.
  (2) **평가 TP 선택이 패딩을 제외하지 않음 — 실제 버그**: `VAD.py` 가 `goal_cand_mask` 를 결과로 안 넘겼고 `eval_holdout_l2_and_tinfer.py` 는 전 후보 argmin.
      패딩 목표점 = 오프셋 원값(≈(0,0)) 이라 정지 근처 TP 에서 패딩이 뽑힐 수 있었다. 수정: 마스크 전달 + `nearest_valid_goal`, 마스크 없으면 에러.
      평가 전용 경로라 학습 중인 두 run 에는 영향 없음. 감사 두 config 통과. A5000 사본은 GPU 사용 중이라 미동기화(평가는 3090 에서).
  리뷰의 설계 의견(1칸 기본, 지원 확인된 곳만 분할, 유턴 병합, 최근접 거리≠제출 L2)은 현재 lat1/adaptive 쌍 설계와 일치.
- ★ [측정 2026-09-17 13:20] goal 모델 plan_reg 가 대조군보다 높은 원인 — **goal 분류 CE 가 planner feature 를 점령**.
  [사용자] "최소한 clean 보다는 좋아야 되는 거 아닌가, 로스가 잘 안 떨어지네".
  `tools/diag_goal_grad_conflict.py` (adaptive epoch_6, 실제 train batch 8개, image 경로 freeze, optimizer step 없음) — 손실별 ego_feats gradient:
  | 손실 | |grad| (plan_reg 대비) | plan_reg 와 cos |
  |---|---|---|
  | loss_goal_cls | **5.2× ~ 8.9×** (두 번 측정, 샘플 무작위성) | −0.03 ~ 0.01 (직교) |
  | loss_goal_select | 0.8× | 0.8 ~ 0.98 |
  | loss_goal_off | 0.4~0.5× | 0.07 ~ 0.10 |
  | loss_goal_follow | ≈0 | — |
  해석 [Claude 분석]: 공유층 gradient 의 대부분이 planning 과 무관한 방향(분류)이라 Adam 정규화 아래 planning 신호가 몇 분의 1 로 줄어든다 → 구 goalpred·lat1·adaptive 모두 같은 +10~14% 격차. 앵커 설계와 무관.
- 수정 옵션 구현 (기본 1.0 = 현재 run 과 동일, 실행 중 학습 영향 없음): `VAD_head(goal_head_grad_scale)` — goal_cls/off 헤드 입력의 **backward 만** 스케일(forward 값 동일).
  [측정] 같은 체크포인트에서 0.1 로: goal_cls 0.57×, goal_off 0.03× (plan_reg 대비). 헤드 자체 학습은 그대로.
  재학습은 사용자 결정 대기.

### ★★ [측정 2026-09-15 22:10] dropout 끈 fine-tune epoch 1 = **0.3772** (테스트 조건) — 현 최고 compliant
`stage2_nodistill_nodrop_ft/epoch_1.pth` (nodistill ep12 + disable_dropout, lr 1e-5, 1 epoch). 3프레임, `--test-commands`.
L2@avg **0.3772** (nodistill ep12 0.5018 → −25%), LANE_KEEP 0.3599, STOP 0.2689.
T_infer 293ms 는 같은 머신에서 학습이 돌던 중이라 **무효** — 단독 재측정 필요.
[사용자] epoch 1 에서 중단 지시 → epoch 2 는 학습하지 않음. v1 0.4218 을 넘은 첫 compliant 모델.

### [확정 2026-09-16] 추론 시 TP 선택 옵션 구현 (재학습 불필요) + 규정 해석 정정
[사용자] "나는 추론 때 cmd 처럼 선택해서 쓰는 걸 말한 건데".
- **정정**: 내가 Q&A 2절 A1(신경망 미입력 정보 후처리 금지)을 들어 이 방향을 배제했던 건 과했다.
  `Q&A.md` 4절이 target point 에 대해 더 구체적이고 더 나중이다 — 08-25 "여러 출력 중 선택에만 활용되는 경우 허용",
  08-26 "기준점 기반으로 궤적을 새로 생성/보정이 아니라 **선택에만 쓰이면 허용**", 08-27 "베이스라인처럼 출력 중 선택은 가능".
  같은 이유로 예전에 제거한 TP 기반 STOP 규칙도 실은 선택 용도였다.
- 구현(학습 불필요, 기본 꺼짐):
  `VAD_head(goal_expose_candidates=True)` → 추론에서 앵커 12개 각각의 예측 목표로 궤적을 만들어
  `goal_cand_trajs [M,K,T,2]`, `goal_cand_points [M,K,2]` 출력 (후보 생성에 TP 미사용).
  `VAD.simple_test_pts` 가 결과에 전달. eval config `..._fast_eval_clean_goalpred_cand.py`.
  `eval_holdout_l2_and_tinfer.py --select-goal-by-tp`: 주어진 TP 에 가장 가까운 후보를 채점.
  감사는 이 옵션의 존재를 `[!!]` 로 경고만 한다(기본 꺼짐, 제출 수치가 어느 쪽인지 명확히 하려고).
- [측정] 실제 val 스트림 1창에서 동작 확인: cand_trajs (7,12,6,2), cand_points (7,12,2),
  TP(59.8, 2.1) 에 가장 가까운 후보 bin 6 (거리 2.07m), 최대 59.8m. (도너 가중치라 초기 상태 = 후보 전부 동일)
- 오늘 오후 학습 종료 후 같은 체크포인트로 **TP 미사용 / TP 선택** 두 수치를 모두 측정한다.

### [사용자 2026-09-15 23:10] 방향: 증류(teacher→student) 말고 직접 학습
"지금 우리가 하는 건 student 가 아니라 그냥 하는 거잖아. 증류로는 한계가 있을 것 같아서".
- 현재 두 run(`stage2_clean_goalpred`, `stage2_clean_nodistill`) 모두 **teacher 없음**: `feature_distill_teacher_cfg=None`,
  scene/status distill weight 0, 학습 로그에 distill 손실 0건 [측정].
- config 상속 체인 이름에 `kd_nolcf_split_distill8...` 가 남아 있지만 증류는 꺼져 있다(이름만 계보 흔적).
- 상태 슬롯(`ego_status_est_net`)은 teacher 가 아니라 GT ego_lcf 라벨(`ego_status_decode`, `aux_bev_motion`)로만 학습된다.
- `VADLAW_etri_tiny_clean_student.py`/`student-clean` 역할은 준비만 된 상태로 두고 우선순위에서 제외.

### [확정 2026-09-15 22:45 KST] yaw wrap 수정 후 두 run 재시작
[사용자] "수정 후 둘 다 끊고 재시작".
- 수정: `law_etri_dataset.py` union2one, `VAD_LAW.py` forward_test — `(d + 180) % 360 - 180` (stage1 과 동일).
- [측정] `tools/check_can_bus_yaw.py`: 경계 통과 실샘플 학습 큐 5개·추론 스트림 5개에서 yaw 변화 전부 |d|≤180, GT yaw_rate×0.5s 와 일치(예 −1.61, +1.51). 통과.
- 감사 3c 에 wrap 소스 검사 추가(빠지면 FAIL). A5000 동기화 md5 일치.
- 재시작 iter 100 [측정], 수정 전 run 과 비교 (같은 설정, 같은 iter):
  | | loss_aux_bev_motion | loss_aux_bev_future_motion | loss_plan_reg |
  |---|---|---|---|
  | 3090 goalpred 수정 전 → 후 | 0.0383 → **0.0261** | 0.0216 → **0.0166** | 0.0156 → 0.0143 |
  | A5000 nodistill 수정 전 → 후 | 0.0380 → **0.0259** | 0.0213 → **0.0158** | 0.0152 → 0.0146 |
  자기운동 추정 손실이 두 run 모두 약 32% 낮게 출발 — yaw 오염이 속도·회전 추정에 실제로 영향을 줬다는 첫 신호(iter 100, 한 시점).
- Traceback 0, rc=0 둘 다. goal 실측 검사 재통과.

### ★ [측정 2026-09-15 23:40] 최종 점검 — stage2 에서 can_bus yaw 변화량이 0/360 경계에서 ±359° 로 들어간다
[사용자] "버그나 dropout 같은 성능에 영향 줄 이상한 것들 확인해, 중요한 거야".
점검 결과 (현재 학습 중인 두 config 기준):
- **정상 확인**: nn.Dropout 64개 전부 p=0 · BatchNorm 53개 전부 backbone 안, `norm_eval=True` 로 학습 중에도 eval 모드(학습/추론 동일) ·
  DropPath/GroupNorm 등 없음 · `self.training` 분기는 전부 손실 전용이거나(aux, decode, long_horizon, echo cycle, goal 라벨) 값이 0/꺼짐
  (est dropout, prev_bev_dropout, PRISM, privileged) · fp16 loss_scale 512, grad_clip 35, EMA 0.0002 동일 ·
  런타임 can_bus yaw 는 ego pose 에서 채워지고 0.5s 변화가 GT yaw_rate 와 **상관 1.000**.
- 남은 학습 전용 요소: GridMask(prob 0.7), PhotoMetricDistortion — 표준 증강. GridMask 는 속도 편향에 영향 없음 실측.
- **버그**: yaw 는 0~360° 인데 **stage2(LAW) 경로는 프레임 간 차이를 wrap 하지 않는다** —
  학습 `law_etri_dataset.py:83` `can_bus[-1] -= previous_angle`, 추론 `VAD_LAW.py:987` 동일.
  stage1(VAD) 경로는 둘 다 `(d + 180) % 360 - 180` 로 wrap (`nuscenes_vad_dataset.py:1202`, `VAD.py:394`).
  [측정] 0/360 경계를 넘는 0.5s 쌍: **train 3.57%, val 3.89%** — 그 프레임은 ±1° 대신 ±358~360° 가 들어간다.
  영향: prev_bev 회전은 359°≡−1° 라 무해, shift 는 절대각(can_bus[-2]) 사용이라 무해,
  **`can_bus_mlp` 입력(18차원 그대로)에 정상 범위(±9°) 대신 359 가 들어가 BEV query 가 오염**된다.
  stage2 안에서는 학습·추론 일관이라 dropout 같은 train/eval 편향은 아니지만, 약 4% 프레임(3프레임 창 기준 약 7~8%)의 BEV 가 손상되고
  stage1 도너가 배운 분포와도 다르다. 오늘까지 모든 stage2 수치(0.3772, teacher 0.2182, v1 0.4218)에 포함. **L2 영향 크기 미측정.**
- 수정안 [Claude 제안]: 두 곳에 stage1 과 같은 wrap 적용. 단 학습 중인 두 run 은 wrap 없이 학습 중이라, 코드를 지금 고치면
  그 체크포인트 평가가 학습과 달라진다 → **사용자 결정 대기** (재시작 vs 유지). 두 run 은 동일하게 영향받아 비교 자체는 공정.

### [확정 2026-09-15 22:30 UTC13:30] 비교 학습 시작 — 3090 실험군 / A5000 대조군
[사용자] "딱 저대로 올리렴".
| 서버 | work_dir | config | 차이 |
|---|---|---|---|
| 3090 | `stage2_clean_goalpred` | `VADLAW_etri_tiny_clean_goalpred.py` | goal_pred 7개 설정 |
| A5000 | `stage2_clean_nodistill` | `VADLAW_etri_tiny_clean_nodistill.py` | — |
config diff: goal_* 7개만 다름, load_from·optimizer·lr·epoch·data.train 동일.
[측정] iter 100: 둘 다 Traceback 0. plan_reg 0.0156 / 0.0152, aux_long 0.3688 / 0.3697 (거의 동일 → 공통부 동일 확인),
실험군 goal_cls 1.2156 (초기 1.2425 에서 하강), goal_off 0.0135, goal_follow 0.0035. 시작 전 `check_goal_pred_live` 통과.
- 버그: `verify_start` 가 `set -euo pipefail` 아래에서 `size mismatch` grep 0건(정상)을 실패로 받아 rc=1 로 종료 → `|| true` 로 수정, errexit 상태에서 rc=0 확인. 학습엔 영향 없음.
- 평가 계획: 둘 다 3090 에서 `--test-commands` 3프레임. 실험군은 `outs['goal_pred']` vs TP 거리 오차도 측정. GPU 기종 차이로 0.005 이내 차이는 효과로 단정하지 않음.

### [확정·구현 2026-09-15 23:00] TP 활용 구조를 "예측 5초 목표 추종 planning" 으로 교체 (학습 안 함)
[사용자] "예측 가능성은 최선의 방식을 구현하고 하는 거야. 약점들 구현한 다음 평가. 먼저 최선의 방식을 설명하고 구현해봐".
5×5 격자(학습된 적 없음)의 약점 ① 칸이 거침 ② 확률 혼합이 가속/감속 궤적을 섞음 ③ 칸별 출력 복사로 데이터 분할 → 교체.

`VAD_head(goal_pred=True)` — config `VADLAW_etri_tiny_clean_goalpred.py` (clean_nodistill + goal_pred 만 다름), eval `..._fast_eval_clean_goalpred.py`.
- 목표 = 전방 거리 구간 분류(−5~115m, 10m × 12) + 구간 안 연속 오프셋 회귀(x 는 구간폭 단위, y 는 25m 단위). 좌우는 구간 없이 회귀.
  모두 **커맨드 모드별** ([B,7,12], [B,7,12,2]).
- 모드마다 argmax 구간 + 오프셋 → 목표점 **하나** → `goal_embed` 로 ego_feats 에 더해 **공유 디코더**로 궤적 **하나**. 혼합 없음.
- 디코더는 10스텝(5s) 출력, `ego_fut_preds` 는 앞 6스텝. donor 로드 시 앞 6스텝 복사 + 뒤 4스텝 마지막 스텝 복사(등속).
- 학습 손실: `loss_goal_cls`(CE, GT 커맨드 모드, GT TP 구간) · `loss_goal_off`(smooth L1, 그 구간 오프셋) ·
  planning loss 는 **예측 목표로 생성한 궤적**(제출 출력) · `loss_goal_follow`: 네트워크 자신의 분포에서 뽑은 다른 목표(detach)로
  생성한 궤적의 5초 끝점이 그 목표에 도달 → 디코더가 목표 입력을 따르게 함. **GT TP 는 디코더 입력으로 한 번도 안 들어감**(라벨만).
- 가중: cls 0.5, off 0.5, follow 0.1. PRISM/privileged/refine 과 동시 사용은 생성자에서 거부.
- [측정] `tools/check_goal_pred_live.py` 실제 batch 8개: 초기 CE 1.2425 = 0.5·ln12 정확, 라벨 왕복 오차 ≤6e-8,
  구간 0/1/3/6/7 다양, 손실 스케일 off 0.00~0.03 · follow 0.001~0.02 · plan_reg 0.005~0.02, 초기 3초 출력 donor 와 차 2.4e-7,
  eval 모드 forward 에서 라벨 생성 호출 0회. 감사 통과(디코더 연장 실제 load 확인, TP 게이트). 추론 스트림 경로 실행 확인.
- 평가 때 `outs['goal_pred']`(모드별 예측 목표, m)로 목표 예측 정확도를 따로 잴 수 있다.
- **학습은 사용자 지시 대기.**

### [사용자 2026-09-15 22:25] 학습은 사용자가 지시할 때만 — 두 학습 중단
"3090 그냥 5×5 로 하는 거야? 학습 내가 돌리라 할 때까지 돌리지 말아봐. TP 구현 최선의 방법으로 한 거 맞아?"
→ `stage2_clean_goalgrid`(3090), `stage2_clean_nodistill`(A5000) 둘 다 시작 몇 분 만에 중단. 두 GPU 모두 비어 있음.
설계·코드·감사·평가/진단은 계속하되 train.py 실행은 사용자 지시 후에만.

### [확정 2026-09-15 22:15] TP 목표 격자 학습 시작 (3090) + clean 대조군 (A5000)
[사용자] "TP 구현한 거 돌리는 게 최종 목적", "A5000 student 는 의미 없어" → A5000 구 student 중단.
- **3090 `stage2_clean_goalgrid`** — `VADLAW_etri_tiny_clean_goalgrid.py`: clean_nodistill + 5×5 (−5~110 × ±25).
  리뷰 반영 수정 두 가지:
  (1) `goal_cls_head` 가 **커맨드 모드별** 분포 7×25 를 낸다 (CE 는 GT 커맨드 모드만).
  (2) `ego_fut_preds` 는 학습·추론 모두 **확률 혼합 궤적** — planning loss 가 제출 출력을 직접 학습하고 분류기에 gradient 가 간다.
      GT 칸 궤적은 별도 `loss_goal_cell_traj`(가중 0.1 — 초기 실측 0.14 vs plan_reg 0.014 라 1.0 이면 10배 과대).
  `ego_long_fut_trajs[:6]` == `gt_ego_fut_trajs` 확인(1806 샘플 차 0). `tools/check_goal_grid_live.py` 통과:
  초기 CE 1.6094 = 0.5·ln25, 초기 출력 donor 와 차 1.2e-7, 칸 다양, 추론 게이트. 시작 후 iter100: goal_cls 1.553, cell_traj 0.012, plan_reg 0.015, Traceback 0.
- **A5000 `stage2_clean_nodistill`** — 격자만 뺀 동일 설정. 둘의 차이가 격자 효과.
- PRISM 과 goal grid 동시 사용은 생성자에서 거부.

### [확정 2026-09-15 22:00] 학습/추론 불일치 설정 전면 제거 + 과거 파일 74개 삭제
[사용자] "dropout 같은 예전에 썼다가 이제는 안 쓰는 이상한 것들 다 삭제해, 결과 망치잖아. 너가 판단해서".
- 현재 계보에서 켜져 있던 불일치 설정 4개: `nn.Dropout`(측정: +5% 속도 편향), `prev_bev_dropout=0.5`,
  `ego_status_est_dropout=0.3`, `prism_latent_supervision=True`(학습은 GT 미래 posterior, 추론은 prior 평균).
  뒤의 셋은 **개별 효과 미측정** — 같은 원칙으로 제거.
- 새 config: `VADLAW_etri_tiny_clean_student.py`, `VADLAW_etri_tiny_clean_nodistill.py`, eval `VADLAW_etri_tiny_fast_eval_clean.py`.
  `VAD` 에도 `disable_dropout` 추가(stage1 재학습 대비).
- `audit_pipeline.py` 3c: 위 설정 + refine/shortcut/privileged/비고정 간격이 하나라도 있으면 FAIL.
  구 nodistill config 로 음성 대조 → 4개 FAIL 확인. clean 두 config 통과.
- 삭제: config 41개(현재 계보·실행 중 학습·v1 폴백의 상속 체인 밖), scripts 11개(run_A/B, chain_*, queue, 일회성 평가),
  tools 22개(구 평가·ablation·run_full_pipeline·일회성 probe). 남은 config 40개 전부 로드 확인. A5000 사본도 동일 삭제·동기화(md5 일치).
- 모델 코드의 옵션 분기(PRISM 등)는 삭제하지 않음: 실행 중 학습·기존 체크포인트 평가가 그 코드를 쓴다. 대신 감사가 켜는 것을 막는다.
- 구 goal-grid config(5×5, 불일치 설정 상속)는 설계 논의 중이라 유지, 재설계 시 clean 기반으로 새로 만든다.

### ★ [측정 2026-09-15 21:10] 목표 격자(사용자 그림 5×5) 의 가치 — 격자로 자르면 TP 정보가 거의 사라진다
[사용자] 그림: TP(5s) 를 −5~110m × ±25m 5×5 격자로, planner 7×6×2 → 7×(5×5)×6×2, 범위 밖 약 0.5%.
도구 `tools/goal_grid_value.py`, `tools/goal_tp_noise_value.py`. **선형 대리 모델**(현재 ego 상태 6개 + 명령별/칸별 ridge,
train 90300 → val 22500, 공식 L2). 신경망이 아니라 "그 정보가 있으면 얼마나 줄 수 있나"의 상한 추정.

| 입력 | val L2 | 이득 |
|---|---|---|
| 명령 + 상태 (칸 없음) | 0.2234 | — |
| + **TP 연속값 (완벽)** | **0.1191** | **−47%** |
| + TP x(전방)만 / y(좌우)만 | 0.1533 / 0.1964 | +0.070 / +0.027 |
| + 5×5 칸 (완벽 분류) | 0.2207 | **+0.0027** |
| + 7×3 칸 (완벽 분류) | 0.2143 | +0.0090 |
| + 20×1 칸 (완벽 분류) | 0.1803 | +0.0430 |
| + 5×5 칸을 등가속 외삽으로 선택 (정보 없음, 오분류 21%) | 0.2543 | **−0.031 (악화)** |
| + 등가속 외삽 TP 연속값 | 0.2234 | 0 |

TP 예측 오차(축별 σ, 학습·평가 동일 잡음)에 따른 이득: 0m +0.104 / 1m +0.079 / 2m +0.049 / 3m +0.030 / **4m +0.019** / 6m +0.010 / 8m +0.005.
등가속 외삽 대비 TP 잔차 std: x 4.18m, y 2.81m.

- 범위 −5~110 × ±25 는 적절: val 범위 밖 0.06%, train 0.51%(그림의 0.5% 와 일치). 좁히면 13~40% 가 가장자리로 뭉개짐.
- **5×5 는 완벽하게 맞혀도 1% 이득, 틀리면 악화.** 칸 폭 23m 가 σ≈6.6m 잡음과 같아 정보가 사라진다. 가치는 전방 거리의 연속값에 있다.
- [Claude 제안] 격자 분류 대신 **5s 목표점 연속 회귀 → 예측값으로 planner 조건화**(예측값만 입력, GT 는 라벨: 규정상 vision 추론값).
  단 이득은 예측 오차가 등가속 외삽(4.18m)보다 확실히 작을 때만 생긴다 → **먼저 기존 `aux_long_horizon` head 의 val 5s 오차를 측정**해서
  σ<~3m 이면 구현, 아니면 이 방향 보류. 학습 때도 GT 가 아닌 예측값을 넣어야 함(오늘 dropout 건과 같은 train/eval 불일치 방지).

### ★★ [측정 2026-09-15 18:40] nodistill 이 나쁜 원인 = **dropout 의 train/eval 특징 분포 차이**
도구: `tools/diag_student_speed.py`(스트리밍 추론 경로), `tools/diag_train_path_speed.py`(학습 forward 경로).
nodistill epoch_12, 200 창. 속도는 m/s, bias = 예측−GT.

| 조건 | 속도추정 RMSE / bias | 계획 첫0.5s bias | L2@3s |
|---|---|---|---|
| val, 평소 추론 | 0.645 / **+0.52** (비율 1.05, 속도 비례) | +0.29 | 0.732 |
| val, fp32 / bev-only 끔 / raw(EMA아님) / cache 이미지(정규화 동일) | 전부 +0.52~0.53 | +0.28~0.30 | 0.72~0.74 |
| train 데이터, 평소 추론 | 0.621 / +0.51 | +0.26 | 0.510 |
| train 데이터, **학습 forward 경로 eval 모드** | 0.585 / +0.49 | — | — |
| 학습 forward 경로 **train 모드** (GridMask 켬/끔) | 0.146 / +0.007, 0.139 / +0.003 | — | — |
| 학습 경로 eval + **nn.Dropout 만 train** | 0.139 / −0.004 | — | — |
| 학습 경로 eval + BN 만 train | 2.880 / −1.49 | — | — |
| 학습 경로 eval + **BEV 인코더 dropout 12개만** | 0.146 / −0.006 | — | — |
| val, 추론에 인코더 dropout 만 켬 | 0.242 / +0.05 | +0.29 | 0.742 |
| val, 추론에 **dropout 전부** 켬 | 0.238 / +0.05 | **+0.07** | **0.553 (−24%)** |

- 결론: 속도 추정(인코더 dropout)과 플래너의 속도(디코더 dropout) 둘 다 dropout 이 켜진 특징에 맞춰져,
  dropout 이 꺼지는 추론에서 약 5% 빠르게 읽는다. 프레임 간격(idx−5 = 정확히 5프레임/500ms), 이미지 경로,
  fp16, EMA, bev-only 는 **배제**(측정). 앞서 "이미지 파이프라인 차이"라고 한 판단은 정규화 안 된 cached_eval
  config 로 잰 오류였고, 정규화를 맞추자 동일했다.
- 다른 평가: nodistill val 명령 그대로 **0.4866** (STOP 을 GT 로 주면 STOP 0.0126; 테스트 조건 선택 비용 0.015).
  teacher `--zero-ego-lcf` **11.74** → teacher 는 ego_lcf 를 실제로 쓴다(정상).
- 추론 때 dropout 을 켜는 건 출력이 무작위라 해법이 아님. **[확정] dropout 전부 끄고 이어 학습**:
  `VADLAW(disable_dropout=True)`(64개 nn.Dropout p→0), `VADLAW_etri_tiny_nodistill_nodrop_ft.py`
  (nodistill ep12 에서 lr 1e-5, 2 epoch). 3090 18:40 시작, epoch 1 ≈ 20:40.
- **A5000 증류 student 도 같은 조건(dropout 켬)으로 학습 중** — fine-tune 결과로 수정이 확인되면 disable_dropout 으로 재시작 검토.
- 이 현상은 이전 모든 계보(v1 0.4218 포함)에도 있었을 가능성이 높다 (미측정).

### ★ [측정 2026-09-15 17:46] stage2 첫 평가 — teacher 0.2182, **nodistill 0.5018 (나쁨)**
3프레임(0,−5,−10), fp16, bev-only-history, `--test-commands`(STOP은 모델 속도 추정으로 선택), val 4045 샘플.
T_infer는 두 평가 동시 실행 중 측정 → 단독보다 부풀었을 수 있음.

| 모델 | L2@1s | L2@2s | L2@3s | **avg** | LANE_KEEP | STOP | T_median | 페널티 |
|---|---|---|---|---|---|---|---|---|
| teacher (ego_lcf 입력, 제출 불가) | 0.0688 | 0.1922 | 0.3936 | **0.2182** | 0.2091 | 0.1263 | 157.8ms | ×1.289 |
| nodistill (제출 가능) | 0.2738 | 0.4873 | 0.7443 | **0.5018** | 0.5049 | 0.2411 | 144.8ms | ×1.224 |

- teacher 0.2182 < 이전 A teacher v1 0.2328 → teacher는 정상 범위 ("0.20~0.23 진행" 분기).
- **nodistill 0.5018은 stage1 nolcf 도너 refine-off 0.4807보다도, A student v1 0.4218보다도, 베이스라인 0.4885보다도 나쁘다.**
  학습 `loss_plan_reg` 0.0091(teacher 0.0047)로 train 쪽 차이는 2배인데 eval은 2.3배, **L2@1s가 4배(0.274 vs 0.069)**
  → 1초 오차는 거의 속도 추정 오차. student 상태 슬롯 경로의 train/eval 불일치 또는 속도 추정 실패 의심. **원인 미확인.**
- 조치: goal-grid 대기열 **보류**(같은 student 경로를 물려받으므로). val 명령 평가·teacher zero-lcf 대조군 진행 중.

### [확정 2026-09-15 17:43] goal grid(TP 목표 칸) 학습 준비 완료 → 3090 평가 끝나면 자동 시작
[사용자] "TP를 커멘드처럼 GT로, 위배 안 되는 선에서 선택만". `VADLAW_etri_tiny_nodistill_goalgrid.py`
(nodistill 대비 `goal_grid_size=(7,3)` 하나만 다름), `scripts/queue_goalgrid_after_evals.sh`.
- 목표 격자 = **라벨 공간** 전방 −5~110m(7칸, 16.4m) × 좌우 ±25m(3칸). **BEV `pc_range`는 그대로 60×30m**.
- 학습: TP가 감독할 칸 + `goal_cls_head` 라벨. 추론: TP 안 읽음(`self.training` 게이트), 분류기 soft 선택.
- [측정] donor 전이: 마지막 층 84→1764행, 실제 load 후 21칸 전부 donor와 **비트 동일**. step0 출력 차 2.4e-7.
  초기 `loss_goal_cls` 기대 0.5·ln21 = 1.522.
- [측정] train TP 칸 분포(행=전방 bin, 열=좌·중·우): 중앙열 84.7%. 전방 거리는 **이봉** — 60~65m 봉우리와
  100~105m 봉우리(후자 샘플 평균 속도 20.4 m/s, 5s·v=102m = 고속도로). 마지막 bin 11390개는 이 때문이지 오류 아님.
- 감사 donor 검사가 goal grid 복제를 실제 load로 확인하도록 수정 (이전엔 shape만 봐서 FAIL로 오판).

### 해소된 항목 (기록용)
- ~~`loss_plan_reg=0.0` 논쟁~~ → **[확정]** Qwen teacher hold-out 0.3511 측정으로 (가) 자동 탈락.
  (나) GT만으로 결정, 현재 stage1 둘 다 그 구성.
- ~~A student forward 미검증~~ → v1이 완주하고 0.4218을 냈다.
- ~~`aux_bev_motion_temporal` loss-only 미측정~~ → 현재 stage1 둘 다 켜고 돌고 있고,
  `loss_aux_bev_motion`이 정상 감소 중(0.4925 → 0.4720).

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
- **student가 도달 가능한 L2** — v1 예측 0.44~0.46이었고 **실측 0.4218로 예측보다 좋았다.**
  재구축 후 목표는 **0.33~0.38**. 근거: v1은 운동학 오라클 기준 "속도는 알고 가속도는 모르는"
  위치(0.5965와 0.2708 사이)에 있었고, 이번에 가속도(idx 2,3 + 3프레임 2차 차분)와
  미래 속도 프로파일을 추가했다. 다만 **오라클 격차가 그대로 전달된다는 보장은 없다** —
  student는 그 값을 vision으로 *추정*할 뿐이고, 추정 오차가 곧 L2 오차다.
- **0.2 미만은 현 구조로 불가능** [측정 근거 있음]. 증류 천장이 teacher(0.2328)이고,
  1등 0.14는 완벽한 운동학 오라클(0.2708)마저 48% 앞선다 = 미래 기동 자체를 장면에서
  예측한다는 뜻. **단 1등 0.14가 같은 split·같은 metric인지는 여전히 미확인.**

### 열린 쟁점
- **can_bus 경로의 규정 해석** — **[확정 2026-09-12] 유지한다.**
  `can_bus[7:16]`(가속도/각속도/속도)이 `can_bus_mlp`를 통해 BEV 쿼리에 임베딩된다.
  `d(loss_plan_reg)/d(ego_lcf_feat)==0` 컴플라이언스 증명이 이 경로는 다루지 않아
  한때 제거를 검토했으나, 결정적 근거는 **배포된 베이스라인 자체가 `can_bus`는 켜고
  `ego_lcf_feat_idx=None`으로 막아뒀다는 것**이다. 주최측이 둘을 의도적으로 구분했다.
  구조적으로도 다르다 — can_bus는 이전 BEV를 자차 이동만큼 회전·평행이동 보정하는
  **기하 정합용**이고, 검출·맵·모션 head가 전부 공유한다. 규정의 "간접 활용(공통 feature
  개선)"에 해당. 제거 시 10.4m로 붕괴하므로 베이스라인과의 비교 자체가 불가능해진다.
- **1등 0.14가 같은 split·같은 metric인가** — 3회 이상 제기됐으나 **여전히 미확인.**
  우리 결론("0.2 미만 불가능")이 이 가정 위에 서 있으므로, 확인되면 목표 재설정이 필요하다.

---

## 10. Codex 의견

### [Codex 의견 2026-09-15] BEV 크기 질문 — 두 grid를 구분

- 코드/실행 config 확인: `stage1_best_nolcf`의 실제 BEV는 `100×100`,
  `aux_bev_motion_grid=8`은 별도로 BEV를 `8×8`로 pooling하는 descriptor 설정이다.
- `VAD_base_stage_2.py`에는 `200×200` 설정이 있다. 모델의 고정 상한은 발견하지
  못했으나, **현재 학습 구성/GPU의 최대 크기는 실측하지 않았다.**
- BEV 해상도를 변경하면 learned BEV/position embedding 크기와 회전 중심 등도
  맞춰야 하므로 기존 checkpoint를 그대로 호환된다고 가정하면 안 된다.
- `100→200`으로 양축을 늘리면 BEV 셀 수는 4배다. 전체 VRAM/시간이 정확히
  4배라는 의미도, 성능이 개선된다는 측정도 아니다. 설정 변경/학습 실행은 하지 않음.

### [Codex 의견 2026-09-15] 사용자 그림의 5×5는 목표 구역별 궤적 후보

- [사용자] 5초 TP, 전방 −5~110m/좌우 ±25m, `7×6×2 → 7×(5×5)×6×2` 그림을 제시.
- 코드 확인 시 `VADLAW_etri_tiny_nodistill_goalgrid.py`는 이미 `goal_grid_size=(5,5)`였다.
  앞선 공유 기록의 `(7,3)`은 현재 파일과 다르다. Codex가 config를 변경한 것은 아님.
- 이는 BEV feature 해상도/관측 범위를 바꾸지 않는 **목표 라벨 공간**이다.
  5초 TP 구역을 감독하되 출력 6스텝(3초)의 끝을 TP에 맞추는 구조는 아니다.
- 구현은 학습 때 GT 구역의 궤적 감독 + 구역 분류 CE, 추론 때 TP 없이 예측 확률을 사용.
  현재 `goal_grid_select='soft'`는 후보 하나 선택이 아니라 25개 궤적의 가중평균이다.
  후보가 갈라지면 평균 궤적이 각 후보와 달라질 수 있으므로 hard 선택과의 차이를
  구분해야 한다. 어느 방식이 더 좋은지는 미측정. 모델 수정/학습 실행은 하지 않음.
