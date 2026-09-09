# 실행 스크립트

A안/B안 × teacher/student 네 가지를 헷갈리지 않게 분리해둔 것.
모든 스크립트는 **로컬(3090, vcl-server-5)에서 실행**하면 된다. B안 스크립트는
내부에서 알아서 A5000으로 ssh한다.

## 전체 구도

|  | **A안 — 분리 증류** | **B안 — 융합 증류** |
|---|---|---|
| 아이디어 | 512(장면)/64(자기운동)를 **따로** 증류, student는 64칸을 디코더 입력으로 씀 | 디코더 첫 Linear 직후 **융합된 은닉값 512** 하나만 증류 |
| teacher의 ego_lcf | `MLP(8→64)` 학습 임베딩 | raw 8칸 그대로 |
| student의 추가 위험 | 추정 채널이 추론 경로에 박힘 → modality dropout으로 완화 | 없음 (디코더 입력은 순수 vision 512) |
| 담당 머신 | **3090** (로컬) | **A5000** (`10.10.52.49`) |

## 실행 순서

```
1단계 (병렬, 각 ~22h)
  ./scripts/run_A_teacher.sh          # 3090
  ./scripts/run_B_teacher.sh          # A5000

2단계 teacher 평가 (각 ~30분)
  ./scripts/eval_l2.sh A teacher
  ./scripts/eval_l2.sh B teacher
  → 판정: 0.2166(금지 전 ego_lcf-ON 기록) 수준이 재현되는가?
     0.20 이하 정상 / 0.25~0.30 진행 가치 있음 / 0.35 이상이면 원인부터 조사

3단계 (병렬, 각 ~22h)
  ./scripts/run_A_student.sh          # 3090
  ./scripts/run_B_student.sh          # A5000

4단계 student 평가
  ./scripts/eval_l2.sh A student
  ./scripts/eval_l2.sh B student
  → 비교 기준: compliant 베이스라인 0.4885m
  → 평균만 보지 말 것. LANE_KEEP이 전체 오차의 85%다
```

## 파일 대응표

| 스크립트 | config | work_dir | 초기 체크포인트 |
|---|---|---|---|
| `run_A_teacher.sh` | `VADLAW_etri_tiny_kd_lcfemb_teacher.py` | `stage2_kd_lcfemb_teacher` | `stage2_init_merged_lcfemb64.pth` |
| `run_B_teacher.sh` | `VADLAW_etri_tiny_kd_lcfon_diag.py` | `stage2_kd_lcfon_diag` | `stage2_init_merged_lcfon8.pth` |
| `run_A_student.sh` | `VADLAW_etri_tiny_kd_nolcf_split_distill.py` | `stage2_kd_nolcf_split_distill` | `stage2_init_merged_lcfemb64.pth` (teacher와 공용) |
| `run_B_student.sh` | `VADLAW_etri_tiny_kd_nolcf_feature_distill.py` | `stage2_kd_nolcf_feature_distill` | `stage2_init_merged.pth` |

student 스크립트는 해당 teacher의 `epoch_12.pth`가 없으면 **시작을 거부한다.**
(teacher 없이 돌면 조용히 엉뚱한 결과가 나오므로)

## 로그 보기

```bash
./scripts/tail_log.sh A teacher
./scripts/tail_log.sh B student
```
