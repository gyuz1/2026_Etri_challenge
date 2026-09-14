# AGENTS.md

ETRI 2026 자율주행 챌린지 · LAW_split 트랙.

## 공유 기록 규칙

이 프로젝트는 여러 에이전트(Claude, Codex)가 함께 작업한다. `SHARED_CONTEXT.md`가
공통 기억이다.

**사용자 요청에 답하거나 작업을 시작하기 전에 `SHARED_CONTEXT.md`를 읽는다.**
목표·확정 사항·측정된 수치·이미 배제한 대안·버그 이력이 거기 있다.

**기록에 적힌 제안을 확정된 요구사항으로 착각하지 않는다.** 표기를 구분해서 읽는다.
- `[사용자]` / `[확정]` / `[측정]` — 근거로 삼아도 되는 것
- `[Claude 제안]` / `[Codex 의견]` — **아직 제안일 뿐이다.** 다른 에이전트가 그렇게
  판단했다는 사실일 뿐, 사용자가 승인했다는 뜻이 아니다

**의견을 요청받으면 기록의 근거를 검토하고 독립적으로 판단한다.** 이전 에이전트의 결론을
그대로 반복하지 말고, 그 논거가 실제 측정에 기반한 것인지 추론에 기반한 것인지 확인한다.
동의하지 않으면 근거를 들어 반대 의견을 낸다. 이 프로젝트에서는 이미 여러 번,
"그럴듯한데 측정되지 않은 가정"이 틀린 것으로 드러났다.

**중요한 논의와 작업 결과를 `SHARED_CONTEXT.md`에 기록한다.** 표기 규칙은 그 파일 상단에
있다. 추측과 측정값을 섞지 않는다.

**Claude가 남긴 의견은 보존한다.** 동의하지 않더라도 지우거나 고쳐 쓰지 않는다.
반론은 `[Codex 의견]`으로 나란히 적는다.

## 이 프로젝트에서 특히 조심할 것

- **조용히 틀리는 버그가 주된 실패 유형이다.** 크래시 없이 잘못된 수치가 나온 사례가
  여러 번 있고, 한 번은 22시간 학습을 날렸다. 긴 학습 전에 config diff·입력값 실측·
  eval config 일치를 확인한다.
- **규정 준수.** ego 상태(`ego_lcf`)와 목표점(`target_point`)은 planner 생성 단계에
  직접 넣을 수 없다. 간접 활용(공통 feature 개선)은 허용된다.
- **GPU는 사용자와 공유한다.** 지시 없이 남의 작업을 중단하지 않는다.
- **학습 로그와 체크포인트 로딩 출력에 요약·필터를 걸지 않는다.** 진단의 결정적 단서가
  딱 한 줄인 경우가 반복해서 있었다. 측정 근거는 `SHARED_CONTEXT.md` 인프라 메모의 RTK 항목.
- 학습 산출물은 컨테이너 안에서 root 소유라 호스트에서 안 지워진다. `docker exec`로 지운다.

---

## 지금 무엇이 도는가 (2026-09-12)

| 서버 | 작업 | work_dir |
|---|---|---|
| 3090 (`gyuz_split_3090`) | stage1 student 도너 | `stage1_best_nolcf` |
| A5000 (`gyuz_split2`) | stage1 teacher 도너 | `stage1_best_lcfon` |

**4090(`gtk@211.42.239.45`)은 건드리지 않는다.**

전체 설계·근거·일정은 **`PIPELINE.md`**. 측정 수치와 버그 이력은 `SHARED_CONTEXT.md`.

## 긴 학습을 시작하기 전에 — 예외 없이

```bash
python tools/audit_pipeline.py <train_config> [--eval-config <cfg>]
python tools/check_accel_block_live.py <train_config>
```

`scripts/run_stage2_best.sh`는 이 둘을 자동으로 돌리고 실패하면 학습을 시작하지 않는다.

**shape 검사만으로는 원리적으로 못 잡는 유형이 있다.** 2026-09-12에 이 유형으로 세 건이
나왔다 — 6144차원 descriptor의 1/3이 0이어도 차원은 6144 그대로다. 그래서 검사 도구가
값과 호출 경로를 직접 읽는다. **"config에 켰다"와 "실제로 값이 흐른다"는 별개다.**

학습이 시작된 뒤에도 epoch 1 체크포인트가 나오면:
```bash
python tools/check_accel_block_trained.py <ckpt> --config <train_config>
```

## 실행 스크립트 — 어느 것이 현재 계보인가

현재 계보 (이것만 쓴다):
- `run_stage1_best.sh <nolcf|lcfon>` — stage1 두 도너
- `run_stage2_best.sh <teacher|student>` — stage2. student가 제출 모델
- `eval_l2.sh`, `tail_log.sh`

**과거 계보 (기록·폴백용, 새로 돌리지 말 것):**
`run_A_teacher.sh` / `run_A_student.sh`는 현재 최고 기록 **0.4218**을 낸 v1 계보를
재현한다. 새 계보가 그걸 못 넘으면 여기로 돌아간다.
`run_B_teacher.sh`는 B teacher(0.2542)를 재현한다.
이들은 2프레임·grid4 descriptor 시절이라 **현재 config와 섞으면 안 된다.**

## A5000 에서는 평가가 안 된다

`gyuz_split2` 컨테이너는 **코드 디렉터리만 마운트**돼 있다:

    /media/vcl/SSD-DATA/gyuz/LAW_split -> /workspace/VAD

원본 데이터셋(`data/train/...`)이 없다. 학습은 `LoadETRIGeometryCache` 로 전처리된
캐시를 읽으므로 돌아가지만, eval config 는 `FastLoadMultiViewImageFromFiles` 로
원본 JPEG 을 읽어서 `FileNotFoundError` 로 죽는다.

**평가는 3090 에서 한다.** A5000 에서 꼭 해야 하면 val 캐시
(`work_dirs/etri_geometry_cache_val_v1`)를 쓰는 test 파이프라인이 필요하고, 그러면
픽셀이 3.5% 달라지므로 3090 에서 잰 기존 수치와 직접 비교하면 안 된다.

## 데이터 경로 함정

config에 적힌 `ann_file`은 **학습에 쓰이지 않는다.** `scripts/_common.sh`의
`launch_train`이 `--cfg-options`로 세 필드를 전부 덮어쓰고, config 안의 경로
(`.causal_regen_split_301_75`, `_10hz` 없음)는 두 머신 어디에도 존재하지 않는다.
실제 경로는 `_common.sh`의 `ANN_DIR`이다. 도구를 새로 만들 때 config의 경로를
믿으면 검사가 조용히 자기를 건너뛴다 (실제로 그랬다).
