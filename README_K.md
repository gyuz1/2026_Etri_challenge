# LAW_fulldata

ETRI 2026 E2E Driving Challenge용 VAD-tiny + LAW(latent world model) 학습/추론
독립 패키지입니다. 대회 데이터셋과 대회 Docker 이미지만 있으면 이 폴더 하나로
전체 파이프라인(stage1 -> merge -> stage2 -> test-set 제출 -> FLOPs -> T_infer)을
재현할 수 있습니다.

**전체 376개 씬**으로 학습합니다 (train/val 분리 없음) -- 최종 제출용 버전입니다.
별도의 301/75 씬 분리 트랙에서 쓰던 것과 동일한 causal ego-motion fix, 5초
target-point goal-conditioning branch, geometry-cache 속도 최적화를 그대로
사용하고, 학습 데이터 범위만 다릅니다.

---

## 1. 사전 준비

- **데이터셋**: ETRI 챌린지의 `train/`, `test/` 디렉토리 (아래 레이아웃 참고).
  여기 포함되어 있지 않음 -- 대회 자료에서 별도로 다운로드.
- **사전학습 체크포인트**: git에 안 올라가 있음 (큰 바이너리) -- stage 1/2
  돌리기 전에 직접 `ckpts/` 밑에 받아둘 것.
  - `ckpts/resnet50-19c8e357.pth` -- torchvision 표준 ImageNet 백본,
    바로 다운로드:
    ```bash
    wget -P ckpts/ https://download.pytorch.org/models/resnet50-19c8e357.pth
    ```
  - `ckpts/law_pretrained_nus.pth` -- VAD-tiny + LAW world model,
    nuScenes로 자체 학습(stage 2, epoch 12)한 체크포인트. Hugging Face Hub에
    올려둠:
    ```bash
    wget -O ckpts/law_pretrained_nus.pth \
      https://huggingface.co/gyuz/law-pretrained-nus/resolve/main/law_pretrained_nus.pth
    ```
- **Docker 이미지**: 대회에서 제공하는 `etri-vad:cu128` 이미지
  (Ubuntu 24.04, CUDA 12.8, Python 3.10 `/opt/venv`, PyTorch 2.7.1,
  mmcv-full 1.4.0, mmdet 2.14.0, mmdet3d 0.17.1). 대회 쪽에서 Dockerfile은
  따로 공개하지 않고, 빌드된 이미지 tar만 배포함 -- 그 tar를 로드:
  ```bash
  sudo docker load -i etri-vad_cu128.tar
  ```
- **GPU**: RTX 3090 (x2) 환경에서 개발/검증함. RTX 4090에서도 그대로
  동작해야 함 (같은 CUDA/Ampere+ compute capability 계열, 하드웨어 종속
  코드 없음) -- 대회 채점 자체가 단일 RTX 4090 기준 T_infer 측정. 아래
  학습 명령들이 `--nproc_per_node=2` 분산학습을 쓰므로 GPU 2개 필요;
  1개만 쓰려면 그 플래그를 수정하고 (메모리 부족하면 configs의
  `samples_per_gpu`도 대략 절반으로).

## 2. 데이터셋 레이아웃

다운로드한 데이터셋을 `dataset/train/`, `dataset/test/`에 넣고 (이 폴더
안엔 빈 placeholder만 있음), annotation pkl을 `dataset/etri/annotations/`에
만들면 (§4 참고 -- 여기엔 안 들어있음, git으로 올리기엔 너무 큼) 아래
구조가 됩니다:
```
dataset/
├── train/{YYYYMMDD}-{HHMMSS}/        # 376개 시나리오 -- 추가할 것 (다운로드)
│   ├── camera_front/ camera_front_left/ camera_front_right/
│   ├── camera_rear_left/ camera_rear_right/ camera_rear_wide/
│   ├── calibration/calibration.parquet
│   ├── annotation/{object,ego_pose,map,hd_map,hd_ego_pose}.parquet
│   └── meta/{timestamps,command}.parquet
├── test/{clip_hash}/                 # 1,125개 클립 -- 추가할 것 (다운로드)
│   ├── camera_*/frame_{-30..0}.jpg
│   ├── ego_pose.parquet  calibration.parquet  command.parquet
└── etri/annotations/                 # 여기서 만들 것, §4 참고
    ├── vad_etri_infos_fulldata_train.pkl
    └── vad_etri_infos_fulldata_test.pkl
```

## 3. 컨테이너 설정

```bash
sudo docker run --gpus all -it --ipc=host \
  -v /path/to/LAW_fulldata:/workspace/VAD \
  -v /path/to/LAW_fulldata/dataset:/workspace/VAD/data \
  -v /path/to/LAW_fulldata/work_dirs:/workspace/VAD/work_dirs \
  -w /workspace/VAD \
  --name ad2026 \
  etri-vad:cu128 bash
```
(마운트 3개: 코드 폴더를 `/workspace/VAD`로, 그 안의 `dataset/` 하위폴더를
다시 `/workspace/VAD/data`에, `work_dirs/`를 `/workspace/VAD/work_dirs`에 --
파이프라인 스크립트들의 경로 상수가 가정하는 마운트 패턴과 일치. `work_dirs`
마운트가 중요한 이유: 안 그러면 체크포인트/로그/wandb가 컨테이너 자체의
writable layer에만 남는데, 40시간 넘게 걸리는 학습을 실수로 `docker rm` 한
번에 날릴 위험을 감수할 이유가 없음.)

나중에 다시 붙을 때는 `docker start -i ad2026` (멈춰있으면) 또는
`docker exec -it ad2026 bash` (이미 실행 중이면).

`wandb`는 대회 기본 이미지에 포함되어 있지 않음 -- stage 1/2 돌리기 전에
컨테이너마다 한 번 설치 (또는 설치 생략하고 `fulldata/configs/*.py`의
`log_config`에서 `WandbLoggerHook` 항목을 지워서 로깅 자체를 끄기):
```bash
python3 -m pip install wandb
wandb login
```

## 4. Annotation pkl

이 repo엔 안 들어있음 (총 ~1.5GB, GitHub 일반 push 용량 제한 초과) --
`train/`/`test/` 폴더(§2)가 준비되면 직접 생성. fulldata 파이프라인은
결과물 두 개(`vad_etri_infos_fulldata_{train,test}.pkl`)만 필요함.
`--val-scenes`는 스크립트가 요구하는 필수 인자지만 그 결과물
(`..._train_split.pkl`/`..._val_split.pkl`)은 fulldata 파이프라인에서
안 쓰임 -- `splits/val_scenes.txt`(여기 포함되어 있음)는 그 요구조건만
맞추기 위해 넘기는 것:
```bash
python tools/data_converter/regenerate_causal_infos.py \
  --train-root data/train --test-root data/test \
  --output-dir data/etri/annotations \
  --val-scenes splits/val_scenes.txt
```
이 명령이 만든 두 파일(`vad_etri_infos_temporal_{train,test}.pkl`)을
`vad_etri_infos_fulldata_{train,test}.pkl`로 이름을 바꿔서 fulldata
configs(§5)가 기대하는 이름과 맞출 것.

## 5. 전체 실행

컨테이너 안 `/workspace/VAD`에서 실행할 것 (스크립트 내부에서도 `cd`
하지만, 그 전에 스크립트 자체를 찾을 수 있어야 함):
```bash
cd /workspace/VAD
nohup bash fulldata/run_fulldata_pipeline.sh \
    > work_dirs/fulldata_pipeline.log 2>&1 &
# 또는 tmux 안에서:
#   tmux new -s fulldata_pipeline
#   cd /workspace/VAD && bash fulldata/run_fulldata_pipeline.sh 2>&1 | tee work_dirs/fulldata_pipeline.log
```
중간에 끊겨도 다시 실행하면 안전함 -- 각 단계는 결과 파일이 이미 있으면
건너뜀. 순서대로:
1. geometry cache 구축 (376개 씬 전체; CPU/디스크만 사용)
2. Stage 1 -- `fulldata/configs/VAD_etri_tiny_stage1_cached_fulldata.py`, 48 epoch
3. stage1 perception 가중치와 LAW world model 병합
   (`ckpts/law_pretrained_nus.pth`)
4. Stage 2 -- `fulldata/configs/VADLAW_etri_tiny_cached_fulldata.py`, 12 epoch
5. 테스트셋 추론 -> `work_dirs/stage2_etri_fulldata_submission.json`
6. FLOPs 측정 -> 같은 submission.json에 병합 (`__flops__`)
7. T_infer 측정 -> `work_dirs/stage2_etri_fulldata_t_infer.log`
   (`/data_fast/test`가 먼저 있어야 함 -- 느린 디스크 위에서 재면 그
   디스크 I/O가 측정값을 지배해서 T_infer 숫자 자체가 의미 없어짐.
   `/data_fast`가 아직 없으면 이 단계가 에러로 멈춤; 한 번만 만들고
   재실행: `mkdir -p /data_fast && cp -r data/test /data_fast/test`.
   `/data_fast`가 이 머신 사정에 안 맞으면 스크립트의 `NVME_TEST_DIR`을
   실제로 쓸 수 있는 빠른 로컬 디스크로 바꿀 것.)

hold-out L2 단계는 없음 -- 전체 씬이 다 학습셋에 들어가므로 남는 val
split이 없음. wandb 로깅은 기본 활성화되어 있음
(`project=etri-2026-e2e-vad`, run 이름 `stage1_fulldata`/`stage2_fulldata`,
설정은 §3 참고).

## 6. 폴더 구성

```
LAW_fulldata/
├── dataset/            # 비어있음 -- 다운로드한 데이터셋을 여기에
├── work_dirs/           # 비어있음 -- 체크포인트/로그/wandb가 여기 쌓임
├── splits/              # val_scenes.txt (301/75 split 파일, §4 참고)
├── ckpts/                # 비어있음 -- 사전학습 가중치를 여기에, §1 참고
├── projects/             # 모델 코드 (VAD/LAW 아키텍처, configs)
├── tools/                # 데이터 컨버터, train/test/cache/measurement 스크립트
└── fulldata/
    ├── configs/           # 376씬(분리 없음) 학습 configs
    └── run_fulldata_pipeline.sh
```
