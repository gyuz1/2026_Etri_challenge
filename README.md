# LAW_fulldata

Self-contained VAD-tiny + LAW (latent world model) training/inference package
for the ETRI 2026 E2E Driving Challenge. Anyone with the competition dataset
and the competition's Docker image can reproduce the full pipeline
(stage1 -> merge -> stage2 -> test-set submission -> FLOPs -> T_infer) from
this folder alone.

Trains on **all 376 scenes** (no train/val split held out) -- this is the
final-submission variant. It reuses the same causal ego-motion fix, 5s
target-point goal-conditioning branch, and geometry-cache speedups
developed on a separate 301/75-scene split track; only the training data
scope differs.

---

## 1. Prerequisites

- **Dataset**: the ETRI challenge's `train/` and `test/` directories
  (see layout below). Not included here -- download separately from the
  competition materials.
- **Pretrained checkpoints**: not tracked in git (large binaries) -- place
  both under `ckpts/` yourself before running stage 1/2.
  - `ckpts/resnet50-19c8e357.pth` -- standard torchvision ImageNet
    backbone, download directly:
    ```bash
    wget -P ckpts/ https://download.pytorch.org/models/resnet50-19c8e357.pth
    ```
  - `ckpts/law_pretrained_nus.pth` -- VAD-tiny + LAW world model,
    nuScenes-pretrained (stage 2, epoch 12), trained in-house. Hosted on
    the Hugging Face Hub:
    ```bash
    wget -O ckpts/law_pretrained_nus.pth \
      https://huggingface.co/gyuz/law-pretrained-nus/resolve/main/law_pretrained_nus.pth
    ```
- **Docker image**: the competition-provided `etri-vad:cu128` image
  (Ubuntu 24.04, CUDA 12.8, Python 3.10 via `/opt/venv`, PyTorch 2.7.1,
  mmcv-full 1.4.0, mmdet 2.14.0, mmdet3d 0.17.1). No Dockerfile is provided
  by the competition -- load the image tar they distribute:
  ```bash
  sudo docker load -i etri-vad_cu128.tar
  ```
- **GPU**: developed and verified on RTX 3090 (x2). Should run unchanged
  on RTX 4090 (same CUDA/Ampere+ compute capability family, no
  hardware-specific code) -- the competition itself grades T_infer on a
  single RTX 4090. Needs 2 GPUs for the `--nproc_per_node=2` distributed
  training commands below; drop to 1 by editing that flag (and roughly
  halving `samples_per_gpu` in the configs if you hit a memory ceiling).

## 2. Dataset layout

Place the downloaded dataset into `dataset/train/` and `dataset/test/`
(empty placeholders in this folder), then build the annotation pkls into
`dataset/etri/annotations/` (see §4 -- not bundled here, too large for a
plain git push) so it looks like:
```
dataset/
├── train/{YYYYMMDD}-{HHMMSS}/        # 376 scenarios -- add this (download)
│   ├── camera_front/ camera_front_left/ camera_front_right/
│   ├── camera_rear_left/ camera_rear_right/ camera_rear_wide/
│   ├── calibration/calibration.parquet
│   ├── annotation/{object,ego_pose,map,hd_map,hd_ego_pose}.parquet
│   └── meta/{timestamps,command}.parquet
├── test/{clip_hash}/                 # 1,125 clips -- add this (download)
│   ├── camera_*/frame_{-30..0}.jpg
│   ├── ego_pose.parquet  calibration.parquet  command.parquet
└── etri/annotations/                 # build this, see §4
    ├── vad_etri_infos_fulldata_train.pkl
    └── vad_etri_infos_fulldata_test.pkl
```

## 3. Container setup

```bash
sudo docker run --gpus all -it --ipc=host \
  -v /path/to/LAW_fulldata:/workspace/VAD \
  -v /path/to/LAW_fulldata/dataset:/workspace/VAD/data \
  -v /path/to/LAW_fulldata/work_dirs:/workspace/VAD/work_dirs \
  -w /workspace/VAD \
  --name ad2026 \
  etri-vad:cu128 bash
```
(Three separate mounts: the code folder as `/workspace/VAD`, its own
`dataset/` subfolder re-mounted over `/workspace/VAD/data`, and `work_dirs/`
re-mounted over `/workspace/VAD/work_dirs` -- matches the mount pattern the
pipeline scripts' path constants assume. The `work_dirs` mount matters:
checkpoints/logs/wandb otherwise live only in the container's own writable
layer, which a 40+ hour training run really shouldn't risk losing to an
accidental `docker rm`.)

Re-attach later with `docker start -i ad2026` (stopped) or
`docker exec -it ad2026 bash` (already running).

`wandb` is not part of the competition's base image -- install it once per
container before running stage 1/2 (or skip this and remove the
`WandbLoggerHook` entries from `fulldata/configs/*.py`'s `log_config` to
disable logging instead):
```bash
python3 -m pip install wandb
wandb login
```

## 4. Annotation pkls

Not bundled in this repo (~1.5GB total, over GitHub's plain-push size
limit) -- build them once your `train/`/`test/` folders (§2) are in place.
The fulldata pipeline only needs the two output files
(`vad_etri_infos_fulldata_{train,test}.pkl`). `--val-scenes` is required by
the script but its split output (`..._train_split.pkl`/`..._val_split.pkl`)
isn't used by the fulldata pipeline -- `splits/val_scenes.txt` (bundled
here) is passed only to satisfy that requirement:
```bash
python tools/data_converter/regenerate_causal_infos.py \
  --train-root data/train --test-root data/test \
  --output-dir data/etri/annotations \
  --val-scenes splits/val_scenes.txt
```
Rename the two files this produces (`vad_etri_infos_temporal_{train,test}.pkl`)
to `vad_etri_infos_fulldata_{train,test}.pkl` to match what the fulldata
configs (§5) expect.

## 5. Run everything

Run from `/workspace/VAD` inside the container (the script itself also
`cd`s there, but it has to be found first):
```bash
cd /workspace/VAD
nohup bash fulldata/run_fulldata_pipeline.sh \
    > work_dirs/fulldata_pipeline.log 2>&1 &
# or inside tmux:
#   tmux new -s fulldata_pipeline
#   cd /workspace/VAD && bash fulldata/run_fulldata_pipeline.sh 2>&1 | tee work_dirs/fulldata_pipeline.log
```
Safe to re-run after an interruption -- each stage is skipped if its output
file already exists. Stages, in order:
1. Build the geometry cache (all 376 scenes; CPU/disk only)
2. Stage 1 -- `fulldata/configs/VAD_etri_tiny_stage1_cached_fulldata.py`, 48 epochs
3. Merge stage1 perception weights with the LAW world model
   (`ckpts/law_pretrained_nus.pth`)
4. Stage 2 -- `fulldata/configs/VADLAW_etri_tiny_cached_fulldata.py`, 12 epochs
5. Test-set inference -> `work_dirs/stage2_etri_fulldata_submission.json`
6. FLOPs measurement -> merged into the same submission.json (`__flops__`)
7. T_infer measurement -> `work_dirs/stage2_etri_fulldata_t_infer.log`
   (needs `/data_fast/test` to exist first -- a slow disk under `data/test`
   otherwise dominates the measured latency and makes the T_infer number
   meaningless. If `/data_fast` isn't set up yet, this step stops with an
   error; create it once and re-run:
   `mkdir -p /data_fast && cp -r data/test /data_fast/test`. Point
   `NVME_TEST_DIR` in the script at whatever fast local disk you have if
   `/data_fast` doesn't apply to your machine.)

No hold-out L2 step -- there's no val split left once every scene is in
the training set. wandb logging is enabled by default
(`project=etri-2026-e2e-vad`, run names `stage1_fulldata`/`stage2_fulldata`,
see §3 for setup).

## 6. What's in here

```
LAW_fulldata/
├── dataset/            # empty -- put the downloaded dataset here
├── work_dirs/           # empty -- checkpoints/logs/wandb land here
├── splits/              # val_scenes.txt (301/75 split file, see §4)
├── ckpts/                # empty -- place pretrained weights here, see §1
├── projects/             # model code (VAD/LAW architecture, configs)
├── tools/                # data converters, train/test/cache/measurement scripts
└── fulldata/
    ├── configs/           # 376-scene (no split) training configs
    └── run_fulldata_pipeline.sh
```
