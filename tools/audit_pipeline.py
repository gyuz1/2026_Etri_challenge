"""One pass over a train config that checks the things this project keeps
getting silently wrong.

Every failure mode below has actually happened here, and none of them raised
at the time -- mmcv loads checkpoints with strict=False, so a mismatched or
missing module is a warning line in a log nobody reads, not a crash. The
22h-wasted incidents were all of this shape.

Checks, in order of how expensive the mistake was:

  1. DONOR SHAPES. Does load_from's ego_fut_decoder actually fit the model
     being built? A 576-wide donor in a 520-wide model silently trains from
     a random planner.
  2. TRAIN/EVAL PARITY. Does the paired eval config build the identical
     network? A setting that changes a module's width must be repeated there
     or evaluation scores a differently-initialized model.
  3. COMPLIANCE. For a submittable (student) config: ego_lcf must not reach
     the planner. Checks ego_lcf_feat_idx is None and that the status slot,
     if any, comes from the vision estimator.
  4. TEACHER/STUDENT SYMMETRY. The student distills ego_scene_feats from the
     teacher's BEV encoder, so both sides' motion descriptors should be
     built the same way -- otherwise the student is asked to reproduce
     structure the teacher was never pushed to encode.
  5. DATA LEAKAGE. train ann_file must not contain val scenes, and the KD
     teacher cache (when used) must not be read for val scenes.

Usage:
    python tools/audit_pipeline.py <train_config> [--eval-config <cfg>]
                                   [--ann-dir <dir>]
"""
import argparse
import os
import pickle

import mmcv
import torch
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401


OK, WARN, FAIL = '  [OK]  ', '  [!!]  ', '  [FAIL]'


class Audit:
    def __init__(self):
        self.failed = False

    def ok(self, msg):
        print(OK + msg)

    def warn(self, msg):
        print(WARN + msg)

    def fail(self, msg):
        print(FAIL + ' ' + msg)
        self.failed = True


def build(path):
    cfg = mmcv.Config.fromfile(path)
    model_cfg = cfg.model.copy()
    # Training-only object whose checkpoint may not exist yet.
    model_cfg.pop('feature_distill_teacher_cfg', None)
    model_cfg.pop('feature_distill_teacher_ckpt', None)
    model = build_model(model_cfg, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    return cfg, model


def head_cfg(cfg):
    return cfg.model.pts_bbox_head


def check_donor(a, cfg, model):
    print('\n1. 도너 체크포인트 shape')
    lf = cfg.get('load_from')
    if not lf:
        a.warn('load_from 없음 (scratch 학습이면 정상)')
        return
    if not os.path.exists(lf):
        a.warn(f'load_from 파일이 이 머신에 없음: {lf}')
        return
    sd = torch.load(lf, map_location='cpu')
    sd = sd.get('state_dict', sd)
    msd = model.state_dict()
    bad = [(k, tuple(sd[k].shape), tuple(msd[k].shape))
           for k in sd if k in msd and sd[k].shape != msd[k].shape]
    dec = [b for b in bad if 'ego_fut_decoder' in b[0]]
    # A stage-1 config warm-starts from the nuScenes LAW checkpoint, whose
    # decoder cannot transfer by construction: it was trained with a
    # different ego_fut_mode, so the final layer's shape differs and the
    # hidden width follows from that. Reinitializing there is correct, and
    # is what every stage 1 in this repo has always done. Only a stage-2
    # donor (a stage2_init_merged_*.pth) is supposed to carry the decoder.
    stage1_warmstart = 'law_pretrained_nus' in os.path.basename(lf)
    if dec and stage1_warmstart:
        a.ok(f'ego_fut_decoder {len(dec)}개 불일치 -- nuScenes warm-start 라 '
             '정상 (플래너는 stage1 에서 새로 학습)')
    elif dec:
        a.fail(f'ego_fut_decoder {len(dec)}개 불일치 -> 플래너가 랜덤 초기화됨')
        for k, s, m in dec:
            print(f'          {k}: donor{s} vs model{m}')
    else:
        a.ok(f'ego_fut_decoder 전이 정상 ({os.path.basename(lf)})')
    other = [b for b in bad if 'ego_fut_decoder' not in b[0]]
    if other:
        a.warn(f'기타 {len(other)}개 불일치 (새 모듈이면 정상)')
        for k, s, m in other[:5]:
            print(f'          {k}: donor{s} vs model{m}')


def check_parity(a, train_path, eval_path):
    print('\n2. train/eval config 일치')
    if not eval_path:
        a.warn('eval config 미지정 -- 건너뜀')
        return
    _, tm = build(train_path)
    _, em = build(eval_path)
    t, e = tm.state_dict(), em.state_dict()
    mism = sorted(k for k in t.keys() & e.keys() if t[k].shape != e[k].shape)
    eval_only = sorted(e.keys() - t.keys())
    if mism:
        a.fail(f'{len(mism)}개 shape 불일치 -> 평가 시 조용히 랜덤 초기화')
        for k in mism[:5]:
            print(f'          {k}: train{tuple(t[k].shape)} vs eval{tuple(e[k].shape)}')
    else:
        a.ok('shape 전부 일치')
    if eval_only:
        a.fail(f'eval 에만 있는 모듈 {len(eval_only)}개 (체크포인트에 가중치 없음)')
    else:
        a.ok('eval 전용 모듈 없음')


def check_compliance(a, cfg):
    print('\n3. 규정 (제출 모델 기준)')
    h = head_cfg(cfg)
    lcf_idx = h.get('ego_lcf_feat_idx')
    est = h.get('ego_status_est_dim')
    if lcf_idx is None:
        a.ok('ego_lcf_feat_idx=None -- 실제 자기상태가 플래너로 안 감')
        if est:
            a.ok(f'상태 슬롯 {est}차원은 ego_status_est_net(BEV) 에서 생성 '
                 '-- 2026-09-04 Q&A 가 허용한 vision 추론값')
    else:
        a.warn(f'ego_lcf_feat_idx={lcf_idx} -- 제출 불가 (teacher 면 정상)')
    if h.get('target_point_shortcut'):
        a.fail('target_point_shortcut 켜짐 -- 목표점이 생성에 개입, 제출 불가')
    else:
        a.ok('target_point_shortcut 꺼짐')
    if h.get('aux_bev_motion_feedback'):
        a.warn('aux_bev_motion_feedback 켜짐 -- 과거 측정 0.5635->0.6419 악화')


def check_silent_noops(a, cfg):
    """Settings that build cleanly but do nothing unless a second flag is on.

    These are worse than a crash: the config reads as intended, the model
    builds, training runs to completion, and the feature was never active.
    """
    print('\n3b. 조용한 no-op')
    h = head_cfg(cfg)
    frames = h.get('aux_bev_motion_frames')
    grid = h.get('aux_bev_motion_grid')
    temporal = h.get('aux_bev_motion_temporal')
    future = h.get('aux_bev_future_motion')
    motion = h.get('aux_bev_motion')

    if (frames or grid) and not temporal:
        a.fail('aux_bev_motion_frames/grid 를 설정했는데 '
               'aux_bev_motion_temporal 이 꺼져 있음 -- descriptor 가 단일 '
               '프레임 global mean 으로 떨어져 두 설정 모두 무시된다')
    elif temporal:
        a.ok(f'temporal descriptor 활성 (frames={frames or 2}, '
             f'grid={grid or 4})')
    if future and not motion:
        a.fail('aux_bev_future_motion 은 aux_bev_motion 의 descriptor 를 '
               '읽는다 -- aux_bev_motion=True 필요')
    idx = h.get('aux_bev_motion_idx') or ()
    norm = h.get('aux_bev_motion_norm')
    if idx and not norm:
        a.warn(f'aux_bev_motion_idx={idx} 인데 정규화 없음 -- 측정상 vx/speed 가 '
               'L1 의 98.7%, yaw_rate 는 0.1% 로 사실상 무감독')
    elif norm and len(norm) != len(idx):
        a.fail(f'aux_bev_motion_norm 길이 {len(norm)} != idx 길이 {len(idx)}')


def check_symmetry(a, cfg, train_path):
    print('\n4. teacher/student descriptor 대칭')
    tcfg_path = cfg.model.get('feature_distill_teacher_cfg')
    if not tcfg_path:
        a.warn('증류 teacher 없음 -- 건너뜀')
        return
    if not os.path.exists(tcfg_path):
        a.fail(f'teacher config 파일 없음: {tcfg_path}')
        return
    tcfg = mmcv.Config.fromfile(tcfg_path)
    th, sh = head_cfg(tcfg), head_cfg(cfg)
    keys = ('aux_bev_motion_frames', 'aux_bev_motion_grid',
            'aux_bev_motion_proj_dim', 'aux_bev_motion_temporal',
            'aux_bev_motion_idx')
    diff = [(k, th.get(k), sh.get(k)) for k in keys if th.get(k) != sh.get(k)]
    if diff:
        a.warn('teacher 와 student 의 descriptor 설정이 다름 -- '
               'student 가 teacher 가 인코딩한 적 없는 구조를 재현하게 됨')
        for k, tv, sv in diff:
            print(f'          {k}: teacher={tv} student={sv}')
    else:
        a.ok('descriptor 설정 동일')
    ckpt = cfg.model.get('feature_distill_teacher_ckpt')
    if ckpt and not os.path.exists(ckpt):
        a.warn(f'teacher 체크포인트 아직 없음: {ckpt} (학습 전이면 정상)')


def check_leakage(a, cfg, ann_dir):
    print('\n5. 데이터 누수')
    # scripts/_common.sh launch_train overrides all three ann_file fields via
    # --cfg-options, so the path written in the config is NOT what trains.
    # Auditing the config's own path silently skipped this check entirely
    # (the configs still name a directory that does not exist on either
    # machine), which made the leakage check a no-op on every run so far.
    # Resolve the same way the launcher does.
    cfg_ann = cfg.data.train.get('ann_file')
    train_ann = os.path.join(ann_dir, 'vad_etri_infos_temporal_train_split.pkl')
    val_ann = os.path.join(ann_dir, 'vad_etri_infos_temporal_val_split.pkl')
    print(f'          config 의 train ann_file : {cfg_ann}')
    print(f'          실제 학습에 쓰이는 ann_dir: {ann_dir}')
    print('          (launch_train 이 --cfg-options 로 덮어쓴다)')
    missing = [p for p in (train_ann, val_ann) if not os.path.exists(p)]
    if missing:
        a.fail('ann 파일이 없음 -- 검사 5 를 수행할 수 없다. '
               '--ann-dir 로 실제 경로를 지정할 것')
        for p in missing:
            print(f'          없음: {p}')
        return
    def scenes(p):
        with open(p, 'rb') as fh:
            return {i['scene_token'] for i in pickle.load(fh)['infos']}
    tr, va = scenes(train_ann), scenes(val_ann)
    overlap = tr & va
    if overlap:
        a.fail(f'train 과 val 이 {len(overlap)}개 scene 겹침 -- 평가가 무의미해짐')
    else:
        a.ok(f'train {len(tr)} / val {len(va)} scene, 겹침 0')

    # KD teacher cache: only matters if the pipeline actually loads it.
    pipe = cfg.data.train.get('pipeline', [])
    tcache = next((t.get('cache_path') for t in pipe
                   if t.get('type') == 'LoadTeacherWaypoints'), None)
    if tcache is None:
        a.ok('LoadTeacherWaypoints 없음 -- Qwen KD 캐시 미사용')
    else:
        a.warn(f'Qwen KD 캐시 사용: {tcache}')
        print('          캐시에 val scene 항목이 있어도 train ann_file 로만 '
              '조회되므로 누수는 아니지만, hold-out 0.3511 / train 0.1798 인 '
              '암기 신호임을 기억할 것')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    p.add_argument('--eval-config')
    p.add_argument('--ann-dir',
                   default='data/etri/.causal_regen_split_301_75_10hz',
                   help='실제 학습이 쓰는 ann 디렉터리. scripts/_common.sh 의 '
                        'ANN_DIR 과 같아야 한다 (config 안의 경로가 아니다)')
    args = p.parse_args()

    print('=' * 70)
    print('감사 대상:', args.train_config)
    print('=' * 70)
    a = Audit()
    cfg, model = build(args.train_config)
    check_donor(a, cfg, model)
    check_parity(a, args.train_config, args.eval_config)
    check_compliance(a, cfg)
    check_silent_noops(a, cfg)
    check_symmetry(a, cfg, args.train_config)
    check_leakage(a, cfg, args.ann_dir)
    print('\n' + '=' * 70)
    print('판정:', '실패 -- 위 [FAIL] 항목 해결 전 학습 금지' if a.failed
          else '통과')
    return 1 if a.failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
