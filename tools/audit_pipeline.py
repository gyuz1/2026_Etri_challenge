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
import re
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
    # goal_pred heads lengthen the last decoder layer from fut_ts to
    # goal_long_ts steps on purpose, and VADHead._load_from_state_dict copies
    # the donor into the first fut_ts steps and repeats its last step. Don't
    # trust the shape arithmetic: run the real load and compare bit for bit.
    head = model.pts_bbox_head
    if dec and getattr(head, 'goal_pred', False):
        model.load_state_dict(dict(sd), strict=False)
        ext_ok = True
        m, t, tl = head.ego_fut_mode, head.fut_ts, head.ego_dec_ts
        for key, s_, _ in dec:
            got = model.state_dict()[key]
            src = sd[key].reshape(m, t, 2, *sd[key].shape[1:])
            new = got.reshape(m, tl, 2, *got.shape[1:])
            if not torch.equal(new[:, :t], src):
                ext_ok = False
                print(f'          {key}: 앞 {t} 스텝이 donor 와 다름')
            if not torch.equal(new[:, t:], src[:, -1:].expand_as(new[:, t:])):
                ext_ok = False
                print(f'          {key}: 연장 스텝이 donor 마지막 스텝 복사가 아님')
        if ext_ok:
            a.ok(f'ego_fut_decoder 마지막 층 donor{dec[0][1]} -> model{dec[0][2]}: '
                 f'앞 {t}스텝 donor 그대로, {tl - t}스텝 등속 연장 (실제 load 로 확인)')
            dec = []
        else:
            a.fail('goal_pred 디코더 연장 실패 -> 플래너 일부가 랜덤 초기화됨')
            return
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
    check_behaviour_flags(a, train_path, eval_path)


# Settings that may differ between train and eval because they only shape a
# loss, a dropout, or a training-only module whose forward is gated on
# self.training. Everything NOT matched here must be identical.
TRAIN_ONLY_KEYS = re.compile(
    r'(^|\.)('
    r'loss_[a-z_]+\.loss_weight|[a-z_]*_weight|echo_cycle_weight|'
    r'prev_bev_dropout|ego_status_est_dropout|aux_bev_motion_norm|'
    r'aux_ego_motion(_idx)?|ego_status_decode|ego_status_distill_idx|'
    r'plan_reg_ts_weight_mode|privileged_distill(_idx)?|'
    r'remove_auxiliary_planning_losses|feature_distill_[a-z_]+|disable_dropout|'
    r'train_cfg\..*'
    r')$')


def _flat(d, prefix=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f'{prefix}{k}.'))
    elif (isinstance(d, (list, tuple)) and d
          and all(isinstance(x, dict) for x in d)):
        for i, v in enumerate(d):
            out.update(_flat(v, f'{prefix}{i}.'))
    else:
        out[prefix[:-1]] = d
    return out


def check_behaviour_flags(a, train_path, eval_path):
    # The shape check above cannot see a flag that changes the forward pass
    # without changing a tensor. On 2026-09-15 two eval configs were wrong in
    # exactly that way and would both have produced numbers without an error:
    # a duplicated top-level `model = dict(...)` dropped grid 8 (that one did
    # change shapes), and the teacher's eval config lacked
    # ego_lcf_embed_residual=True (that one did not).
    tr = _flat(mmcv.Config.fromfile(train_path).model.to_dict())
    ev = _flat(mmcv.Config.fromfile(eval_path).model.to_dict())
    bad = []
    for k in sorted(set(tr) | set(ev)):
        if TRAIN_ONLY_KEYS.search(k):
            continue
        if tr.get(k, '<없음>') != ev.get(k, '<없음>'):
            bad.append((k, tr.get(k, '<없음>'), ev.get(k, '<없음>')))
    if bad:
        a.fail(f'동작을 바꾸는 설정 {len(bad)}개가 train/eval 에서 다름')
        for k, x, y in bad:
            print(f'          {k}: train={x!r} eval={y!r}')
    else:
        a.ok('동작 설정 전부 일치 (loss/dropout/학습 전용 항목 제외)')


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

    # Project rule [사용자 2026-09-15]: ground truth is used the way the
    # command is, and only the command is a permitted test-time input. So the
    # target point may be a training label, never something inference reads
    # -- not in the network, and not in post-processing that picks among the
    # network's trajectories (Q&A A1). The head gates its one read on
    # self.training; the scripts that run inference must not read it at all.
    import re as _re
    for rel in ('tools/etri_test_submit.py',
                'tools/eval_holdout_l2_and_tinfer.py'):
        if not os.path.exists(rel):
            continue
        hits = []
        lines = open(rel).read().split('\n')
        for n, line in enumerate(lines, 1):
            code = line.split('#', 1)[0]
            if not _re.search(r'target_point', code):
                continue
            stripped = code.strip()
            # argparse help text and similar string-only lines
            if stripped[:1] in ('"', "'") or stripped.startswith('help='):
                continue
            if "'--zero-target-point'" in code:
                continue
            # --zero-target-point ERASES the target point to prove a model
            # does not depend on it; that removes information rather than
            # using it, so it is the opposite of what this check guards.
            window = '\n'.join(lines[max(0, n - 8):n])
            if 'args.zero_target_point' in window:
                continue
            hits.append(n)
        if hits:
            a.fail(f'{rel} 이 추론 경로에서 target_point 를 읽는다 '
                   f'(줄 {hits[:5]}) -- GT 는 학습 라벨로만 쓴다')
        else:
            a.ok(f'{os.path.basename(rel)}: 추론에서 target_point 미사용')
    if h.get('goal_pred'):
        import inspect as _insp
        from projects.mmdet3d_plugin.VAD.VAD_head import VADHead as _VH
        fsrc = _insp.getsource(_VH.forward)
        gate = 'if self.training and ego_target_point is not None:'
        # The target point may appear only to build labels, inside the gate.
        gated = fsrc.split(gate, 1)
        outside = gated[0] if len(gated) == 2 else fsrc
        if (len(gated) != 2 or fsrc.count('_goal_label_from_target(') != 1
                or '_goal_label_from_target(' in outside):
            a.fail('goal_pred 의 target_point 읽기가 self.training 게이트 밖에 있다 '
                   '-- 추론에서 TP 를 쓰면 규정 위반')
        else:
            a.ok(f"goal_pred (전방 {len(h.get('goal_bin_edges')) - 1}구간 + 연속 오프셋): "
                 'TP 는 학습 라벨로만, 추론은 네트워크 예측 목표만 사용')


def check_train_eval_mismatch(a, cfg):
    """Settings that make training see something inference never does.

    Every one of these has cost this project results without raising: the
    network fits a regime it is not scored in. Measured 2026-09-15 for
    dropout alone: speed read +5% fast at inference, L2@3s 0.732 vs 0.553.
    A config that turns any of them on does not launch.
    """
    print('\n3c. 학습/추론 불일치 설정 (하나라도 켜져 있으면 학습 금지)')
    m = cfg.model
    h = m.pts_bbox_head
    bad = []
    if not m.get('disable_dropout', False):
        bad.append('disable_dropout 이 True 가 아님 -- nn.Dropout 이 학습에만 켜진다 '
                   '(속도 +5% 편향 실측)')
    if (m.get('prev_bev_dropout') or 0) > 0:
        bad.append(f"prev_bev_dropout={m.get('prev_bev_dropout')} -- 추론은 항상 이전 BEV 가 있다")
    if (h.get('ego_status_est_dropout') or 0) > 0:
        bad.append(f"ego_status_est_dropout={h.get('ego_status_est_dropout')} -- 추론은 슬롯을 항상 채운다")
    if h.get('prism_latent_supervision'):
        bad.append('prism_latent_supervision=True -- 학습은 GT 미래를 본 posterior, 추론은 prior 평균')
    if h.get('bev_residual_refine'):
        bad.append('bev_residual_refine=True -- L2 21~25% 악화 실측')
    if h.get('target_point_shortcut'):
        bad.append('target_point_shortcut=True -- 규정 위반 진단용')
    if h.get('privileged_distill'):
        bad.append('privileged_distill=True -- 학습 전용 특권 경로')
    # Yaw delta must be wrapped to [-180, 180) on the LAW path, in the training
    # queue and at inference (2026-09-15: unwrapped, 3.6% of frames fed ~359
    # into can_bus_mlp). Read the source, since both paths run without error
    # either way.
    try:
        law_ds = open('projects/mmdet3d_plugin/datasets/law_etri_dataset.py').read()
        law_md = open('projects/mmdet3d_plugin/LAW/VAD_LAW.py').read()
        wrap_ok = ('- previous_angle + 180) % 360 - 180' in law_ds
                   and '- self.prev_frame_info["prev_angle"] + 180) % 360 - 180' in law_md
                   and 'meta["can_bus"][-1] -= previous_angle' not in law_ds)
    except OSError:
        wrap_ok = False
    if not wrap_ok:
        bad.append('can_bus yaw 변화량이 LAW 학습 큐 또는 VADLAW 추론에서 ±180 으로 wrap 되지 않는다')
    if cfg.data.train.get('history_sampling', 'random') != 'fixed':
        bad.append("history_sampling 이 'fixed' 가 아님 -- 프레임 간격이 추론과 다르다")
    if bad:
        for x in bad:
            a.fail(x)
    else:
        a.ok('dropout·prev_bev_dropout·슬롯 dropout·PRISM·refine·shortcut 전부 꺼짐, 간격 고정, yaw wrap')


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
    # The 3-frame second difference is acceleration only when the two frame
    # gaps are equal. The dataset's default history sampling drops one of the
    # candidates at random, which makes them unequal in 67% of samples and
    # injects a v*dt term 46x the real a*dt^2 -- and evaluation always streams
    # equal gaps, so this is a train/eval mismatch on top of a wrong target.
    if frames and frames >= 3:
        sampling = cfg.data.train.get('history_sampling', 'random')
        if sampling != 'fixed':
            a.fail(f"aux_bev_motion_frames={frames} 인데 "
                   f"data.train.history_sampling='{sampling}' -- 학습 67% 에서 "
                   '프레임 간격이 불일치해 2차차분이 가속도가 아니게 된다. '
                   "'fixed' 필요")
        else:
            a.ok("history_sampling='fixed' -- 프레임 간격 균등 (평가와 동일)")

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
    # Anything that shapes ego_feats belongs here, not just the descriptor:
    # ego_scene_feats and ego_status_feats ARE the two halves of ego_feats, so
    # a loss the teacher never had pulls the student's features away from the
    # target it is being aligned to. bev_residual_refine additionally decides
    # whether the teacher is any good -- it costs 21-25% L2 (measured).
    keys = ('aux_bev_motion_frames', 'aux_bev_motion_grid',
            'aux_bev_motion_proj_dim', 'aux_bev_motion_temporal',
            'aux_bev_motion_idx', 'ego_status_distill_idx',
            'bev_residual_refine', 'bev_refine_steps',
            'aux_long_horizon', 'aux_long_horizon_residual',
            'ego_status_decode')
    diff = [(k, th.get(k), sh.get(k)) for k in keys if th.get(k) != sh.get(k)]
    if diff:
        a.warn('teacher 와 student 의 descriptor 설정이 다름 -- '
               'student 가 teacher 가 인코딩한 적 없는 구조를 재현하게 됨')
        for k, tv, sv in diff:
            print(f'          {k}: teacher={tv} student={sv}')
    else:
        a.ok('descriptor 설정 동일')
    # ego_lcf columns 5 and 6 are ego_length and ego_width: measured std
    # exactly 0 over the train split, one unique value each. Inside a cosine
    # target they are 21.5% of the squared norm on average and 99.7% of it on
    # stopped samples, so the student scores a near perfect cosine on the
    # frames where the teacher knew most. Narrowing the target costs nothing
    # -- ego_feats keeps all eight columns and the decoder stays 520 wide.
    if cfg.model.get('feature_distill_mode') == 'split':
        didx = sh.get('ego_status_distill_idx')
        tlcf = th.get('ego_lcf_feat_idx') or ()
        const_in_target = {5, 6} & set(tlcf)
        if const_in_target and (didx is None or const_in_target & set(didx)):
            a.warn('status 증류 타깃에 상수열(ego_length/ego_width)이 들어 있다 '
                   '-- 정지 샘플에서 타깃의 99.7% 가 상수라 cosine 이 공짜로 '
                   '맞는다. ego_status_distill_idx=(0,1,2,3,4,7) 권장')
        elif didx:
            a.ok(f'status 증류 타깃 = {didx} (상수열 제외)')

    # The teacher is loaded with mmcv's load_checkpoint, which is
    # strict=False like everything else here. A teacher whose config and
    # checkpoint disagree loads anyway, with the mismatched modules randomly
    # initialized -- and then the student spends 12 epochs aligning itself to
    # a target produced by an untrained network. Nothing raises, and the
    # distillation losses look perfectly healthy while it happens.
    ckpt = cfg.model.get('feature_distill_teacher_ckpt')
    if ckpt and not os.path.exists(ckpt):
        a.warn(f'teacher 체크포인트 아직 없음: {ckpt} (학습 전이면 정상)')
    elif ckpt:
        tmodel = build_model(tcfg.model, train_cfg=tcfg.get('train_cfg'),
                             test_cfg=tcfg.get('test_cfg'))
        tsd = torch.load(ckpt, map_location='cpu')
        tsd = tsd.get('state_dict', tsd)
        msd = tmodel.state_dict()
        bad = [(k, tuple(tsd[k].shape), tuple(msd[k].shape))
               for k in tsd if k in msd and tsd[k].shape != msd[k].shape]
        missing = [k for k in msd if k not in tsd]
        if bad:
            a.fail(f'teacher 체크포인트가 teacher config 와 {len(bad)}개 '
                   '불일치 -> 해당 모듈이 랜덤 초기화된 채 증류 타깃을 만든다')
            for k, sh, mh in bad[:5]:
                print(f'          {k}: ckpt{sh} vs model{mh}')
        if missing:
            a.fail(f'teacher 체크포인트에 없는 모듈 {len(missing)}개 '
                   '-> 랜덤 초기화')
            for k in missing[:5]:
                print(f'          {k}')
        if not bad and not missing:
            a.ok(f'teacher 체크포인트가 config 와 완전히 일치 '
                 f'({os.path.basename(ckpt)})')


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
    check_train_eval_mismatch(a, cfg)
    check_silent_noops(a, cfg)
    check_symmetry(a, cfg, args.train_config)
    check_leakage(a, cfg, args.ann_dir)
    print('\n' + '=' * 70)
    print('판정:', '실패 -- 위 [FAIL] 항목 해결 전 학습 금지' if a.failed
          else '통과')
    return 1 if a.failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
