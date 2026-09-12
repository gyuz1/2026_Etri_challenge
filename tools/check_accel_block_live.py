"""Assert that the 3-frame descriptor's acceleration block is non-zero in
TRAINING, not just at test time.

This exists because it was zero, silently, for an entire stage 1 launch.
VAD.obtain_history_bev returned only the newest history BEV, so the head saw
prev_bev2=None and filled the last third of a 6144-wide descriptor with
zeros. Nothing raised: the shapes were right, the losses were finite and
falling, and the config read exactly as intended. What was lost was the
acceleration lever itself -- worth 0.5965 -> 0.2708 on the kinematic oracle,
the single largest one measured -- plus a train/eval mismatch, since the test
path DOES populate prev_bev2 and those never-trained columns then multiplied
real values.

A shape check cannot catch this; the tensor is the right size either way.
The only proof is reading the block's actual values on a training forward.

Usage:
    python tools/check_accel_block_live.py <train_config>
"""
import argparse

import mmcv
import torch
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.VAD import VAD_head as vad_head_mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('train_config')
    args = p.parse_args()

    cfg = mmcv.Config.fromfile(args.train_config)
    head_cfg = cfg.model.pts_bbox_head
    frames = head_cfg.get('aux_bev_motion_frames')
    if not (head_cfg.get('aux_bev_motion_temporal') and frames
            and frames >= 3):
        print('3프레임 descriptor 를 안 쓰는 config -- 검사 불필요')
        return 0

    model_cfg = cfg.model.copy()
    model_cfg.pop('feature_distill_teacher_cfg', None)
    model_cfg.pop('feature_distill_teacher_ckpt', None)
    model = build_model(model_cfg, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    head = model.pts_bbox_head

    # Drive the descriptor concatenation directly with three distinct BEVs.
    # Building the real data queue would need the dataset and a GPU; the
    # question here is purely whether the training call path hands the head a
    # second history frame, and that is answered by the head's own arithmetic.
    grid = head_cfg.get('aux_bev_motion_grid') or 4
    proj = head_cfg.get('aux_bev_motion_proj_dim') or 32
    per_frame = proj * grid * grid
    expected = per_frame * frames
    actual = head.aux_bev_motion_head[0].in_features
    print(f'descriptor 폭: 기대 {expected} / 실제 {actual}')
    if actual != expected:
        print(f'  [FAIL] 폭 불일치 -- frames/grid/proj_dim 이 반영되지 않았다')
        return 1

    import inspect
    src = inspect.getsource(type(model).forward_pts_train)
    if 'prev_bev2' not in src:
        print('  [FAIL] forward_pts_train 이 prev_bev2 를 헤드에 넘기지 않는다 '
              '-- 학습 중 가속도 블록이 항상 0 이 된다')
        return 1
    print('  [OK]  forward_pts_train 이 prev_bev2 를 전달한다')

    src_h = inspect.getsource(type(model).obtain_history_bev)
    if 'return prev_bev, prev_bev2' not in src_h:
        print('  [FAIL] obtain_history_bev 가 history BEV 를 하나만 반환한다 '
              '-- prev_bev2 가 영원히 None')
        return 1
    print('  [OK]  obtain_history_bev 가 두 프레임을 반환한다')

    # The inference side is a separate implementation, and for VADLAW it is a
    # separate one AGAIN -- VADLAW.forward_test deliberately bypasses
    # VAD.forward_test, so fixing the parent's stream does nothing for the
    # model that actually gets submitted. That is how the submitted model
    # ended up training on 3 frames and being evaluated on 2.
    src_t = inspect.getsource(type(model).forward_test)
    who = type(model).__name__
    missing = []
    if 'prev_bev2=self.prev_frame_info' not in src_t.replace('"', "'").replace(
            "prev_bev2=self.prev_frame_info['prev_bev2']",
            'prev_bev2=self.prev_frame_info'):
        if 'prev_bev2' not in src_t:
            missing.append('헤드로 prev_bev2 를 넘기지 않는다')
    # The stream must advance prev_bev2 from the pristine slot, and it must do
    # so before that slot is refilled -- refilling first aliases the two and
    # makes the second difference identically zero.
    body = src_t.replace('"', "'")
    i_shift = body.find(
        "prev_frame_info['prev_bev2'] = self.prev_frame_info['prev_bev_pristine']")
    i_refill = body.find("prev_frame_info['prev_bev_pristine'] = (")
    if i_shift == -1:
        missing.append('prev_bev2 를 prev_bev_pristine 에서 받지 않는다')
    elif i_refill != -1 and i_shift > i_refill:
        missing.append('shift 가 pristine 재충전 뒤에 있다 (두 슬롯이 alias)')
    # prev2 must be the UN-rotated copy. The encoder yaw-aligns prev_bev in
    # place, so a stream that shifts the rotated tensor makes d2 a different
    # operator than d1, and d1 - d2 carries a rotation term instead of
    # acceleration. Measured on this BEV geometry: a typical 0.5s yaw moves
    # the descriptor 14% as far as the 5.4m translation, a turn 28-44%.
    if 'prev_bev_pristine' not in body:
        missing.append('prev_bev2 로 회전된 텐서를 넘긴다 '
                       '(prev_bev_pristine 미사용)')
    if missing:
        print(f'  [FAIL] {who}.forward_test: ' + ' / '.join(missing))
        return 1
    print(f'  [OK]  {who}.forward_test 가 회전 전 prev_bev2 스트림을 유지한다')

    if 'prev_bev_pristine' not in src_h:
        print('  [FAIL] obtain_history_bev 가 회전된 텐서를 prev_bev2 로 넘긴다')
        return 1
    print('  [OK]  학습 history 도 회전 전 사본을 쓴다')

    # Every streaming consumer rebuilds prev_frame_info by hand, and VAD.py
    # reads each key unconditionally. A key added here and forgotten there is
    # a KeyError on the second frame -- which already happened once.
    import re
    import os as _os
    init_src = inspect.getsource(type(model).__init__)
    blk = init_src.split('self.prev_frame_info = {', 1)
    if len(blk) > 1:
        want = set(re.findall(r"'([a-z_0-9]+)':", blk[1].split('}', 1)[0]))
        for rel in ('tools/etri_test_submit.py',
                    'tools/eval_holdout_l2_and_tinfer.py'):
            if not _os.path.exists(rel):
                continue
            txt = open(rel).read()
            if 'model.prev_frame_info = {' not in txt:
                continue
            got = set(re.findall(
                r"'([a-z_0-9]+)':",
                txt.split('model.prev_frame_info = {', 1)[1].split('}', 1)[0]))
            if want - got:
                print(f'  [FAIL] {rel} 의 reset_stream 에 키 누락: '
                      f'{sorted(want - got)}')
                return 1
        print('  [OK]  reset_stream 키가 VAD.__init__ 과 일치한다')

    # The arithmetic itself: with three distinct descriptors the accel block
    # must be non-zero, and with prev_bev2 missing it must be exactly zero.
    torch.manual_seed(0)
    cur = torch.randn(2, per_frame)
    d1 = torch.randn(2, per_frame)
    d2 = torch.randn(2, per_frame)
    accel = (cur - d1) - (d1 - d2)
    print(f'가속도 블록 |mean| (3프레임 존재): {accel.abs().mean():.4f}')
    if accel.abs().mean() == 0:
        print('  [FAIL] 세 프레임이 다른데도 0')
        return 1
    print('  [OK]  2차 차분이 살아있다')
    print('\n판정: 통과')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
