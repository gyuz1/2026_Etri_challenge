#!/usr/bin/env python3
"""Merge ETRI stage-1 perception weights with the LAW world model.

Stage 1 (VAD_etri_tiny_stage1*.py) trains a plain VAD architecture -- it has
no bev_world_model submodule at all, so those weights never move during
stage 1. This script builds the stage-2 (VADLAW) init checkpoint by taking
everything from the stage-1 checkpoint (ETRI-domain-adapted perception,
detection, map, motion, ego decoder) and adding only the bev_world_model.*
keys from the nuScenes LAW checkpoint, which is the only place a trained
world model exists. No key exists in both sources, so this is a plain
union, not a conflict resolution.
"""
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage1', required=True,
                         help='ETRI stage-1 checkpoint (e.g. epoch_48.pth)')
    parser.add_argument('--world-model-source', required=True,
                         help='checkpoint containing bev_world_model.* '
                              '(e.g. ckpts/law_pretrained_nus.pth)')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    stage1_ckpt = torch.load(args.stage1, map_location='cpu')
    wm_ckpt = torch.load(args.world_model_source, map_location='cpu')

    stage1_sd = stage1_ckpt['state_dict']
    wm_sd = wm_ckpt['state_dict']

    wm_keys = [k for k in wm_sd if k.startswith('bev_world_model.')]
    if not wm_keys:
        raise ValueError(
            f'No bev_world_model.* keys found in {args.world_model_source}')

    overlap = set(wm_keys) & set(stage1_sd.keys())
    if overlap:
        raise ValueError(
            f'Unexpected key overlap between stage1 and world model '
            f'source: {sorted(overlap)}')

    merged_sd = dict(stage1_sd)
    for key in wm_keys:
        merged_sd[key] = wm_sd[key]

    merged_ckpt = dict(stage1_ckpt)
    merged_ckpt['state_dict'] = merged_sd
    merged_ckpt['meta'] = dict(
        stage1_ckpt.get('meta', {}),
        merge_source_stage1=args.stage1,
        merge_source_world_model=args.world_model_source,
        merge_world_model_keys=len(wm_keys))
    merged_ckpt.pop('optimizer', None)

    torch.save(merged_ckpt, args.output)
    print(f'Wrote {args.output}')
    print(f'  base keys (stage1):        {len(stage1_sd)}')
    print(f'  added world model keys:    {len(wm_keys)}')
    print(f'  total keys in merged ckpt: {len(merged_sd)}')


if __name__ == '__main__':
    main()
