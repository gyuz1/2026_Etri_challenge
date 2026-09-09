"""VAD perception/planning with the original LAW temporal world-model loss.

The raw input remains the LAW temporal multi-view queue::

    img: [B, T, Ncam, C, H, W]

The current-frame perception/planning path uses the original ``VADHead``:

* VAD BEV encoder and temporal ``prev_bev``;
* VAD agent queries, detection, and six-mode motion prediction;
* VAD map queries and vector-map prediction;
* VAD ego-agent / ego-map interaction and 3 x 6 ego trajectories;
* the original VAD agent/map/trajectory matching and losses.

Only the following are added or changed:

* optional VAD ego LCF input in the ego planning decoder;
* LAW latent world model: previous BEV + selected ego trajectory -> current BEV;
* ego planning is supervised only by waypoint regression when requested.

Past ego trajectory is intentionally NOT used. ``ego_his_encoder`` must remain
``None``. The ON/OFF switch controls only ``ego_lcf_feat_idx``.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import mmcv
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, load_checkpoint
from mmdet.models import DETECTORS
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.VAD.VAD import VAD
from .bev_latent_world_model import BEVLatentWorldModel


@DETECTORS.register_module()
class VADLAW(VAD):
    """Original VAD head trained with the LAW temporal reconstruction loss.

    Args:
        use_ego_lcf_status: Toggle only VAD's low-level ego-status vector in
            the ego trajectory decoder. Agent/map branches do not consume this
            tensor. Past ego trajectory remains disabled in both modes.
        wm_loss_weight: Weight of the BEV latent reconstruction loss.
        remove_auxiliary_planning_losses: Remove VAD's map-boundary,
            collision, and direction planning losses, keeping waypoint L1.
    """

    def __init__(
        self,
        use_ego_lcf_status: bool = False,
        wm_loss_weight: float = 0.2,
        wm_num_layers: int = 2,
        wm_num_heads: int = 8,
        wm_num_points: int = 4,
        wm_ffn_dims: int = 512,
        wm_dropout: float = 0.1,
        wm_use_cumulative_waypoints: bool = False,
        remove_auxiliary_planning_losses: bool = True,
        prev_bev_dropout: float = 0.0,
        echo_cycle_weight: float = 0.0,
        feature_distill_teacher_cfg: Optional[str] = None,
        feature_distill_teacher_ckpt: Optional[str] = None,
        feature_distill_weight: float = 1.0,
        feature_distill_mode: str = 'fused',
        scene_distill_weight: float = 0.0,
        status_distill_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if wm_loss_weight < 0:
            raise ValueError("wm_loss_weight must be non-negative.")
        if not 0.0 <= prev_bev_dropout <= 1.0:
            raise ValueError("prev_bev_dropout must be in [0, 1].")
        if echo_cycle_weight < 0:
            raise ValueError("echo_cycle_weight must be non-negative.")
        if feature_distill_weight < 0:
            raise ValueError("feature_distill_weight must be non-negative.")
        if feature_distill_mode not in ('fused', 'split'):
            raise ValueError(
                "feature_distill_mode must be 'fused' (Scheme B) or "
                f"'split' (Scheme A), got {feature_distill_mode!r}.")
        if scene_distill_weight < 0 or status_distill_weight < 0:
            raise ValueError(
                'scene_distill_weight and status_distill_weight must be '
                'non-negative.')
        if (feature_distill_mode == 'split'
                and scene_distill_weight == 0 and status_distill_weight == 0):
            # Silently-zero losses are exactly the failure mode this project
            # keeps hitting: training runs to completion and the result
            # looks like a plain no-distillation baseline.
            raise ValueError(
                "feature_distill_mode='split' with both split weights at 0 "
                'would train no distillation at all. Set '
                'scene_distill_weight and/or status_distill_weight.')

        self.use_ego_lcf_status = bool(use_ego_lcf_status)
        self.prev_bev_dropout = float(prev_bev_dropout)
        self.echo_cycle_weight = float(echo_cycle_weight)
        self.wm_loss_weight = float(wm_loss_weight)
        self.wm_use_cumulative_waypoints = bool(
            wm_use_cumulative_waypoints
        )
        self.remove_auxiliary_planning_losses = bool(
            remove_auxiliary_planning_losses
        )
        self.feature_distill_weight = float(feature_distill_weight)
        self.feature_distill_mode = feature_distill_mode
        self.scene_distill_weight = float(scene_distill_weight)
        self.status_distill_weight = float(status_distill_weight)

        self._validate_ego_input_configuration()

        # Frozen non-compliant teacher (ego_lcf ON + target-point shortcut),
        # used ONLY to produce a feature-level distillation target for
        # ego_feats -- never runs at inference (gated on self.training in
        # forward_train), and its own gradient is cut at the source (all
        # params requires_grad=False) rather than relying on the loss's
        # detach alone, matching the compliance shape already used for
        # PRISM's posterior and privileged_distill.
        #
        # Wrapped in a plain list (self._teacher_holder = [teacher]) rather
        # than assigned directly, so nn.Module never registers it as a
        # submodule: it is invisible to .state_dict()/checkpoint saving (it
        # would otherwise roughly double every checkpoint's size), to DDP's
        # parameter sync (it has no gradients to sync), and to
        # wrap_fp16_model's module walk. It is a read-only oracle, not part
        # of the model being trained or saved.
        self._teacher_holder: List[Optional[torch.nn.Module]] = [None]
        if feature_distill_teacher_cfg is not None:
            if feature_distill_teacher_ckpt is None:
                raise ValueError(
                    "feature_distill_teacher_ckpt is required when "
                    "feature_distill_teacher_cfg is set."
                )
            teacher_cfg = mmcv.Config.fromfile(feature_distill_teacher_cfg)
            teacher = build_model(
                teacher_cfg.model,
                train_cfg=teacher_cfg.get('train_cfg'),
                test_cfg=teacher_cfg.get('test_cfg'),
            )
            load_checkpoint(
                teacher, feature_distill_teacher_ckpt, map_location='cpu')
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            self._teacher_holder[0] = teacher

        self.bev_world_model = BEVLatentWorldModel(
            embed_dims=self.pts_bbox_head.embed_dims,
            num_waypoints=self.pts_bbox_head.fut_ts,
            bev_h=self.pts_bbox_head.bev_h,
            bev_w=self.pts_bbox_head.bev_w,
            num_layers=wm_num_layers,
            num_heads=wm_num_heads,
            num_points=wm_num_points,
            ffn_dims=wm_ffn_dims,
            dropout=wm_dropout,
        )

    def _validate_ego_input_configuration(self) -> None:
        """Ensure that only the LCF vector is toggled.

        The VAD paper's optional ego-status input corresponds to
        ``ego_lcf_feat``. Past ego trajectory is a separate code option and is
        fixed OFF here so the two experiments differ only by LCF status.

        target_point_shortcut marks a deliberately non-compliant,
        never-submitted diagnostic teacher build (see VAD_head.py's
        constructor comment) -- for that case specifically, this guard's
        whole point (keep the LCF ablation to one variable) doesn't apply,
        since the build is already using every available privileged
        signal on purpose. Every other config still gets the guard.
        """
        if (self.pts_bbox_head.ego_his_encoder is not None
                and not getattr(self.pts_bbox_head,
                                'target_point_shortcut', False)):
            raise ValueError(
                "ego_his_encoder must be None. This implementation toggles "
                "only VAD ego_lcf_feat; past ego trajectory stays disabled. "
                "(Set pts_bbox_head.target_point_shortcut=True to bypass "
                "this for a deliberately non-compliant diagnostic build.)"
            )

        lcf_indices = self.pts_bbox_head.ego_lcf_feat_idx
        head_uses_lcf = lcf_indices is not None

        if self.use_ego_lcf_status != head_uses_lcf:
            raise ValueError(
                "use_ego_lcf_status must match "
                "pts_bbox_head.ego_lcf_feat_idx: use None for OFF and a "
                "non-empty index list for ON."
            )
        if head_uses_lcf and len(lcf_indices) == 0:
            raise ValueError(
                "ego_lcf_feat_idx must be non-empty when LCF status is ON."
            )

    @staticmethod
    def _stack_meta_tensor(
        img_metas: Sequence[Dict],
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        values = []
        for meta in img_metas:
            if key not in meta:
                raise KeyError(
                    f"'{key}' is missing from temporal img_metas. Use "
                    "LAWVADCustomNuScenesDataset so previous-frame waypoint "
                    "targets and optional LCF status are retained."
                )
            value = meta[key]
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            values.append(tensor)
        return torch.stack(values, dim=0).to(device=device)

    @staticmethod
    def _reshape_command(
        ego_fut_cmd: torch.Tensor,
        batch_size: int,
        num_modes: int,
    ) -> torch.Tensor:
        command = ego_fut_cmd.reshape(batch_size, -1)
        if command.shape[1] != num_modes:
            raise ValueError(
                f"Expected {num_modes} command values per sample, got "
                f"{tuple(command.shape)}."
            )
        return command

    def _select_command_trajectory(
        self,
        ego_fut_preds: torch.Tensor,
        ego_fut_cmd: torch.Tensor,
    ) -> torch.Tensor:
        """Select [B, 6, 2] from VAD's [B, 3, 6, 2] ego output."""
        if ego_fut_preds.ndim != 4 or ego_fut_preds.shape[-1] != 2:
            raise ValueError(
                "ego_fut_preds must be [B,M,T,2], got "
                f"{tuple(ego_fut_preds.shape)}."
            )

        batch_size, num_modes = ego_fut_preds.shape[:2]
        command = self._reshape_command(
            ego_fut_cmd,
            batch_size=batch_size,
            num_modes=num_modes,
        )
        mode_index = command.argmax(dim=-1)
        batch_index = torch.arange(batch_size, device=ego_fut_preds.device)
        trajectory = ego_fut_preds[batch_index, mode_index]

        if self.wm_use_cumulative_waypoints:
            trajectory = trajectory.cumsum(dim=-2)
        return trajectory

    def _history_waypoint_loss(
        self,
        ego_fut_preds: torch.Tensor,
        img_metas: Sequence[Dict],
    ) -> torch.Tensor:
        """Apply LAW-style waypoint supervision to one previous frame."""
        device = ego_fut_preds.device
        batch_size = ego_fut_preds.shape[0]

        command = self._stack_meta_tensor(
            img_metas, "ego_fut_cmd", device=device
        )
        prediction = self._select_command_trajectory(ego_fut_preds, command)

        target = self._stack_meta_tensor(
            img_metas, "ego_fut_trajs", device=device
        ).reshape(batch_size, -1, 2)
        mask = self._stack_meta_tensor(
            img_metas, "ego_fut_masks", device=device
        ).reshape(batch_size, -1)

        if target.shape != prediction.shape:
            raise ValueError(
                "History waypoint prediction/target mismatch: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        if mask.shape != prediction.shape[:2]:
            raise ValueError(
                "History waypoint mask mismatch: "
                f"{tuple(mask.shape)} vs {tuple(prediction.shape[:2])}."
            )

        weight = mask[..., None].expand_as(prediction)
        return self.pts_bbox_head.loss_plan_reg(
            prediction,
            target,
            weight,
        )

    def _teacher_plan_hidden(
        self,
        current_image: torch.Tensor,
        current_metas: Sequence[Dict],
        prev_bev: Optional[torch.Tensor],
        ego_lcf_feat: Optional[torch.Tensor],
        ego_target_point: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frozen teacher's planning activations, for feature KD.

        Runs the teacher's own extract_feat + pts_bbox_head end to end --
        it has its own backbone/BEV encoder, diverged from the student's
        during stage2, so the student's intermediate features cannot be
        reused. The real ego_lcf_feat and ego_target_point are passed
        through (the teacher's own config decides whether it uses them);
        the student never sees either.

        Returns the three distillation targets; which of them matter is
        decided by feature_distill_mode:

        - 'fused'  -> ego_plan_hidden, the post-fusion activation. Not
          ego_feats: the teacher's ego_feats keeps ego status in separate
          trailing columns that never reached its agent/map attention, so
          it is the wrong thing to imitate. See VAD_head.py's
          'ego_plan_hidden' comment.
        - 'split'  -> ego_scene_feats and ego_status_feats, the two halves
          of ego_feats aligned by separate losses.
        """
        teacher = self._teacher_holder[0]
        device = current_image.device
        teacher_param = next(teacher.parameters())
        if teacher_param.device != device or teacher_param.dtype != current_image.dtype:
            teacher.to(device=device, dtype=current_image.dtype)

        teacher_feats = teacher.extract_feat(
            img=current_image, img_metas=current_metas,
        )
        teacher_outs = teacher.pts_bbox_head(
            teacher_feats,
            current_metas,
            prev_bev=prev_bev,
            ego_his_trajs=None,
            ego_lcf_feat=ego_lcf_feat,
            ego_target_point=ego_target_point,
        )
        return {
            k: teacher_outs.get(k)
            for k in ('ego_plan_hidden', 'ego_scene_feats',
                      'ego_status_feats')
        }

    def _history_lcf_input(
        self,
        img_metas: Sequence[Dict],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return only optional LCF status; ego history is always disabled."""
        if not self.use_ego_lcf_status:
            return None
        return self._stack_meta_tensor(
            img_metas,
            "ego_lcf_feat",
            device=device,
        )

    def obtain_history_prediction(
        self,
        imgs_queue: torch.Tensor,
        img_metas_list: List[Dict[int, Dict]],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Process LAW history and return both LAW and VAD temporal outputs.

        Returns:
            history_losses: Previous-frame waypoint losses.
            predicted_current_bev: LAW world-model prediction produced from
                the last previous frame.
            temporal_prev_bev: Last observed history BEV, detached exactly at
                the boundary where original VAD supplies history to the
                current frame.

        Unlike original VAD's memory-saving ``only_bev`` history pass, a full
        VAD head pass is needed here to obtain the previous ego trajectory for
        LAW. The BEV passed to the next/current frame is detached, preserving
        the original VAD no-gradient temporal connection.
        """
        batch_size, queue_length, num_cams, channels, height, width = (
            imgs_queue.shape
        )
        flattened = imgs_queue.reshape(
            batch_size * queue_length,
            num_cams,
            channels,
            height,
            width,
        )
        multi_level_feats = self.extract_feat(
            img=flattened,
            len_queue=queue_length,
        )

        losses: Dict[str, torch.Tensor] = {}
        predicted_next_bev: Optional[torch.Tensor] = None
        temporal_prev_bev: Optional[torch.Tensor] = None

        for frame_index in range(queue_length):
            frame_metas = [
                copy.deepcopy(sample_meta[frame_index])
                for sample_meta in img_metas_list
            ]
            frame_feats = [
                level_feat[:, frame_index]
                for level_feat in multi_level_feats
            ]
            frame_lcf = self._history_lcf_input(
                frame_metas,
                device=frame_feats[0].device,
            )

            # Original VADHead query/decoder path. LCF enters only the final
            # ego planning feature; agent/map outputs are generated earlier.
            frame_outs = self.pts_bbox_head(
                frame_feats,
                frame_metas,
                prev_bev=temporal_prev_bev,
                ego_his_trajs=None,
                ego_lcf_feat=frame_lcf,
            )

            losses[f"prev_frame_loss_waypoint_{frame_index}"] = (
                self._history_waypoint_loss(
                    frame_outs["ego_fut_preds"],
                    frame_metas,
                )
            )

            command = self._stack_meta_tensor(
                frame_metas,
                "ego_fut_cmd",
                device=frame_feats[0].device,
            )
            selected_waypoints = self._select_command_trajectory(
                frame_outs["ego_fut_preds"],
                command,
            )
            predicted_next_bev = self.bev_world_model(
                frame_outs["bev_embed"],
                selected_waypoints,
            )

            # Original VAD obtains history BEV under no_grad. Detaching here
            # gives the same temporal gradient boundary while retaining LAW
            # gradients through the world-model branch above.
            temporal_prev_bev = frame_outs["bev_embed"].detach().clone()

        if predicted_next_bev is None or temporal_prev_bev is None:
            raise RuntimeError("No previous frame was processed.")

        return losses, predicted_next_bev, temporal_prev_bev

    @force_fp32(apply_to=("img", "points", "prev_bev"))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        map_gt_bboxes_3d=None,
        map_gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        map_gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_masks=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        ego_target_point=None,
        ego_long_fut_trajs=None,
        ego_long_fut_masks=None,
        ego_long_fut_valid_flag=None,
        gt_attr_labels=None,
    ) -> Dict[str, torch.Tensor]:
        """Use LAW temporal input, original VAD losses, and LAW BEV loss."""
        del (
            points,
            gt_labels,
            gt_bboxes,
            proposals,
            img_depth,
            img_mask,
            ego_his_trajs,  # Explicitly unused in both LCF modes.
            # Per-step validity within the 0-5s PRISM window -- unused since
            # long_fut_valid_flag is all-or-nothing (see converter), so
            # sample-level masking is already exact without this.
            ego_long_fut_masks,
        )

        if img is None or img.ndim != 6:
            raise ValueError(
                "Expected LAW input img [B,T,Ncam,C,H,W], got "
                f"{None if img is None else tuple(img.shape)}."
            )
        if img.shape[1] < 2:
            raise ValueError(
                "LAW world-model training requires at least one previous "
                "frame and one current frame."
            )
        if not isinstance(img_metas, list):
            raise TypeError("img_metas must be a list of temporal metadata maps.")

        if self.use_ego_lcf_status and ego_lcf_feat is None:
            raise ValueError(
                "ego_lcf_feat is required when use_ego_lcf_status=True."
            )

        queue_length = img.shape[1]
        previous_images = img[:, :-1]
        current_image = img[:, -1]

        (
            history_losses,
            predicted_current_bev,
            temporal_prev_bev,
        ) = self.obtain_history_prediction(previous_images, img_metas)

        current_metas = [
            copy.deepcopy(sample_meta[queue_length - 1])
            for sample_meta in img_metas
        ]
        current_feats = self.extract_feat(
            img=current_image,
            img_metas=current_metas,
        )#BEV encoder  

        current_lcf = ego_lcf_feat if self.use_ego_lcf_status else None

        # Train/test consistency: eval and submission both run a single cold
        # frame per clip (eval_holdout_l2.py calls reset_stream() before each
        # window, so prev_bev is None), while training always hands the head
        # a populated temporal_prev_bev from the history queue. A model
        # trained only ever seeing prev_bev learns to lean on it and then
        # loses that input at test time -- harmless while ego_lcf_feat
        # supplied ego motion directly, but the dominant failure mode once
        # it doesn't. Randomly dropping prev_bev during training makes the
        # cold-start path an in-distribution case instead of an unseen one.
        #
        # Captured before the student's own dropout below: the frozen
        # teacher never trains on the cold-start case (its own config uses
        # prev_bev_dropout=0.0, i.e. always sees a real prev_bev), so it
        # should keep getting one for the distillation target regardless of
        # whether this particular iteration drops it for the student.
        # Cloned, not aliased: the student's own pts_bbox_head call below
        # yaw-aligns prev_bev by writing through this tensor
        # (VAD_transformer.py:268), so a plain reference would hand the
        # teacher an already-rotated BEV to rotate a second time.
        teacher_prev_bev = (
            None if temporal_prev_bev is None
            else temporal_prev_bev.clone()
        )
        if self.training and self.prev_bev_dropout > 0.0:
            if float(torch.rand(())) < self.prev_bev_dropout:
                temporal_prev_bev = None

        # Current-frame VAD path, including the official temporal prev_bev.
        current_outs = self.pts_bbox_head(
            current_feats,
            current_metas,
            prev_bev=temporal_prev_bev,
            ego_his_trajs=None,
            ego_lcf_feat=current_lcf,
            # Only meaningful when pts_bbox_head.target_point_shortcut is
            # True (a deliberately non-compliant diagnostic build) -- VAD_
            # head.forward() falls back to a zero goal when this is None,
            # which every compliant config already relies on (it never sets
            # target_point_shortcut, so this is always a no-op for them).
            ego_target_point=ego_target_point,
            ego_long_fut_trajs=ego_long_fut_trajs,
            ego_long_fut_valid_flag=ego_long_fut_valid_flag,
            # Target-only channel for the head's auxiliary ego-motion
            # regression -- deliberately separate from ego_lcf_feat above,
            # which is the (compliance-gated) INPUT path. Passed regardless
            # of use_ego_lcf_status precisely because the point is to
            # supervise features when the input path is off.
            ego_lcf_target=ego_lcf_feat,
            # Also target-side only: selects which mode the auxiliary 5s
            # regression supervises, the same one-hot loss_planning already
            # uses for the scored 3s output. Never reaches ego_feats.
            ego_fut_cmd=ego_fut_cmd,
        )  # Agent, Map, Ego decoder -- temporal_prev_bev is detached here

        # Original VAD agent detection, six-mode agent motion, map prediction,
        # Hungarian matching, decoder auxiliary losses, and ego waypoint loss.
        losses = self.pts_bbox_head.loss(
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            current_outs,
            ego_fut_trajs,
            ego_fut_masks,
            ego_fut_cmd,
            gt_attr_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            map_gt_bboxes_ignore=map_gt_bboxes_ignore,
            img_metas=current_metas,
        )

        # Feature-level distillation from the frozen non-compliant teacher
        # (ego_lcf ON + target-point shortcut). Train-only, gated the same
        # way as PRISM's posterior and privileged_distill: absent from the
        # inference graph, and the teacher's own params are already
        # requires_grad=False, so no gradient reaches ego_lcf/target_point
        # through this path -- only the student's OWN parameters (which
        # never see either) are updated to make ego_feats resemble the
        # teacher's, matching the organizer's "indirect use that improves
        # shared/vision-derived features" allowance already relied on
        # elsewhere in this file.
        if self.training and self._teacher_holder[0] is not None:
            with torch.no_grad():
                # The student's pts_bbox_head call above already consumed
                # these: the BEV encoder yaw-aligns prev_bev by writing
                # THROUGH the caller's tensor (VAD_transformer.py:268,
                # `prev_bev[:, i] = tmp_prev_bev[:, 0]`) and reads/rewrites
                # can_bus deltas inside img_metas. Handing the teacher the
                # same objects would double-rotate the BEV and feed it
                # already-consumed metadata -- silently wrong features, no
                # error. Give it untouched copies instead.
                teacher_out = self._teacher_plan_hidden(
                    current_image=current_image,
                    current_metas=copy.deepcopy(current_metas),
                    prev_bev=teacher_prev_bev,
                    ego_lcf_feat=ego_lcf_feat,
                    ego_target_point=ego_target_point,
                )

            def _cosine_kd(student, teacher, what):
                """Cosine, not raw MSE: two independently-trained networks
                have no reason to share an activation scale, and a smoke
                test measured raw MSE at ~292 vs every other loss in the
                0.01-5 range -- it would dominate the gradient rather than
                steer the representation's direction toward the teacher's.
                """
                if student is None or teacher is None:
                    raise ValueError(
                        f'{what} distillation is enabled but one side did '
                        f'not produce it (student={student is not None}, '
                        f'teacher={teacher is not None}). Check that both '
                        'configs use the ego_his_encoder-free ego_feats '
                        'path and that the teacher sets ego_lcf_embed_dim.')
                if student.shape[-1] != teacher.shape[-1]:
                    raise ValueError(
                        f'{what} widths differ ({tuple(student.shape)} vs '
                        f'{tuple(teacher.shape)}).')
                return 1.0 - F.cosine_similarity(
                    student, teacher.detach(), dim=-1).mean()

            if self.feature_distill_mode == 'split':
                # Scheme A. Two independent alignments over a shared
                # ego_feats layout: the 2*D scene half against the teacher's
                # scene half, and the student's vision-estimated status
                # against the teacher's ego_lcf embedding. Splitting them
                # is what lets the status target stay a target -- in the
                # fused form its contribution is already smeared across
                # every hidden unit and cannot be weighted separately.
                losses['loss_scene_distill'] = (
                    self.scene_distill_weight * _cosine_kd(
                        current_outs['ego_scene_feats'],
                        teacher_out['ego_scene_feats'], 'scene'))
                losses['loss_status_distill'] = (
                    self.status_distill_weight * _cosine_kd(
                        current_outs['ego_status_feats'],
                        teacher_out['ego_status_feats'], 'status'))
            else:
                # Scheme B. Both sides are ego_fut_decoder's first
                # Linear+ReLU output, so they share a width (hidden_dim)
                # even though their INPUTS differ -- the teacher's first
                # Linear is (512+lcf -> 512), the student's is (512 -> 512).
                # No slicing: see VAD_head.py's 'ego_plan_hidden' comment.
                if (teacher_out['ego_plan_hidden'].shape[-1]
                        != current_outs['ego_plan_hidden'].shape[-1]):
                    raise ValueError(
                        'Teacher and student hidden widths differ. Set the '
                        "teacher config's pts_bbox_head."
                        "ego_fut_dec_hidden_dim to the student's (the "
                        'student leaves it unset, so it defaults to its own '
                        'ego_fut_dec_in_dim).')
                losses['loss_feature_distill'] = (
                    self.feature_distill_weight * _cosine_kd(
                        current_outs['ego_plan_hidden'],
                        teacher_out['ego_plan_hidden'], 'plan hidden'))

        # The requested ego branch predicts waypoints only. Agent/map losses
        # are not modified. Can be toggled later to A/B this against always
        # including the auxiliary losses.
        if self.remove_auxiliary_planning_losses:
            for key in (
                "loss_plan_bound",
                "loss_plan_col",
                "loss_plan_dir",
            ):
                losses.pop(key, None)

        observed_current_bev = self.bev_world_model.to_batch_first(
            current_outs["bev_embed"]
        ).detach() #recontruction loss : current bev encoder.detach(pseudo gt)
        loss_rec = F.mse_loss(
            predicted_current_bev,
            observed_current_bev,
        )
        losses["loss_rec"] = self.wm_loss_weight * loss_rec

        # Echo-planning cycle consistency (arXiv:2505.18945), adapted to the
        # world model we already have. loss_rec above is the FORWARD half
        # (prev BEV + planned waypoints -> current BEV); this adds the ECHO
        # half: roll the current BEV forward along the plan to a predicted
        # future BEV, then roll that back with the negated plan and require
        # it to land on the current BEV again.
        #
        # The world model is a generic (bev, waypoints) -> bev map, so the
        # reverse pass reuses the exact same weights with -waypoints rather
        # than adding an inverse module -- weight sharing is what makes the
        # cycle a constraint on the plan instead of two independent
        # predictors that can each learn to ignore it. Train-only: inference
        # never calls bev_world_model at all, so this costs nothing at test
        # time. Gradient reaches the planner through `waypoints`, which is
        # the point -- a plan inconsistent with how the scene actually moves
        # cannot close the cycle.
        if self.echo_cycle_weight > 0.0 and self.training:
            current_bev_bf = self.bev_world_model.to_batch_first(
                current_outs["bev_embed"]
            )
            current_command = self._stack_meta_tensor(
                current_metas, "ego_fut_cmd", device=current_bev_bf.device)
            current_waypoints = self._select_command_trajectory(
                current_outs["ego_fut_preds"], current_command)
            predicted_future_bev = self.bev_world_model(
                current_outs["bev_embed"], current_waypoints)
            echoed_current_bev = self.bev_world_model(
                predicted_future_bev, -current_waypoints)
            losses["loss_echo_cycle"] = self.echo_cycle_weight * F.mse_loss(
                echoed_current_bev,
                current_bev_bf.detach(),
            )

        losses.update(history_losses)
        return losses
    

    @staticmethod
    def _latest_test_meta(meta):
        """Convert LAW temporal metadata to one current-frame metadata dict."""
        if not isinstance(meta, dict):
            return meta

        frame_keys = [
            key for key in meta.keys()
            if isinstance(key, int)
        ]

        # Normal VAD metadata dict:
        # {"scene_token": ..., "can_bus": ..., ...}
        if not frame_keys:
            return meta

        # LAW temporal metadata:
        # {0: previous_meta, 1: current_meta, ...}
        current_index = max(frame_keys)
        return copy.deepcopy(meta[current_index])


    def forward_test(
        self,
        img_metas,
        img=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        ego_target_point=None,
        gt_attr_labels=None,
        bev_only=False,
        **kwargs,
    ):
        """Run VAD inference on the latest LAW temporal frame.

        ``bev_only=True`` runs the history-frame fast path: the only thing
        a non-scored frame contributes is the ``bev_embed`` the NEXT frame's
        temporal fusion consumes, so the detection/map/motion/ego decoders
        are skipped and their (discarded) outputs never computed. Measured
        on a 3090 at fp16: 27.7ms vs 61.9ms for the full pass, i.e. the
        decoders are 55% of a forward. A 2-frame window costs 89.6ms this
        way instead of 123.8ms, which is the difference between clearing
        and missing the 100ms T_infer penalty threshold. The BEV itself is
        bit-identical either way -- the decoders do not feed back into it.

        This method intentionally bypasses ``VAD.forward_test``. The parent
        implementation applies an additional ``img[0]`` augmentation unwrap,
        which can turn [B, Ncam, C, H, W] into [Ncam, C, H, W] when the
        LAW temporal input has already been normalized. That makes six cameras
        look like a batch of six samples.

        Inference does not execute ``bev_world_model``; it uses only the
        current-frame VAD BEV/agent/map/ego branches.
        """

        def unwrap_singleton(value):
            """Remove only singleton augmentation/container wrappers."""
            while (
                isinstance(value, (list, tuple))
                and len(value) == 1
            ):
                value = value[0]
            return value

        def first_augmentation(value):
            """Match the one-augmentation convention used by VAD test."""
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return value
                return value[0]
            return value

        # -------------------------------------------------------------
        # 1. Normalize image to [B, Ncam, C, H, W].
        # -------------------------------------------------------------
        current_img = unwrap_singleton(img)
        if not torch.is_tensor(current_img):
            raise TypeError(
                "Expected test image Tensor after unwrapping, got "
                f"{type(current_img)}."
            )

        if current_img.ndim == 6:
            # LAW queue: [B, T, Ncam, C, H, W] -> latest frame.
            current_img = current_img[:, -1, ...]

        elif current_img.ndim == 4:
            # Defensive recovery for an already-squeezed single sample:
            # [Ncam, C, H, W] -> [1, Ncam, C, H, W].
            current_img = current_img.unsqueeze(0)

        if current_img.ndim != 5:
            raise ValueError(
                "Expected current test image [B,Ncam,C,H,W], got "
                f"{tuple(current_img.shape)}."
            )

        # Official VAD evaluation supports batch size 1.
        if current_img.shape[0] != 1:
            raise ValueError(
                "VAD test currently expects batch size 1, got "
                f"{current_img.shape[0]}."
            )

        # -------------------------------------------------------------
        # 2. Normalize metadata to list[dict] for the current frame.
        # -------------------------------------------------------------
        meta_value = unwrap_singleton(img_metas)

        if isinstance(meta_value, dict):
            current_meta = self._latest_test_meta(meta_value)
            current_metas = [copy.deepcopy(current_meta)]

        elif isinstance(meta_value, (list, tuple)):
            current_metas = [
                copy.deepcopy(self._latest_test_meta(sample_meta))
                for sample_meta in meta_value
            ]

        else:
            raise TypeError(
                "Unexpected test metadata structure after unwrapping: "
                f"{type(meta_value)}."
            )

        if len(current_metas) != current_img.shape[0]:
            raise ValueError(
                "Image/metadata batch mismatch: image batch "
                f"{current_img.shape[0]}, metadata batch {len(current_metas)}."
            )

        num_cameras = current_img.shape[1]
        lidar2img = current_metas[0].get("lidar2img")
        if lidar2img is None:
            raise KeyError("'lidar2img' is missing from current metadata.")
        if len(lidar2img) != num_cameras:
            raise ValueError(
                "Camera/metadata mismatch: image has "
                f"{num_cameras} cameras, lidar2img has {len(lidar2img)}."
            )

        # -------------------------------------------------------------
        # 3. Preserve original VAD temporal prev_bev bookkeeping.
        # -------------------------------------------------------------
        scene_token = current_metas[0]["scene_token"]
        if scene_token != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None

        self.prev_frame_info["scene_token"] = scene_token

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        can_bus = current_metas[0]["can_bus"]
        tmp_pos = copy.deepcopy(can_bus[:3])
        tmp_angle = copy.deepcopy(can_bus[-1])

        if self.prev_frame_info["prev_bev"] is not None:
            current_metas[0]["can_bus"][:3] -= (
                self.prev_frame_info["prev_pos"]
            )
            current_metas[0]["can_bus"][-1] -= (
                self.prev_frame_info["prev_angle"]
            )
        else:
            current_metas[0]["can_bus"][:3] = 0
            current_metas[0]["can_bus"][-1] = 0

        # -------------------------------------------------------------
        # 4. Call simple_test directly. Do not call super().forward_test().
        # -------------------------------------------------------------
        if bev_only:
            # History-frame fast path (see the docstring). Everything above
            # -- image normalization and the can_bus prev_pos/prev_angle
            # bookkeeping -- still runs, because the NEXT frame's BEV
            # encoder consumes those deltas to align this frame's BEV to
            # its own ego pose. Skipping them here would misalign the
            # temporal fusion, which is the whole point of running this
            # frame at all.
            img_feats = self.extract_feat(
                img=current_img, img_metas=current_metas)
            new_prev_bev = self.pts_bbox_head(
                img_feats,
                current_metas,
                self.prev_frame_info["prev_bev"],
                only_bev=True,
            )
            # get_bev_features (the only_bev branch) returns [B, N, D];
            # the full transformer.forward returns [N, B, D] after its own
            # permute. Normalize to the full path's convention so
            # prev_frame_info["prev_bev"] holds exactly one format no
            # matter which path produced it -- a [B, N, D] tensor read as
            # [N, B, D] is shape-valid at B=1 and silently scrambles which
            # channel vector sits at which BEV cell.
            head = self.pts_bbox_head
            if new_prev_bev.shape[1] == head.bev_h * head.bev_w:
                new_prev_bev = new_prev_bev.permute(1, 0, 2)
            bbox_results = [dict()]
        else:
            new_prev_bev, bbox_results = self.simple_test(
                img_metas=current_metas,
                img=current_img,
                prev_bev=self.prev_frame_info["prev_bev"],
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                ego_his_trajs=first_augmentation(ego_his_trajs),
                ego_fut_trajs=first_augmentation(ego_fut_trajs),
                ego_fut_cmd=first_augmentation(ego_fut_cmd),
                ego_lcf_feat=first_augmentation(ego_lcf_feat),
                ego_target_point=first_augmentation(ego_target_point),
                gt_attr_labels=gt_attr_labels,
                **kwargs,
            )

        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        return bbox_results