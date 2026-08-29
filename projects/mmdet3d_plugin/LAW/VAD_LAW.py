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

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS

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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if wm_loss_weight < 0:
            raise ValueError("wm_loss_weight must be non-negative.")

        self.use_ego_lcf_status = bool(use_ego_lcf_status)
        self.wm_loss_weight = float(wm_loss_weight)
        self.wm_use_cumulative_waypoints = bool(
            wm_use_cumulative_waypoints
        )
        self.remove_auxiliary_planning_losses = bool(
            remove_auxiliary_planning_losses
        )

        self._validate_ego_input_configuration()

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
        """
        if self.pts_bbox_head.ego_his_encoder is not None:
            raise ValueError(
                "ego_his_encoder must be None. This implementation toggles "
                "only VAD ego_lcf_feat; past ego trajectory stays disabled."
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

        # Current-frame VAD path, including the official temporal prev_bev.
        current_outs = self.pts_bbox_head(
            current_feats,
            current_metas,
            prev_bev=temporal_prev_bev,
            ego_his_trajs=None,
            ego_lcf_feat=current_lcf,
            ego_long_fut_trajs=ego_long_fut_trajs,
            ego_long_fut_valid_flag=ego_long_fut_valid_flag,
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
        **kwargs,
    ):
        """Run VAD inference on the latest LAW temporal frame.

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