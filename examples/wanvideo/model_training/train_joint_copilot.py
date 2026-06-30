"""Joint training: Wan2.1 T2V DiT SFT (mistake-forcing) + VideoCopilot v2.

Goal: do everything that the two-stage pipeline does
(``Wan2.1-T2V-1.3B-OmniWorld-MistakeForcing.sh`` -> dumped tensors ->
``video_copilot/run_train.sh``) in a single process, *without* the on-disk dump:

  * Walk the full OmniWorld JSONL manifest (no ``--max_data_items 100`` cap).
  * Run DiT SFT exactly like ``train.py`` task=``sft:mistake_forcing``.
  * Instead of ``WanMistakeRecorder.write()`` writing ``.pt`` files, the same
    payload (``clean_prediction``, ``velocity_residual``, ``time_embedding``,
    ``condition_embedding``, ``hidden_mean``) is detached and fed straight into
    a ``VideoCopilotDecoderV2`` for a parallel MSE objective.
  * Copilot prediction is purely a side head: detached inputs guarantee
    gradients cannot leak back into the DiT.

Differences vs. the upstream loop are intentionally minimal — same dataset
operator, same Wan pipeline construction, same ZeRO-3 accelerate config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import accelerate
import torch
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion import ModelLogger  # noqa: F401  re-exported
from diffsynth.diffusion.runner import (
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
)

try:
    from .mistake_forcing import (
        OmniWorldFrameSequenceOperator,
        OmniWorldManifestDataset,
        WanHiddenStateCapture,
    )
    from .wisa_dataset import (
        WISAManifestDataset,
        WISAVideoOperator,
        build_video_index,
    )
    from .train import WanTrainingModule, wan_parser
except ImportError:
    from mistake_forcing import (  # type: ignore
        OmniWorldFrameSequenceOperator,
        OmniWorldManifestDataset,
        WanHiddenStateCapture,
    )
    from wisa_dataset import (  # type: ignore
        WISAManifestDataset,
        WISAVideoOperator,
        build_video_index,
    )
    from train import WanTrainingModule, wan_parser  # type: ignore

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_COPILOT_DIR = "/mnt/workspace/hwzhang/code/mistake_forcing/video_copilot"


class InMemoryCopilotTrainer:
    """Drop-in replacement for ``WanMistakeRecorder``.

    The mistake-forcing loss calls ``write(payload, metadata)`` after every
    DiT forward. The original recorder serialises ``payload`` to ``.pt``;
    this one instead:

      1. ``detach()`` every tensor in the payload (so the copilot graph is
         disjoint from the DiT graph — no gradient feedback into the DiT).
      2. Runs the copilot model (v2 or v3) on the detached inputs.
      3. Stores ``self.last_loss`` so the outer training module can sum it
         into the total loss for a single ``accelerator.backward`` pass.
    """

    def __init__(
        self,
        copilot_module: torch.nn.Module,
        copilot_version: str = "v2",
        gap_sampler=None,
        fuse_copilot_into_loss: bool = False,
        copilot_fuse_scale: float = 1.0,
        grad_to_dit: bool = False,
    ):
        self.copilot = copilot_module
        self.copilot_version = copilot_version
        self.last_loss: torch.Tensor | None = None
        # Raw copilot prediction (grad -> copilot params; DiT-detached inputs).
        # Consumed by FlowMatchSFTMistakeForcingLoss when fusing into the DiT loss.
        self.last_copilot_output: torch.Tensor | None = None
        # DiT-only MSE kept for logging when fused mode replaces the return value.
        self.last_dit_only_loss: torch.Tensor | None = None
        self.gap_sampler = gap_sampler
        # Read by the loss fn via getattr() to switch on the fused objective.
        self.fuse_copilot_into_loss = bool(fuse_copilot_into_loss)
        self.copilot_fuse_scale = float(copilot_fuse_scale)
        # Auxiliary-head mode: when True, hidden states + time embedding are NOT
        # detached in write(), so the copilot MSE backprops into the DiT. Must be
        # paired with WanHiddenStateCapture(detach=False) upstream.
        self.grad_to_dit = bool(grad_to_dit)

    def reset(self) -> None:
        self.last_loss = None
        self.last_copilot_output = None
        self.last_dit_only_loss = None

    def write(self, payload: dict, metadata: dict):
        # clean_prediction is a final-output derivative; keep it detached so the
        # auxiliary-head gradient enters the DiT through mid-layer hidden states,
        # not via a second output-side path.
        clean_prediction = payload["clean_prediction"].detach()
        # Target stays fixed (detached) even in auxiliary-head mode.
        velocity_residual = payload["velocity_residual"].detach()
        # Frozen T5 context: detach either way.
        condition_embedding = payload["condition_embedding"].detach()
        # hidden states + time embedding carry the copilot->DiT gradient when
        # grad_to_dit is on. Their capture was left non-detached upstream
        # (WanHiddenStateCapture.detach=False), so NOT detaching here lets the
        # copilot MSE backprop into the DiT backbone (deep supervision). When off,
        # detach here to reproduce the original detached side-head behaviour.
        keep = self.grad_to_dit
        time_embedding = (
            payload["time_embedding"] if keep else payload["time_embedding"].detach()
        )

        if self.copilot_version == "v3":
            # selected_hidden_states from capture: (L, B, N, D) → v3 expects (B, L, N, D)
            shs = (
                payload["selected_hidden_states"]
                if keep
                else payload["selected_hidden_states"].detach()
            )
            if shs.dim() == 4:
                # Always transpose by contract; comparing axis sizes fails when
                # batch size happens to equal the number of selected layers.
                shs = shs.permute(1, 0, 2, 3).contiguous()
            pred = self.copilot(
                clean_prediction, time_embedding, condition_embedding, shs
            )
        else:
            hidden_input = (
                payload["hidden_mean"] if keep else payload["hidden_mean"].detach()
            )
            pred = self.copilot(
                clean_prediction, time_embedding, condition_embedding, hidden_input
            )

        self.last_copilot_output = pred
        self.last_loss = torch.nn.functional.mse_loss(
            pred.float(), velocity_residual.float()
        )

        if self.gap_sampler is not None:
            self.gap_sampler.update(
                timestep_id=metadata.get("timestep_id", 0),
                residual_norm_sq=metadata.get("residual_norm_sq", 0.0),
                copilot_loss=float(self.last_loss.detach().cpu().item()),
            )

        return None


class JointWanCopilotModule(WanTrainingModule):
    """Wan SFT module + a sibling VideoCopilot v2 head, trained jointly."""

    def __init__(
        self,
        *args,
        copilot_dir: str = DEFAULT_COPILOT_DIR,
        copilot_version: str = "v2",
        copilot_dim: int = 1024,
        copilot_depth: int = 10,
        copilot_num_heads: int = 16,
        copilot_mlp_ratio: float = 4.0,
        copilot_lr: float = 1e-4,
        copilot_loss_weight: float = 1.0,
        fuse_copilot_into_loss: bool = False,
        copilot_fuse_scale: float = 1.0,
        copilot_grad_to_dit: bool = False,
        copilot_use_gradient_checkpointing: bool = False,
        copilot_resume: str | None = None,
        copilot_n_selected_layers: int = 4,
        copilot_selected_hidden_dim: int = 1536,
        optimal_gap_sampling: bool = False,
        ogs_num_bins: int = 50,
        ogs_ema_decay: float = 0.99,
        ogs_warmup_steps: int = 200,
        ogs_alpha: float = 1.0,
        ogs_beta: float = 1.0,
        ogs_floor_ema_decay: float = 0.999,
        **kwargs,
    ):
        kwargs.setdefault("task", "sft:mistake_forcing")
        if kwargs.get("task") != "sft:mistake_forcing":
            raise ValueError(
                "JointWanCopilotModule requires task='sft:mistake_forcing'; "
                f"got {kwargs.get('task')!r}."
            )

        super().__init__(*args, **kwargs)

        if copilot_dir not in sys.path:
            sys.path.insert(0, copilot_dir)

        self._copilot_version = copilot_version

        if copilot_version == "v3":
            from model_v3 import VideoCopilotDecoderV3, count_params  # type: ignore
            copilot = VideoCopilotDecoderV3(
                dim=copilot_dim,
                depth=copilot_depth,
                num_heads=copilot_num_heads,
                mlp_ratio=copilot_mlp_ratio,
                n_selected_layers=copilot_n_selected_layers,
                selected_hidden_dim=copilot_selected_hidden_dim,
                zero_init_output=False,
            )
        else:
            from model_v2 import VideoCopilotDecoderV2, count_params  # type: ignore
            copilot = VideoCopilotDecoderV2(
                dim=copilot_dim,
                depth=copilot_depth,
                num_heads=copilot_num_heads,
                mlp_ratio=copilot_mlp_ratio,
                zero_init_output=False,
            )

        if copilot_use_gradient_checkpointing:
            copilot.enable_gradient_checkpointing()
        if copilot_resume is not None:
            sd = torch.load(copilot_resume, map_location="cpu")
            copilot.load_state_dict(sd, strict=True)
            print(f"[JointCopilot] loaded copilot {copilot_version} weights from {copilot_resume}")

        self.copilot = copilot
        self.copilot_lr = float(copilot_lr)
        self.copilot_loss_weight = float(copilot_loss_weight)
        self._fuse_copilot_into_loss = bool(fuse_copilot_into_loss)
        self._copilot_fuse_scale = float(copilot_fuse_scale)
        self._copilot_grad_to_dit = bool(copilot_grad_to_dit)
        if self._copilot_grad_to_dit and self._fuse_copilot_into_loss:
            raise ValueError(
                "copilot_grad_to_dit and fuse_copilot_into_dit_loss are mutually "
                "exclusive: both route copilot influence into the DiT but with "
                "different (conflicting) gradient semantics. Enable at most one."
            )
        # Controls WanHiddenStateCapture(detach=...) in BOTH the single-forward
        # (train.py:get_pipeline_inputs) and batched-forward (_forward_batched)
        # paths. Off => detached side head; on => copilot MSE backprops into the
        # DiT through the captured mid-layer hidden states (deep supervision).
        self._capture_detach_hidden = not self._copilot_grad_to_dit
        self._copilot_param_count = count_params(copilot)

        self._gap_sampler = None
        if optimal_gap_sampling:
            from diffsynth.diffusion.optimal_gap_sampler import OptimalGapTimestepSampler
            self._gap_sampler = OptimalGapTimestepSampler(
                num_total_timesteps=1000,
                num_bins=ogs_num_bins,
                ema_decay=ogs_ema_decay,
                warmup_steps=ogs_warmup_steps,
                alpha=ogs_alpha,
                beta=ogs_beta,
                floor_ema_decay=ogs_floor_ema_decay,
                min_timestep_id=int(kwargs.get("min_timestep_boundary", 0) * 1000),
                max_timestep_id=int(kwargs.get("max_timestep_boundary", 1) * 1000),
            )

        self._copilot_trainer = InMemoryCopilotTrainer(
            self.copilot,
            copilot_version=copilot_version,
            gap_sampler=self._gap_sampler,
            fuse_copilot_into_loss=self._fuse_copilot_into_loss,
            copilot_fuse_scale=self._copilot_fuse_scale,
            grad_to_dit=self._copilot_grad_to_dit,
        )
        self.last_dit_loss: torch.Tensor | None = None
        self.last_copilot_loss: torch.Tensor | None = None

    # -- intercept the mistake_recorder factory --------------------------------
    def get_or_create_mistake_recorder(self):
        return self._copilot_trainer

    # -- inject presampled timestep_id when OGS is active -----------------------
    def get_pipeline_inputs(self, data):
        inputs_shared, inputs_posi, inputs_nega = super().get_pipeline_inputs(data)
        if self._gap_sampler is not None:
            sampled_id = self._gap_sampler.sample_timestep_id()
            inputs_shared["presampled_timestep_id"] = torch.tensor([sampled_id], dtype=torch.long)
        return inputs_shared, inputs_posi, inputs_nega

    # -- separate LRs for DiT and copilot via param groups ---------------------
    def trainable_modules(self):
        dit_params = [p for p in self.pipe.dit.parameters() if p.requires_grad]
        copilot_params = [p for p in self.copilot.parameters() if p.requires_grad]
        return [
            {"params": dit_params},  # default lr from optimizer ctor
            {"params": copilot_params, "lr": self.copilot_lr},
        ]

    # -- keep ModelLogger's "DiT-only" save semantics --------------------------
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        # Drop copilot.* keys; they are checkpointed separately by the launcher.
        state_dict = {
            name: param
            for name, param in state_dict.items()
            if not name.startswith("copilot.")
        }
        return super().export_trainable_state_dict(
            state_dict, remove_prefix=remove_prefix
        )

    # -- joint loss ------------------------------------------------------------
    def forward(self, data, inputs=None):
        # collate_fn hands the raw list of samples through, so batch_size > 1
        # arrives as a list[dict] and batch_size == 1 (or a cached-inputs step)
        # as a single dict. Dispatch on that.
        if isinstance(data, list):
            if len(data) == 1:
                return self._forward_single(data[0], inputs=inputs)
            return self._forward_batched(data)
        return self._forward_single(data, inputs=inputs)

    def _forward_single(self, data, inputs=None):
        self._copilot_trainer.reset()
        loss = super().forward(data, inputs=inputs)
        return self._combine_losses(loss)

    def _forward_batched(self, batch_list):
        """Real batched DiT forward without touching DiffSynth core.

        ``pipe.units`` are preprocessing only (the DiT lives in the loss's
        ``model_fn``), so VAE/T5 encoding is run per sample — reusing the proven
        bs=1 path — and the resulting latents/context are stacked into [B, ...]
        and pushed through ONE FlowMatchSFTMistakeForcingLoss call. That single
        call is where the DiT (and the copilot head) actually run a batched
        forward, which is what lifts single-GPU utilisation.
        """
        self._copilot_trainer.reset()
        pipe = self.pipe

        input_latents_list, context_list, prompt_list, sample_ids = [], [], [], []
        template = None
        for sample in batch_list:
            inputs = self.get_pipeline_inputs(sample)
            inputs = self.transfer_data_to_device(inputs, pipe.device, pipe.torch_dtype)
            for unit in pipe.units:
                inputs = pipe.unit_runner(unit, pipe, *inputs)
            shared, posi, nega = inputs
            if "input_latents" not in shared:
                raise RuntimeError(
                    "Batched joint training needs the VAE to produce "
                    "input_latents (got none). batch_size > 1 does not support "
                    "cached datasets or image-only inputs."
                )
            input_latents_list.append(shared["input_latents"])
            context_list.append(posi["context"])
            if "prompt" in posi:
                prompt_list.append(posi["prompt"])
            sample_ids.append(sample.get("sample_id"))
            if template is None:
                template = (shared, posi, nega)

        shared, posi, nega = template
        shared, posi = dict(shared), dict(posi)

        # Frame count is constant (fixed_frames operator) so latents share T and
        # context shares the tokenizer's fixed seq_len → a plain cat on dim 0.
        shared["input_latents"] = torch.cat(input_latents_list, dim=0)
        # Drop the stale bs=1 noise; the loss regenerates randn_like(input_latents).
        shared.pop("latents", None)
        posi["context"] = torch.cat(context_list, dim=0)
        if prompt_list:
            posi["prompt"] = prompt_list

        # One capture for the whole batch: the single batched DiT forward
        # accumulates [B, N, D] hidden states, so finalize() yields batched
        # payloads the copilot head consumes in one shot.
        shared["mistake_capture"] = WanHiddenStateCapture(
            self.mistake_selected_layers, detach=self._capture_detach_hidden
        )
        if "mistake_metadata" in shared:
            meta = dict(shared["mistake_metadata"])
            meta["sample_id"] = sample_ids
            shared["mistake_metadata"] = meta

        loss = self.task_to_loss[self.task](pipe, shared, posi, nega)
        return self._combine_losses(loss)

    def _combine_losses(self, loss):
        copilot_loss = self._copilot_trainer.last_loss
        if self._fuse_copilot_into_loss:
            # Fused mode: the loss fn already returned
            #   MSE(noise_pred + scale * copilot_out, training_target),
            # updating BOTH the base DiT (via noise_pred) and the copilot (via
            # copilot_out). The standalone copilot residual MSE is kept on top so
            # the copilot keeps its direct supervision; its target
            # (velocity_residual) is detached, so that term adds gradient to the
            # copilot only. last_dit_loss logs the DiT-only term the loss stashed.
            dit_only = self._copilot_trainer.last_dit_only_loss
            self.last_dit_loss = dit_only if torch.is_tensor(dit_only) else None
        else:
            self.last_dit_loss = loss.detach() if torch.is_tensor(loss) else None
        self.last_copilot_loss = (
            copilot_loss.detach() if torch.is_tensor(copilot_loss) else None
        )
        if copilot_loss is None:
            return loss
        return loss + self.copilot_loss_weight * copilot_loss


def save_copilot_checkpoint(accelerator, model, output_path, file_name):
    """Pull a full bf16 copilot state dict from the (possibly ZeRO-3) engine
    and persist it on the main process only. The DiT half is saved by
    ``ModelLogger.save_model`` separately."""
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        copilot_sd = {
            name[len("copilot."):]: param
            for name, param in state_dict.items()
            if name.startswith("copilot.")
        }
        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, file_name)
        torch.save(copilot_sd, path)
        print(f"[JointCopilot] saved copilot weights -> {path}")


def launch_joint_training(accelerator, dataset, model, model_logger, args):
    """Mirrors ``launch_training_task`` but logs DiT/copilot loss separately
    and checkpoints the copilot head alongside the DiT."""

    optimizer_class = get_optimizer_class(args.customized_optimizer)
    optimizer = optimizer_class(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # Hand the raw list of samples to model.forward, which dispatches to the
        # single- or batched-forward path. The default collate would try to
        # stack PIL frame lists / prompt strings and fail.
        collate_fn=lambda samples: samples,
        num_workers=args.dataset_num_workers,
    )

    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    initialize_deepspeed_gradient_checkpointing(accelerator)

    if accelerator.is_main_process:
        os.makedirs(args.output_path, exist_ok=True)
    copilot_loss_path = os.path.join(args.output_path, "copilot_loss.jsonl")
    copilot_loss_fp = (
        open(copilot_loss_path, "a", buffering=1)
        if accelerator.is_main_process
        else None
    )

    last_logged_step = -1  # so we only print once per optimizer step

    try:
        for epoch_id in range(args.num_epochs):
            pbar = tqdm(
                dataloader,
                disable=not accelerator.is_main_process,
                desc=f"epoch {epoch_id}",
            )
            for local_step, data in enumerate(pbar, start=1):
                with accelerator.accumulate(model):
                    runtime_model = accelerator.unwrap_model(model)
                    if hasattr(runtime_model, "set_runtime_state"):
                        runtime_model.set_runtime_state(
                            epoch=epoch_id,
                            local_step=local_step,
                            global_step=model_logger.num_steps + 1,
                            rank=accelerator.process_index,
                        )
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data[0])
                    else:
                        loss = model(data)
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    model_logger.on_step_end(
                        accelerator, model, args.save_steps, loss=loss
                    )

                    if accelerator.sync_gradients:
                        dit_loss = runtime_model.last_dit_loss
                        copilot_loss = runtime_model.last_copilot_loss
                        total_val = float(loss.detach().float().item())
                        dit_val = (
                            float(dit_loss.float().item())
                            if dit_loss is not None
                            else float("nan")
                        )
                        cop_val = (
                            float(copilot_loss.float().item())
                            if copilot_loss is not None
                            else float("nan")
                        )
                        current_step = model_logger.num_steps

                        if accelerator.is_main_process:
                            if copilot_loss_fp is not None:
                                copilot_loss_fp.write(
                                    json.dumps(
                                        {
                                            "global_step": current_step,
                                            "epoch": epoch_id,
                                            "local_step": local_step,
                                            "total_loss": total_val,
                                            "dit_loss": (
                                                None
                                                if dit_loss is None
                                                else dit_val
                                            ),
                                            "copilot_loss": (
                                                None
                                                if copilot_loss is None
                                                else cop_val
                                            ),
                                        }
                                    )
                                    + "\n"
                                )
                            # Push the per-component losses to every enabled
                            # logger (wandb / tensorboard / swanlab share the
                            # same .log(key, value, step) API). ModelLogger
                            # already logged the total as "loss"; here we add the
                            # breakdown the joint/fused objective cares about.
                            for _logger in model_logger.loggers:
                                _logger.log("total_loss", total_val, current_step)
                                if dit_loss is not None:
                                    _logger.log("dit_loss", dit_val, current_step)
                                if copilot_loss is not None:
                                    _logger.log("copilot_loss", cop_val, current_step)
                                if runtime_model._gap_sampler is not None:
                                    _logger.log(
                                        "ogs_entropy",
                                        runtime_model._gap_sampler.distribution_entropy(),
                                        current_step,
                                    )
                            pbar.set_postfix(
                                total=f"{total_val:.4f}",
                                dit=f"{dit_val:.4f}",
                                copilot=f"{cop_val:.4f}",
                            )
                            if (
                                args.log_every > 0
                                and current_step != last_logged_step
                                and current_step % args.log_every == 0
                            ):
                                ogs_info = ""
                                if runtime_model._gap_sampler is not None:
                                    gs = runtime_model._gap_sampler
                                    ogs_info = (
                                        f" ogs_warmed={gs.is_warmed_up}"
                                        f" ogs_entropy={gs.distribution_entropy():.3f}"
                                        f" ogs_bins={gs.num_active_bins()}/{gs.num_bins}"
                                    )
                                print(
                                    f"[joint] step {current_step} "
                                    f"epoch {epoch_id} local {local_step} | "
                                    f"total={total_val:.5f} "
                                    f"dit={dit_val:.5f} "
                                    f"copilot={cop_val:.5f}"
                                    f"{ogs_info}",
                                    flush=True,
                                )
                                last_logged_step = current_step

                        # Mirror ModelLogger's step-saved DiT cadence with a
                        # per-step copilot checkpoint, so both halves stay in
                        # sync when --save_steps is set (e.g. every 200 steps).
                        if (
                            args.save_steps is not None
                            and current_step > 0
                            and current_step % args.save_steps == 0
                        ):
                            save_copilot_checkpoint(
                                accelerator,
                                model,
                                args.output_path,
                                f"copilot_step-{current_step}.pt",
                            )

            if args.save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)
                if (
                    args.save_copilot_every_epoch > 0
                    and (epoch_id + 1) % args.save_copilot_every_epoch == 0
                ):
                    save_copilot_checkpoint(
                        accelerator,
                        model,
                        args.output_path,
                        f"copilot_epoch-{epoch_id}.pt",
                    )

        model_logger.on_training_end(accelerator, model, args.save_steps)
        save_copilot_checkpoint(
            accelerator, model, args.output_path, "copilot_final.pt"
        )
    finally:
        if copilot_loss_fp is not None:
            copilot_loss_fp.close()


def joint_parser() -> argparse.ArgumentParser:
    parser = wan_parser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU training batch size. >1 runs a real batched DiT forward "
        "(per-sample VAE/T5 outputs are stacked). Needs a constant frame count, "
        "so the WISA operator switches to fixed-frame sampling automatically "
        "when batch_size > 1. Effective batch = batch_size * "
        "gradient_accumulation_steps * num_processes.",
    )
    parser.add_argument(
        "--copilot_dir",
        type=str,
        default=DEFAULT_COPILOT_DIR,
        help="Path to the video_copilot directory (must contain model_v2.py / model_v3.py).",
    )
    parser.add_argument(
        "--copilot_version",
        type=str,
        default="v2",
        choices=["v2", "v3"],
        help="Copilot model version: v2 (hidden_mean) or v3 (selected_hidden_states fusion).",
    )
    parser.add_argument("--copilot_dim", type=int, default=1024)
    parser.add_argument("--copilot_depth", type=int, default=10)
    parser.add_argument("--copilot_num_heads", type=int, default=16)
    parser.add_argument("--copilot_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--copilot_lr", type=float, default=1e-4)
    parser.add_argument(
        "--copilot_loss_weight",
        type=float,
        default=1.0,
        help="Weight on the copilot MSE in the total loss. Copilot inputs are "
        "always detached, so this only scales copilot-side gradients.",
    )
    parser.add_argument(
        "--fuse_copilot_into_dit_loss",
        action="store_true",
        default=False,
        help="New mode (default off): add a fused term "
        "MSE(noise_pred + copilot_fuse_scale * copilot_out, training_target) that "
        "supervises the SUM of base DiT output and copilot output, on TOP of the "
        "original standalone copilot residual MSE (still weighted by "
        "--copilot_loss_weight). Total = fused_term + copilot_loss_weight * "
        "copilot_residual_MSE. The fused term updates BOTH the base DiT (via "
        "noise_pred) and the copilot (via copilot_out); the residual term adds "
        "extra gradient to the copilot only. Copilot inputs stay detached, so "
        "copilot gradients never leak into the DiT (the DiT is updated only via "
        "the direct residual connection), mirroring the inference-time fusion. "
        "When off, the original loss (separate DiT MSE + detached copilot MSE) "
        "is used.",
    )
    parser.add_argument(
        "--copilot_fuse_scale",
        type=float,
        default=1.0,
        help="Scale on the copilot residual when fusing into the DiT loss "
        "(analogous to inference --copilot_scale). Only used with "
        "--fuse_copilot_into_dit_loss.",
    )
    parser.add_argument(
        "--copilot_grad_to_dit",
        action="store_true",
        default=False,
        help="Auxiliary-head mode (default off): let the copilot MSE gradient "
        "flow back into the DiT through the captured mid-layer hidden states "
        "(hidden_mean for v2 / selected_hidden_states for v3) and the time "
        "embedding, turning the copilot into a deep-supervision head. The target "
        "(velocity_residual), clean_prediction and the frozen T5 context stay "
        "detached. Mutually exclusive with --fuse_copilot_into_dit_loss. Off => "
        "copilot stays a detached side head (no gradient reaches the DiT).",
    )
    parser.add_argument(
        "--copilot_use_gradient_checkpointing", action="store_true"
    )
    parser.add_argument("--copilot_resume", type=str, default=None)
    parser.add_argument(
        "--copilot_n_selected_layers",
        type=int,
        default=4,
        help="(v3 only) Number of selected DiT layers for hidden-state fusion.",
    )
    parser.add_argument(
        "--copilot_selected_hidden_dim",
        type=int,
        default=1536,
        help="(v3 only) Per-layer hidden dimension of selected DiT states.",
    )
    parser.add_argument(
        "--save_copilot_every_epoch",
        type=int,
        default=1,
        help="0 disables periodic copilot checkpoints (final is still written).",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Print per-step (total/dit/copilot) loss every N optimizer steps. "
             "0 disables stdout logging (jsonl still written).",
    )
    # --- Optimal Gap Sampling (OGS) ---
    parser.add_argument(
        "--optimal_gap_sampling",
        action="store_true",
        default=False,
        help="Enable non-uniform timestep sampling based on optimal-gap theory.",
    )
    parser.add_argument("--ogs_num_bins", type=int, default=50,
                        help="Number of bins for discretizing the timestep space.")
    parser.add_argument("--ogs_ema_decay", type=float, default=0.99,
                        help="EMA decay for L_base and L_copilot statistics.")
    parser.add_argument("--ogs_warmup_steps", type=int, default=200,
                        help="Steps with uniform sampling before switching to gap-based.")
    parser.add_argument("--ogs_alpha", type=float, default=1.0,
                        help="Exponent for the learnable-fraction term.")
    parser.add_argument("--ogs_beta", type=float, default=1.0,
                        help="Exponent for the copilot-gap term.")
    parser.add_argument("--ogs_floor_ema_decay", type=float, default=0.999,
                        help="EMA decay for the floor estimate (soft-min of copilot loss).")
    # --- WISA-80K T2V dataset ---
    parser.add_argument(
        "--wisa_manifest_format",
        action="store_true",
        default=False,
        help="Use the WISA-80K jsonl+mp4 dataset (captions+phys_law prompt, "
             "dynamic 4k+1 frames capped at --num_frames, full-span sparse sampling).",
    )
    parser.add_argument(
        "--wisa_video_root",
        type=str,
        default=None,
        help="Root dir holding WISA shard subdirs (0..127) of <sha256>.mp4 files.",
    )
    parser.add_argument(
        "--wisa_prompt_level",
        type=str,
        default="B",
        choices=["A", "B", "C"],
        help="Prompt: A=captions, B=+phys_law (default), C=+qualitative phenomena.",
    )
    return parser


def main():
    parser = joint_parser()
    args = parser.parse_args()

    if args.task in (None, "", "sft"):
        args.task = "sft:mistake_forcing"
    if args.task != "sft:mistake_forcing":
        raise ValueError(
            "Joint copilot training requires --task sft:mistake_forcing, "
            f"got {args.task!r}."
        )
    if not (args.wisa_manifest_format or args.omniworld_manifest_format):
        raise ValueError(
            "Joint copilot training requires --wisa_manifest_format or "
            "--omniworld_manifest_format."
        )
    if args.wisa_manifest_format and args.omniworld_manifest_format:
        raise ValueError(
            "--wisa_manifest_format and --omniworld_manifest_format are mutually exclusive."
        )
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}.")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

    raw_dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=tuple(),
        max_data_items=args.max_data_items,
    )
    if args.wisa_manifest_format:
        if args.wisa_video_root is None:
            raise ValueError(
                "--wisa_video_root is required with --wisa_manifest_format."
            )
        dataset = WISAManifestDataset(
            raw_dataset,
            WISAVideoOperator(
                name2path=build_video_index(args.wisa_video_root),
                frame_processor=ImageCropAndResize(
                    args.height, args.width, args.max_pixels, 16, 16
                ),
                max_frames=args.num_frames,
                prompt_level=args.wisa_prompt_level,
                fixed_frames=args.batch_size > 1,
            ),
        )
    else:
        dataset = OmniWorldManifestDataset(
            raw_dataset,
            OmniWorldFrameSequenceOperator(
                base_path=args.dataset_base_path,
                frame_processor=ImageCropAndResize(
                    args.height, args.width, args.max_pixels, 16, 16
                ),
                num_frames=args.num_frames,
            ),
        )

    if args.batch_size > 1 and dataset.load_from_cache:
        raise ValueError(
            "batch_size > 1 is not supported with cached datasets "
            "(load_from_cache=True). Use batch_size=1 or disable the dataset cache."
        )

    model = JointWanCopilotModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device=(
            "cpu"
            if (args.initialize_model_on_cpu or args.enable_model_cpu_offload)
            else accelerator.device
        ),
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        output_path=args.output_path,
        mistake_selected_layers=args.mistake_selected_layers,
        copilot_dir=args.copilot_dir,
        copilot_version=args.copilot_version,
        copilot_dim=args.copilot_dim,
        copilot_depth=args.copilot_depth,
        copilot_num_heads=args.copilot_num_heads,
        copilot_mlp_ratio=args.copilot_mlp_ratio,
        copilot_lr=args.copilot_lr,
        copilot_loss_weight=args.copilot_loss_weight,
        fuse_copilot_into_loss=args.fuse_copilot_into_dit_loss,
        copilot_fuse_scale=args.copilot_fuse_scale,
        copilot_grad_to_dit=args.copilot_grad_to_dit,
        copilot_use_gradient_checkpointing=args.copilot_use_gradient_checkpointing,
        copilot_resume=args.copilot_resume,
        copilot_n_selected_layers=args.copilot_n_selected_layers,
        copilot_selected_hidden_dim=args.copilot_selected_hidden_dim,
        optimal_gap_sampling=args.optimal_gap_sampling,
        ogs_num_bins=args.ogs_num_bins,
        ogs_ema_decay=args.ogs_ema_decay,
        ogs_warmup_steps=args.ogs_warmup_steps,
        ogs_alpha=args.ogs_alpha,
        ogs_beta=args.ogs_beta,
        ogs_floor_ema_decay=args.ogs_floor_ema_decay,
    )
    if accelerator.is_main_process:
        print(
            f"[JointCopilot] copilot {args.copilot_version} params: "
            f"{model._copilot_param_count / 1e6:.2f}M | "
            f"copilot_lr={args.copilot_lr} | "
            f"copilot_loss_weight={args.copilot_loss_weight} | "
            f"fuse_into_dit_loss={args.fuse_copilot_into_dit_loss} | "
            f"copilot_fuse_scale={args.copilot_fuse_scale} | "
            f"grad_to_dit={args.copilot_grad_to_dit} | "
            f"optimal_gap_sampling={args.optimal_gap_sampling} | "
            f"dataset_size={len(dataset)}"
        )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )

    launch_joint_training(accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    main()
