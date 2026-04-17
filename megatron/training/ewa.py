"""Exponential Weight Averaging (EWA) for Megatron-LM.

Supports one or more parallel shadow tracks (different coefficients).

Standard (time_scaled=False), per track i with decay τ_i:

    ξ_{t+1} = τ_i · ξ_t + (1 − τ_i) · θ_t

Time-scaled (time_scaled=True), each coefficient c_i specifies the fraction
of elapsed iterates to average over.  At timestep t (1-based since
``start_iter``), the mixing rate is:

    β_t = min(1, 1 / (c_i · t))
    ξ_{t+1} = (1 − β_t) · ξ_t + β_t · θ_t

So c=0.04 means "average over the last 4 % of iterates so far".  For
t < 1/c the shadow simply copies live weights (β_t = 1).

Evaluation can swap in any shadow track for validation.
"""

import os
from contextlib import contextmanager
from typing import List, Optional

import torch

from megatron.core.utils import log_single_rank
from megatron.training.utils import print_rank_0, unwrap_model

import logging

logger = logging.getLogger(__name__)


def ewa_beta_tag(beta: float) -> str:
    """Stable string tag for logging keys (avoid '.' in metric names)."""
    s = f"{beta:.10g}"
    return s.replace(".", "p").replace("-", "m")


class EWAState:
    """Manages one or more shadow (EWA-averaged) copies of model parameters."""

    def __init__(
        self,
        model: list,
        betas: List[float],
        start_iter: int = 0,
        time_scaled: bool = False,
    ):
        if not betas:
            raise ValueError("EWAState requires at least one beta / decay value.")
        self.betas = list(betas)
        self.start_iter = int(start_iter)
        self.time_scaled = bool(time_scaled)
        # _shadow[k] = shadow dict for betas[k]
        self._shadow: List[List[dict]] = []

        for _ in self.betas:
            shadow_chunks: List[dict] = []
            for model_chunk in unwrap_model(model):
                shadow_dict = {}
                for name, param in model_chunk.named_parameters():
                    if param.requires_grad:
                        shadow_dict[name] = param.data.float().clone()
                shadow_chunks.append(shadow_dict)
            self._shadow.append(shadow_chunks)

        log_single_rank(
            logger,
            logging.INFO,
            f'[EWA] Initialized {len(self.betas)} shadow tracks '
            f'(betas={self.betas}, time_scaled={self.time_scaled}, '
            f'start_iter={self.start_iter}, chunks={len(self._shadow[0])}, '
            f'params={sum(len(s) for s in self._shadow[0])})',
        )

    @torch.no_grad()
    def update(self, model: list, iteration: int):
        """Update every shadow track from current model weights."""
        unwrapped = unwrap_model(model)
        for k, beta in enumerate(self.betas):
            self._update_track(unwrapped, iteration, k, beta)

    @torch.no_grad()
    def _update_track(self, unwrapped, iteration: int, track_idx: int, beta: float):
        shadow_chunks = self._shadow[track_idx]
        if iteration < self.start_iter:
            for shadow_dict, model_chunk in zip(shadow_chunks, unwrapped):
                for name, param in model_chunk.named_parameters():
                    if name in shadow_dict:
                        shadow_dict[name].copy_(param.data.float())
            return

        if self.time_scaled:
            t = iteration - self.start_iter + 1
            c = float(beta)
            # β_t = min(1, 1/(c*t)) — average over last c fraction of iterates
            if c * t < 1.0:
                mix = 1.0
            else:
                mix = 1.0 / (c * t)
        else:
            tau = float(beta)
            mix = 1.0 - tau

        for shadow_dict, model_chunk in zip(shadow_chunks, unwrapped):
            for name, param in model_chunk.named_parameters():
                if name in shadow_dict:
                    shadow_dict[name].lerp_(param.data.float(), mix)

    @contextmanager
    def swap_for_eval(self, model: list, track_idx: int = 0):
        """Swap model weights with shadow track ``track_idx`` for eval; restore after."""
        if track_idx < 0 or track_idx >= len(self._shadow):
            raise ValueError(f"Invalid EWA track_idx={track_idx} (num tracks={len(self.betas)})")
        unwrapped = unwrap_model(model)
        shadow_chunks = self._shadow[track_idx]
        saved = []
        try:
            for shadow_dict, model_chunk in zip(shadow_chunks, unwrapped):
                chunk_saved = {}
                for name, param in model_chunk.named_parameters():
                    if name in shadow_dict:
                        chunk_saved[name] = param.data.clone()
                        param.data.copy_(shadow_dict[name].to(param.data.dtype))
                saved.append(chunk_saved)
            yield
        finally:
            for chunk_saved, model_chunk in zip(saved, unwrapped):
                for name, param in model_chunk.named_parameters():
                    if name in chunk_saved:
                        param.data.copy_(chunk_saved[name])

    def state_dict(self):
        """Return serializable state for checkpointing."""
        return {
            'betas': self.betas,
            'time_scaled': self.time_scaled,
            'start_iter': self.start_iter,
            'shadow': [
                [{name: tensor.cpu() for name, tensor in sd.items()} for sd in chunks]
                for chunks in self._shadow
            ],
        }

    def load_state_dict(self, state_dict):
        """Load shadow params from a checkpoint."""
        # Legacy format: {'decay': float, 'shadow': [chunk_dict, ...]}
        if 'decay' in state_dict and state_dict.get('shadow') and isinstance(
            state_dict['shadow'][0], dict
        ):
            legacy_beta = float(state_dict['decay'])
            if len(self.betas) != 1 or abs(self.betas[0] - legacy_beta) > 1e-12:
                print_rank_0(
                    f'[EWA] Warning: legacy checkpoint decay={legacy_beta} vs current betas={self.betas}'
                )
            shadow_saved = [state_dict['shadow']]
        else:
            saved_betas = state_dict.get('betas')
            if saved_betas is not None and list(saved_betas) != list(self.betas):
                print_rank_0(
                    f'[EWA] Warning: checkpoint betas {saved_betas} != current {self.betas}; '
                    f'loading tensors where shapes match.'
                )
            shadow_saved = state_dict['shadow']

        for tr, chunks_saved in enumerate(shadow_saved):
            if tr >= len(self._shadow):
                break
            for sd_saved, sd_live in zip(chunks_saved, self._shadow[tr]):
                for name in sd_live:
                    if name in sd_saved:
                        sd_live[name].copy_(sd_saved[name])
        log_single_rank(logger, logging.INFO, '[EWA] Loaded shadow params from checkpoint')


def save_ewa_checkpoint(ewa_state, checkpoint_dir, iteration):
    """Save EWA state as a separate file next to the main checkpoint."""
    if ewa_state is None:
        return
    if torch.distributed.get_rank() == 0:
        path = os.path.join(checkpoint_dir, f'ewa_state_iter_{iteration:07d}.pt')
        torch.save(ewa_state.state_dict(), path)
        latest_path = os.path.join(checkpoint_dir, 'latest_ewa_state.pt')
        torch.save(ewa_state.state_dict(), latest_path)
        print_rank_0(f'[EWA] Saved shadow params to {path}')


def load_ewa_checkpoint(ewa_state, checkpoint_dir):
    """Load EWA state from the latest saved file."""
    if ewa_state is None:
        return
    path = os.path.join(checkpoint_dir, 'latest_ewa_state.pt')
    if os.path.exists(path):
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        ewa_state.load_state_dict(state_dict)
    else:
        print_rank_0(f'[EWA] No saved shadow params found at {path}, starting fresh')
