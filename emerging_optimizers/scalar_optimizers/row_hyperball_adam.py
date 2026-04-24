# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Row-wise Hyperball Adam optimizer.

This optimizer is intended for matrices whose rows should stay on a fixed-radius
sphere, for example untied LM-head or embedding weights with shape [vocab, hidden].

Initialization normalizes each row to the configured target_row_norm.

Each step is stateless with respect to radii -- it measures the current row norms
r_i = ||w_i||_2, then:
    1. Compute the standard Adam update u_i.
    2. Normalize the update row-wise: d_i = u_i / ||u_i||_2.
    3. Take a tangent step: w_i <- w_i - lr * r_i * d_i.
    4. Retract back to the measured radius: w_i <- r_i * w_i / ||w_i||_2.
"""

from typing import Tuple

import torch
from torch.optim import Optimizer

from emerging_optimizers.scalar_optimizers.adam import calculate_adam_update


__all__ = ["RowWiseHyperballAdam"]


def _normalize_rows_(
    tensor: torch.Tensor, target_row_norm: float = 1.0, eps: float = 1e-12
) -> None:
    """Normalize each row in-place to the target norm."""
    if tensor.ndim != 2:
        raise ValueError(f"RowWiseHyperballAdam expects 2D tensors, got shape {tuple(tensor.shape)}")
    row_norms = torch.norm(tensor.float(), p=2, dim=-1, keepdim=True).clamp_min(eps)
    tensor.mul_(target_row_norm)
    tensor.div_(row_norms.to(dtype=tensor.dtype))


def _row_norms(
    tensor: torch.Tensor, eps: float = 1e-12, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Return per-row L2 norms with numerical floor."""
    row_norms = torch.norm(tensor.float(), p=2, dim=-1, keepdim=True).clamp_min(eps)
    return row_norms.to(dtype=tensor.dtype if dtype is None else dtype)


def _retract_rows_(tensor: torch.Tensor, row_norms: torch.Tensor) -> None:
    """Project each row in-place back onto its measured-radius sphere."""
    current = _row_norms(tensor, dtype=tensor.dtype)
    tensor.mul_(row_norms / current)


class RowWiseHyperballAdam(Optimizer):
    """Adam with row-wise sphere projection.

    At init, rows are normalized to ``target_row_norm``.  Each step measures
    the current row norms, uses them to scale the update direction, and
    retracts back to those same radii.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
        target_row_norm: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if target_row_norm <= 0.0:
            raise ValueError(f"Invalid target row norm: {target_row_norm}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
            target_row_norm=target_row_norm,
        )
        super().__init__(params, defaults)

        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    _normalize_rows_(p.data, target_row_norm=group["target_row_norm"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            bias_correction = group["bias_correction"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if p.ndim != 2:
                    raise ValueError(
                        f"RowWiseHyperballAdam only supports 2D tensors, got shape {tuple(p.shape)}"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # Measure current row radii -- this is the radius we preserve.
                row_radii = _row_norms(p.data)

                state["step"] += 1
                step_count = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                u_t = calculate_adam_update(
                    grad=grad,
                    exp_avg=exp_avg,
                    exp_avg_sq=exp_avg_sq,
                    betas=betas,
                    correct_bias=bias_correction,
                    use_nesterov=False,
                    step=step_count,
                    eps=eps,
                )

                row_update_norms = _row_norms(u_t, dtype=u_t.dtype)
                d_t = u_t / row_update_norms

                p.data.add_(row_radii * d_t, alpha=-lr)
                _retract_rows_(p.data, row_radii)

        return loss
