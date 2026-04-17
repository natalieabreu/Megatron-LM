"""Megatron Scion optimizer wrapper."""

import logging
from typing import Callable, List, Optional

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import log_single_rank

from . import _get_param_groups, get_megatron_optimizer
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig
from emerging_optimizers.orthogonalized_optimizers import Scion

logger = logging.getLogger(__name__)


def get_megatron_scion_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
) -> MegatronOptimizer:
    """Get the Scion optimizer for model chunks.

    This function creates a chained optimizer where:
    - Linear weights (2D tensors) use Scion
    - Non-linear parameters (biases, norms, embeddings, output layer) use Adam
    """
    if config.use_distributed_optimizer:
        raise Exception('scion with distributed optimizer is not supported.')

    log_single_rank(logger, logging.INFO, f'Setting up Scion optimizer with config {config}')

    optimizers = []
    linear_params = []
    nonlinear_params = []

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            param.param_name = name

            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    for param in nonlinear_params:
        param.requires_grad = False

    # Scion uses the decay term as part of its constrained Frank-Wolfe-style update,
    # so its owned 2D weights must keep wd_mult=1.0 instead of being zeroed out.
    linear_no_weight_decay_cond = lambda name, param: False
    linear_param_groups = _get_param_groups(
        model_chunks,
        linear_no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )

    scion_optimizer = Scion(
        linear_param_groups,
        lr=config.lr,
        momentum_beta=config.scion_momentum,
        fp32_matmul_prec=config.scion_fp32_matmul_prec,
        coefficient_type=config.scion_coefficient_type,
        num_ns_steps=config.scion_num_ns_steps,
        scale_mode=config.scion_scale_mode,
        spectral_radius=config.scion_spectral_radius,
    )

    if hasattr(scion_optimizer, 'log_per_module_update_rms'):
        scion_optimizer.log_per_module_update_rms = config.log_per_module_update_rms
    if hasattr(scion_optimizer, 'log_per_module_grad_rms'):
        scion_optimizer.log_per_module_grad_rms = config.log_per_module_grad_rms

    original_optimizer = config.optimizer
    config.optimizer = 'adam'

    def scion_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    if config.fp16:
        raise Exception('scion with fp16 is not supported.')

    if config.bf16:
        scion_optimizer = Float16OptimizerWithFloat16Params(
            scion_optimizer, config, None, scion_init_state_fn
        )
    else:
        scion_optimizer = FP32Optimizer(scion_optimizer, config, scion_init_state_fn)

    optimizers.append(scion_optimizer)

    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    chained_adam = get_megatron_optimizer(
        config, model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult, use_gloo_process_groups
    )

    for param in linear_params:
        param.requires_grad = True

    config.optimizer = original_optimizer
    optimizers += chained_adam.chained_optimizers

    return ChainedOptimizer(optimizers)
