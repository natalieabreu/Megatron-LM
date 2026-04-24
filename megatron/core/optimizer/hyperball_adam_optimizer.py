"""Megatron HyperballAdam optimizer wrapper."""

import logging
from typing import Callable, List, Optional

import torch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
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
from emerging_optimizers.scalar_optimizers.hyperball_adam import HyperballAdam
from emerging_optimizers.scalar_optimizers.row_hyperball_adam import RowWiseHyperballAdam, _normalize_rows_

logger = logging.getLogger(__name__)


def _is_lm_head_param(name: str, param: torch.nn.Parameter) -> bool:
    """Return True for the untied LM head weight."""
    return "output_layer" in name and "embedding" not in name and len(param.shape) == 2


def _is_embedding_param(name: str, param: torch.nn.Parameter) -> bool:
    """Return True for token embedding weights."""
    return "word_embeddings" in name and len(param.shape) == 2


def _resolve_row_target_norm(param: torch.nn.Parameter, value: float, mode: str) -> float:
    """Resolve a configured row target norm."""
    if mode == "absolute":
        return float(value)
    if mode == "times_sqrt_d":
        return float(value) * (param.shape[-1] ** 0.5)
    raise ValueError(f"Unsupported row target norm mode: {mode}")


def _install_post_load_renormalize(wrapper, target_row_norm: float):
    """Wrap load_state_dict so fp32 master params are re-normalized after checkpoint load."""
    original_load = wrapper.load_state_dict

    def _load_and_renormalize(state_dict):
        original_load(state_dict)
        groups = getattr(wrapper, 'fp32_from_float16_groups',
                         getattr(wrapper, 'fp32_from_fp32_groups', []))
        for group in groups:
            for p in group:
                if p.ndim == 2:
                    _normalize_rows_(p.data, target_row_norm=target_row_norm)

    wrapper.load_state_dict = _load_and_renormalize


def get_megatron_hyperball_adam_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
) -> MegatronOptimizer:
    """Get the HyperballAdam optimizer for model chunks.

    This function creates a chained optimizer where:
    - Linear weights (2D tensors) use HyperballAdam with Frobenius-norm sphere constraint
    - Optionally, the untied LM head also uses HyperballAdam
    - Remaining parameters (biases, norms, embeddings) use standard Adam

    The update rule for 2D weights:
        W_{t+1} = R * Normalize(W_t - lr * R * Normalize(u_t))
    where R = ||W_0||_F and u_t is the Adam update direction.

    Args:
        config: OptimizerConfig instance.
        model_chunks: List of model chunks to optimize.
        no_weight_decay_cond: Optional function to determine if a parameter should skip weight decay.
        scale_lr_cond: Optional function to determine if a parameter should use scaled learning rate.
        lr_mult: Learning rate multiplier for scaled parameters.
        use_gloo_process_groups: Whether to use Gloo process groups.

    Returns:
        MegatronOptimizer instance (ChainedOptimizer).
    """
    # Distributed optimizer is not supported
    if config.use_distributed_optimizer:
        raise Exception('hyperball_adam with distributed optimizer is not supported.')
    if config.hyperball_lm_head and config.row_hyperball_lm_head:
        raise ValueError("hyperball_lm_head and row_hyperball_lm_head are mutually exclusive.")
    if config.hyperball_embeddings and config.row_hyperball_embeddings:
        raise ValueError(
            "hyperball_embeddings and row_hyperball_embeddings are mutually exclusive."
        )

    log_single_rank(
        logger, logging.INFO, f'Setting up HyperballAdam optimizer with config {config}'
    )

    optimizers = []
    linear_params = []
    lm_head_params = []
    embedding_params = []
    nonlinear_params = []

    # Categorize parameters into linear (2D) and non-linear (1D, embeddings)
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            # Store parameter name for logging
            param.param_name = name

            is_hyperball_lm_head = config.hyperball_lm_head and _is_lm_head_param(name, param)
            is_row_hyperball_lm_head = config.row_hyperball_lm_head and _is_lm_head_param(name, param)
            is_hyperball_embedding = config.hyperball_embeddings and _is_embedding_param(name, param)
            is_row_hyperball_embedding = (
                config.row_hyperball_embeddings and _is_embedding_param(name, param)
            )

            # Linear weights: 2D tensors that are not embeddings/output params,
            # plus the untied LM head / embeddings when explicitly requested.
            if (
                (not getattr(param, 'is_embedding_or_output_parameter', False) and len(param.shape) == 2)
                or is_hyperball_lm_head
                or is_hyperball_embedding
            ):
                linear_params.append(param)
            elif is_row_hyperball_lm_head:
                lm_head_params.append(param)
            elif is_row_hyperball_embedding:
                embedding_params.append(param)
            else:
                nonlinear_params.append(param)

    # ==================== Setup HyperballAdam for linear params ====================
    # Freeze non-linear params temporarily
    for param in nonlinear_params:
        param.requires_grad = False
    for param in lm_head_params:
        param.requires_grad = False
    for param in embedding_params:
        param.requires_grad = False

    # Get param groups for linear params
    # Force all linear params to have wd_mult=0.0 (no weight decay for linear layers)
    # HyperballAdam constrains weights to the Frobenius sphere, so weight decay is unnecessary
    linear_no_weight_decay_cond = lambda name, param: True  # All linear params skip weight decay
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

    # Create HyperballAdam optimizer
    hyperball_adam_optimizer = HyperballAdam(
        linear_param_groups,
        lr=config.lr,
        betas=(config.hyperball_adam_beta1, config.hyperball_adam_beta2),
        eps=config.hyperball_adam_eps,
        weight_decay=0.0,  # No weight decay needed — sphere constraint replaces it
        bias_correction=config.hyperball_adam_bias_correction,
    )

    # Save original optimizer name and switch to adam for the rest
    original_optimizer = config.optimizer
    config.optimizer = 'adam'

    # Define init state function for HyperballAdam
    def hyperball_adam_init_state_fn(opt, config=None):
        """Initialize HyperballAdam optimizer state for checkpointing."""
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    opt.state[p]['step'] = 0
                    opt.state[p]['initial_frobenius_norm'] = torch.norm(
                        p.data.float(), p='fro'
                    ).item()

    def row_hyperball_adam_init_state_fn(opt, config=None):
        """Initialize RowWiseHyperballAdam optimizer state for checkpointing."""
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                    opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    opt.state[p]['step'] = 0

    # Define init state function for Adam
    def adam_init_state_fn(opt, config=None):
        """Initialize Adam optimizer state for checkpointing."""
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    # Wrap in precision-aware optimizer
    if config.fp16:
        raise Exception('hyperball_adam with fp16 is not supported.')

    if config.bf16:
        hyperball_adam_optimizer = Float16OptimizerWithFloat16Params(
            hyperball_adam_optimizer, config, None, hyperball_adam_init_state_fn
        )
    else:
        hyperball_adam_optimizer = FP32Optimizer(
            hyperball_adam_optimizer, config, hyperball_adam_init_state_fn
        )

    optimizers.append(hyperball_adam_optimizer)

    def _wrap_rowwise_optimizer(params, target_row_norm):
        for param in nonlinear_params:
            param.requires_grad = False
        for param in linear_params:
            param.requires_grad = False
        for param in lm_head_params:
            param.requires_grad = False
        for param in embedding_params:
            param.requires_grad = False
        for param in params:
            param.requires_grad = True

        no_weight_decay_cond = lambda name, param: True
        param_groups = _get_param_groups(
            model_chunks,
            no_weight_decay_cond,
            scale_lr_cond,
            lr_mult,
            lr=config.lr,
            min_lr=config.min_lr,
            decoupled_lr=config.decoupled_lr,
            decoupled_min_lr=config.decoupled_min_lr,
        )

        optimizer = RowWiseHyperballAdam(
            param_groups,
            lr=config.lr,
            betas=(config.hyperball_adam_beta1, config.hyperball_adam_beta2),
            eps=config.hyperball_adam_eps,
            weight_decay=0.0,
            bias_correction=config.hyperball_adam_bias_correction,
            target_row_norm=target_row_norm,
        )

        if config.bf16:
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, row_hyperball_adam_init_state_fn
            )
        else:
            optimizer = FP32Optimizer(
                optimizer, config, row_hyperball_adam_init_state_fn
            )

        _install_post_load_renormalize(optimizer, target_row_norm)
        optimizers.append(optimizer)

    if lm_head_params:
        _wrap_rowwise_optimizer(
            lm_head_params,
            _resolve_row_target_norm(
                lm_head_params[0],
                config.row_hyperball_lm_head_target_row_norm,
                config.row_hyperball_lm_head_target_row_norm_mode,
            ),
        )

    if embedding_params:
        _wrap_rowwise_optimizer(
            embedding_params,
            _resolve_row_target_norm(
                embedding_params[0],
                config.row_hyperball_embeddings_target_row_norm,
                config.row_hyperball_embeddings_target_row_norm_mode,
            ),
        )

    # ==================== Setup Adam for non-linear params ====================
    # Unfreeze non-linear params and freeze linear params
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False
    for param in lm_head_params:
        param.requires_grad = False
    for param in embedding_params:
        param.requires_grad = False

    # Get Adam optimizer for non-linear params, with optional fallback-specific overrides.
    fallback_lr = config.lr * config.hyperball_adam_fallback_lr_scale
    fallback_weight_decay = (
        config.weight_decay
        if config.hyperball_adam_fallback_weight_decay is None
        else config.hyperball_adam_fallback_weight_decay
    )
    original_lr = config.lr
    original_weight_decay = config.weight_decay
    config.lr = fallback_lr
    config.weight_decay = fallback_weight_decay
    chained_adam = get_megatron_optimizer(
        config, model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult, use_gloo_process_groups
    )
    config.lr = original_lr
    config.weight_decay = original_weight_decay

    # Unfreeze all params
    for param in linear_params:
        param.requires_grad = True
    for param in lm_head_params:
        param.requires_grad = True
    for param in embedding_params:
        param.requires_grad = True

    # Restore original optimizer name
    config.optimizer = original_optimizer

    # Chain optimizers together
    optimizers += chained_adam.chained_optimizers

    return ChainedOptimizer(optimizers)
