"""Megatron MuonHyperball optimizer wrapper."""

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
from emerging_optimizers.orthogonalized_optimizers.muon_hyperball import MuonHyperball
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


def get_megatron_muon_hyperball_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Get the MuonHyperball optimizer for model chunks.

    MuonHyperball uses Muon's msign orthogonalization with Frobenius-norm sphere
    retraction (instead of MuonBall's spectral-norm retraction).

    This function creates a chained optimizer where:
    - Linear weights (2D tensors) use MuonHyperball with Frobenius sphere constraints
    - Optionally, the untied LM head uses HyperballAdam
    - Remaining parameters (biases, norms, embeddings) use Adam

    Args:
        config: OptimizerConfig instance.
        model_chunks: List of model chunks to optimize.
        no_weight_decay_cond: Optional function to determine if a parameter should skip weight decay.
        scale_lr_cond: Optional function to determine if a parameter should use scaled learning rate.
        lr_mult: Learning rate multiplier for scaled parameters.
        use_gloo_process_groups: Whether to use Gloo process groups.
        pg_collection: Optional ProcessGroupCollection for distributed training.

    Returns:
        MegatronOptimizer instance (ChainedOptimizer).
    """
    # Distributed optimizer is not supported
    if config.use_distributed_optimizer:
        raise Exception('muon_hyperball with distributed optimizer is not supported.')
    if config.hyperball_lm_head and config.row_hyperball_lm_head:
        raise ValueError("hyperball_lm_head and row_hyperball_lm_head are mutually exclusive.")
    if config.hyperball_embeddings and config.row_hyperball_embeddings:
        raise ValueError(
            "hyperball_embeddings and row_hyperball_embeddings are mutually exclusive."
        )

    # Set up process groups
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

    log_single_rank(
        logger, logging.INFO, f'Setting up MuonHyperball optimizer with config {config}'
    )

    optimizers = []
    linear_params = []
    lm_head_params = []
    embedding_params = []
    nonlinear_params = []

    # Categorize parameters into linear (2D) and non-linear (1D, embeddings)
    # Tag QKV and expert parameters for TP-aware version
    qkv_split_shapes: Optional[list[int]] = None
    fc1_split_shapes: Optional[list[int]] = None
    for model_chunk in model_chunks:
        # derive qkv split shapes from model config if available
        try:
            num_attention_heads = model_chunk.config.num_attention_heads
            num_query_groups = model_chunk.config.num_query_groups
            kv_channels = model_chunk.config.kv_channels
            qkv_split_shapes = [
                num_attention_heads // num_query_groups * kv_channels,
                kv_channels,
                kv_channels,
            ]
        except Exception:
            pass
        # derive fc1 split shapes for gated linear units (SwiGLU)
        try:
            if model_chunk.config.gated_linear_unit:
                ffn_hidden_size = model_chunk.config.ffn_hidden_size
                fc1_split_shapes = [ffn_hidden_size, ffn_hidden_size]  # gate, up
        except Exception:
            pass
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            # Store parameter name for logging
            param.param_name = name

            # expert flag for MoE
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # QKV fused linear
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            # FC1 fused linear for gated linear units (SwiGLU)
            if 'linear_fc1.weight' in name and len(param.shape) == 2:
                param.is_fc1 = True
            # add flag for GroupedMLP weight1/weight2 (MoE experts)
            if 'experts.weight1' in name or 'experts.weight2' in name:
                param.is_grouped_moe = True
                try:
                    param.num_local_experts = model_chunk.config.num_moe_experts // model_chunk.config.expert_model_parallel_size
                    param.moe_ffn_hidden_size = model_chunk.config.moe_ffn_hidden_size
                    param.is_gated = model_chunk.config.gated_linear_unit
                except Exception:
                    param.is_grouped_moe = False

            is_lm_head = (
                config.hyperball_lm_head or config.row_hyperball_lm_head
            ) and _is_lm_head_param(name, param)
            is_embedding = (
                config.hyperball_embeddings or config.row_hyperball_embeddings
            ) and _is_embedding_param(name, param)

            # Linear weights: 2D tensors that are not embeddings/output params.
            # The untied LM head and embeddings can optionally be routed to HyperballAdam instead.
            if is_lm_head:
                lm_head_params.append(param)
            elif is_embedding:
                embedding_params.append(param)
            elif (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    # ==================== Setup MuonHyperball for linear params ====================
    # Freeze params that are not owned by the MuonHyperball optimizer.
    # In particular, when hyperball_lm_head is enabled the untied LM head is
    # handled by a separate HyperballAdam instance and must be excluded here to
    # avoid duplicate optimizer state entries during checkpoint save/load.
    for param in nonlinear_params:
        param.requires_grad = False
    for param in lm_head_params:
        param.requires_grad = False
    for param in embedding_params:
        param.requires_grad = False

    # Get param groups for linear params
    # Force all linear params to have wd_mult=0.0 (no weight decay for linear layers)
    linear_no_weight_decay_cond = lambda name, param: True
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

    # Create MuonHyperball optimizer
    muon_hyperball_optimizer = MuonHyperball(
        linear_param_groups,
        lr=config.lr,
        momentum_beta=config.muon_hyperball_momentum,
        use_nesterov=config.muon_hyperball_use_nesterov,
        weight_decay=config.weight_decay,
        weight_decay_method="decoupled" if config.decoupled_weight_decay else "coupled",
        fp32_matmul_prec="medium",
        msign_steps=config.muon_hyperball_msign_steps,
        radius_mode=config.muon_hyperball_radius_mode,
        scale_mode=config.muon_hyperball_scale_mode,
        split_qkv=config.muon_hyperball_split_qkv,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=tuple(qkv_split_shapes) if qkv_split_shapes is not None else None,
        qkv_split_mode=config.muon_hyperball_qkv_split_mode,
        split_fc1=config.muon_hyperball_split_fc1,
        is_fc1_fn=lambda p: getattr(p, 'is_fc1', False),
        fc1_split_shapes=tuple(fc1_split_shapes) if fc1_split_shapes is not None else None,
        split_moe_experts=config.muon_hyperball_split_moe_experts,
        is_grouped_moe_fn=lambda p: getattr(p, 'is_grouped_moe', False),
        pg_collection=pg_collection,
        tp_mode='duplicated',
        ns_init=config.muon_hyperball_ns_init,
    )

    # Save original optimizer name and switch to adam for the rest
    original_optimizer = config.optimizer
    config.optimizer = 'adam'

    # Define init state function for MuonHyperball
    def muon_hyperball_init_state_fn(opt, config=None):
        """Initialize MuonHyperball optimizer state for checkpointing."""
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

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
        raise Exception('muon_hyperball with fp16 is not supported.')

    if config.bf16:
        muon_hyperball_optimizer = Float16OptimizerWithFloat16Params(
            muon_hyperball_optimizer, config, None, muon_hyperball_init_state_fn
        )
    else:
        muon_hyperball_optimizer = FP32Optimizer(
            muon_hyperball_optimizer, config, muon_hyperball_init_state_fn
        )

    optimizers.append(muon_hyperball_optimizer)

    # ==================== Optional HyperballAdam for LM head / embeddings ====================
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

    def _wrap_special_hyperball_optimizer(params, use_rowwise, target_row_norm=None):
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

        if use_rowwise:
            optimizer = RowWiseHyperballAdam(
                param_groups,
                lr=config.lr,
                betas=(config.hyperball_adam_beta1, config.hyperball_adam_beta2),
                eps=config.hyperball_adam_eps,
                weight_decay=0.0,
                bias_correction=config.hyperball_adam_bias_correction,
                target_row_norm=target_row_norm,
            )
            init_state_fn = row_hyperball_adam_init_state_fn
        else:
            optimizer = HyperballAdam(
                param_groups,
                lr=config.lr,
                betas=(config.hyperball_adam_beta1, config.hyperball_adam_beta2),
                eps=config.hyperball_adam_eps,
                weight_decay=0.0,
                bias_correction=config.hyperball_adam_bias_correction,
            )
            init_state_fn = hyperball_adam_init_state_fn

        if config.bf16:
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, init_state_fn
            )
        else:
            optimizer = FP32Optimizer(optimizer, config, init_state_fn)

        if use_rowwise and target_row_norm is not None:
            _install_post_load_renormalize(optimizer, target_row_norm)

        optimizers.append(optimizer)

    if lm_head_params:
        _wrap_special_hyperball_optimizer(
            lm_head_params,
            use_rowwise=config.row_hyperball_lm_head,
            target_row_norm=_resolve_row_target_norm(
                lm_head_params[0],
                config.row_hyperball_lm_head_target_row_norm,
                config.row_hyperball_lm_head_target_row_norm_mode,
            ) if config.row_hyperball_lm_head else None,
        )

    if embedding_params:
        _wrap_special_hyperball_optimizer(
            embedding_params,
            use_rowwise=config.row_hyperball_embeddings,
            target_row_norm=_resolve_row_target_norm(
                embedding_params[0],
                config.row_hyperball_embeddings_target_row_norm,
                config.row_hyperball_embeddings_target_row_norm_mode,
            ) if config.row_hyperball_embeddings else None,
        )

    # ==================== Setup Adam for non-linear params ====================
    # Unfreeze non-linear params and freeze matrix-constrained params
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False
    for param in lm_head_params:
        param.requires_grad = False
    for param in embedding_params:
        param.requires_grad = False

    # Get Adam optimizer for non-linear params
    chained_adam = get_megatron_optimizer(
        config, model_chunks, no_weight_decay_cond, scale_lr_cond, lr_mult, use_gloo_process_groups
    )

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
