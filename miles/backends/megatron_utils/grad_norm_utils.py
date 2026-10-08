"""Gradient-norm helpers that can exclude MTP-only parameters.

The MTP (multi-token-prediction) loss is isolated to the dedicated MTP parameters
(``model.mtp.*``): its gradient is cut off from the backbone / embedding / output
layer by detaches in ``multi_token_prediction.py`` (``_get_embeddings`` uses
``make_viewless_tensor(..., keep_graph=False)`` and ``decoder_input.detach()``) and
in ``gpt_model.py`` (output-layer params are detached for the MTP path). This is
guarded by the CI invariant ``check_mtp_only_grad``. Therefore *excluding*
``model.mtp.*`` from the gradient norm yields the exact pure main-loss grad norm,
with no residual MTP contribution to subtract.

These helpers reconstruct the optimizer's own ``step()`` building blocks
(``prepare_grads`` -> clip -> ``count_zeros`` -> ``step_with_ready_grads``) so that
clipping, the skip-gate and the logged ``grad_norm`` can all use the MTP-excluded
norm, without monkeypatching Megatron.

Scope: the standard distributed / fp16 optimizer path (``use_distributed_optimizer``
is always True here). FSDP grad layout (``param.grad._local_tensor``) is not handled.
"""


def _suboptimizers(optimizer):
    """A ChainedOptimizer (MoE with a separate expert optimizer) wraps several
    sub-optimizers; a plain optimizer is treated as a single-element list."""
    return getattr(optimizer, "chained_optimizers", None) or [optimizer]


def _iter_optimizer_params(optimizer):
    """Yield the parameters the optimizer steps on (the grad-norm iterates these)."""
    for sub in _suboptimizers(optimizer):
        yield from sub.get_parameters()


def _optimizer_config(optimizer):
    """OptimizerConfig for the (first) underlying optimizer.

    ChainedOptimizer has no single ``.config``; its sub-optimizers do, and they
    share the relevant flags.
    """
    return _suboptimizers(optimizer)[0].config


def mtp_optimizer_param_ids(model):
    """Ids of the optimizer-visible params that belong to ``model.mtp.*``.

    The optimizer iterates main-param copies, not the named model params, so we
    bridge via ``param.main_param`` (set by the distributed optimizer in
    ``distrib_optimizer.py`` and by the fp16 optimizer in ``optimizer.py``). When
    there is no separate main param we fall back to the param object itself.
    """
    ids = set()
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if ".mtp." in name:
                ids.add(id(getattr(param, "main_param", param)))
    return ids


def _split_grads_by_mtp(optimizer, model, keep, get_grad):
    """Partition optimizer grads into ``(non_mtp, mtp)`` lists.

    ``keep(param)`` is the inclusion predicate (non-shared and not a TP duplicate),
    matching Megatron's ``get_main_grads_for_grad_norm`` filtering; ``get_grad(param)``
    returns the gradient tensor to use. Pure: no CUDA / distributed calls here.
    """
    mtp_ids = mtp_optimizer_param_ids(model)
    non_mtp, mtp = [], []
    for param in _iter_optimizer_params(optimizer):
        grad = get_grad(param)
        if grad is None or not keep(param):
            continue
        (mtp if id(param) in mtp_ids else non_mtp).append(grad)
    return non_mtp, mtp


def grad_norms_split_by_mtp(optimizer, model):
    """Return ``(main_norm, mtp_norm)``: grad norms of non-MTP and MTP params.

    Both are reduced over the optimizer's grad-stats group. Every rank calls
    ``get_grad_norm_fp32`` even with an empty list, because its all-reduce over the
    grad-stats group is unconditional (``clip_grads.py``) -- early-returning on an
    empty list would desync the collective (MTP params live only on the last
    pipeline stage). Because MTP loss is isolated to MTP params, ``main_norm`` is
    exactly the pure main-loss grad norm.
    """
    from megatron.core import tensor_parallel
    from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
    from megatron.core.transformer.module import param_is_not_shared

    config = _optimizer_config(optimizer)
    tp_group = getattr(_suboptimizers(optimizer)[0], "tp_group", None)

    if config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:

        def get_grad(param):
            return getattr(param, "decoupled_grad", None)

    else:

        def get_grad(param):
            return param.grad

    def keep(param):
        return param_is_not_shared(param) and tensor_parallel.param_is_not_tensor_parallel_duplicate(param, tp_group)

    non_mtp, mtp = _split_grads_by_mtp(optimizer, model, keep, get_grad)
    group = optimizer.get_grad_stats_parallel_group()
    main_norm = get_grad_norm_fp32(non_mtp, grad_stats_parallel_group=group)
    mtp_norm = get_grad_norm_fp32(mtp, grad_stats_parallel_group=group)
    return main_norm, mtp_norm


def optimizer_step_excluding_mtp(optimizer, model, clip_grad):
    """Run the optimizer step but clip by the MTP-excluded (main-loss) grad norm.

    Mirrors ``MixedPrecisionOptimizer.step`` (``prepare_grads`` -> clip ->
    ``count_zeros`` -> ``step_with_ready_grads``), substituting the main-loss norm
    for the clip coefficient. Clipping still scales *all* parameters (incl. MTP),
    only the coefficient ``clip_grad / main_norm`` excludes MTP -- so MTP params stay
    trained and stable. Returns ``(success, main_norm, num_zeros, mtp_norm)``.
    """
    from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32

    found_inf = optimizer.prepare_grads()
    if found_inf:
        return False, 0.0, 0, 0.0

    main_norm, mtp_norm = grad_norms_split_by_mtp(optimizer, model)

    config = _optimizer_config(optimizer)
    if clip_grad and clip_grad > 0.0:
        clip_grad_by_total_norm_fp32(
            list(_iter_optimizer_params(optimizer)),
            clip_grad,
            main_norm,
            config.use_precision_aware_optimizer_no_fp8_or_ds_fp8,
        )
    num_zeros = optimizer.count_zeros() if config.log_num_zeros_in_grad else 0
    success = optimizer.step_with_ready_grads()
    return success, main_norm, num_zeros, mtp_norm
