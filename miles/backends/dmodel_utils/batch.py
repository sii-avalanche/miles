import torch


def _global_document_positions(batch: dict, local_positions: torch.Tensor) -> torch.Tensor:
    """Rebuild full CP metadata from the pre-slice rollout samples."""
    from miles.backends.training_utils.parallel import get_parallel_state

    cp_size = get_parallel_state().cp.size
    if cp_size == 1:
        return local_positions
    samples = batch.get("unconcat_tokens")
    if not samples:
        raise ValueError("DModel CP metadata requires pre-slice unconcat_tokens")

    global_parts = []
    for sample in samples:
        sample_length = int(sample.numel())
        padded_length = 2 * cp_size * ((sample_length + 2 * cp_size - 1) // (2 * cp_size))
        positions = torch.arange(padded_length, device=local_positions.device, dtype=torch.long)
        if padded_length > sample_length:
            positions[sample_length:] = 0
        global_parts.append(positions)
    positions = torch.cat(global_parts)
    expected = local_positions.shape[1] * cp_size
    if positions.numel() > expected:
        raise ValueError(
            f"DModel CP metadata exceeds local batch contract: {positions.numel()} > {expected}"
        )
    if positions.numel() < expected:
        positions = torch.nn.functional.pad(positions, (0, expected - positions.numel()), value=0)
    return positions.unsqueeze(0)


def dmodel_forward_kwargs(batch: dict) -> dict:
    """Convert the already CP-local Miles THD batch to DModel's model contract.

    ``get_batch`` owns CP slicing and padding.  Re-slicing here would apply the
    zigzag transform twice when context parallelism is enabled.
    """
    from dmodel.models.attention.metadata import PackedSequenceMetadata

    tokens = batch["tokens"]
    positions = batch["position_ids"]
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError(f"DModel THD tokens must have shape [1, T], got {tuple(tokens.shape)}")
    if positions.shape != tokens.shape:
        raise ValueError(
            f"DModel document positions must match tokens: {tuple(positions.shape)} != {tuple(tokens.shape)}"
        )
    if tokens.dtype != torch.long:
        raise ValueError(f"DModel token ids must be int64, got {tokens.dtype}")
    return {
        "tokens": tokens,
        "varlen_metadata": PackedSequenceMetadata(
            document_positions=_global_document_positions(batch, positions)
        ),
    }
