"""Legacy Dr.GRPO example entrypoint; the core reducer also supports context parallelism."""

from miles.backends.training_utils.data.context_parallel import get_drgrpo_pgloss_reducer

DIVISOR = 1000.0


def get_pg_loss_reducer(
    total_lengths,
    response_lengths,
    loss_masks,
    calculate_per_token_loss=False,
    qkv_format="thd",
    max_seq_lens=None,
):
    return get_drgrpo_pgloss_reducer(
        total_lengths,
        response_lengths,
        loss_masks,
        calculate_per_token_loss,
        qkv_format,
        max_seq_lens,
        fixed_normalizer=DIVISOR,
    )
