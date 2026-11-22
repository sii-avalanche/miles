import pytest
import torch

from miles.backends.dmodel_utils.batch import dmodel_forward_kwargs
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo


def test_packed_batch_uses_positions_as_document_metadata():
    one = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=one,
            intra_dp_cp=one,
            cp=one,
            tp=one,
            pp=one,
            ep=one,
            etp=one,
            indep_dp=one,
        )
    )
    tokens = torch.tensor([[10, 11, 12, 20, 21]], dtype=torch.long)
    positions = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    result = dmodel_forward_kwargs({"tokens": tokens, "position_ids": positions})
    assert result["tokens"] is tokens
    assert set(result) == {"tokens", "varlen_metadata"}
    assert torch.equal(result["varlen_metadata"].document_positions, positions)


def test_packed_batch_preserves_existing_zigzag_ring_context_slice():
    one = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=0, size=2, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=one,
            intra_dp_cp=one,
            cp=cp,
            tp=one,
            pp=one,
            ep=one,
            etp=one,
            indep_dp=one,
        )
    )
    # get_batch has already performed the CP slice and padding before this
    # adapter is called.  The adapter must not apply the transform again.
    tokens = torch.tensor([[10, 11, 0, 0]], dtype=torch.long)
    positions = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    result = dmodel_forward_kwargs(
        {
            "tokens": tokens,
            "position_ids": positions,
            "unconcat_tokens": [torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.long)],
        }
    )
    assert result["tokens"].shape == (1, 4)
    assert result["tokens"] is tokens
    assert torch.equal(
        result["varlen_metadata"].document_positions,
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long),
    )


def test_packed_batch_requires_original_samples_for_cp_metadata():
    one = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=0, size=2, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=one,
            intra_dp_cp=one,
            cp=cp,
            tp=one,
            pp=one,
            ep=one,
            etp=one,
            indep_dp=one,
        )
    )
    with pytest.raises(ValueError, match="unconcat_tokens"):
        dmodel_forward_kwargs(
            {
                "tokens": torch.ones((1, 4), dtype=torch.long),
                "position_ids": torch.arange(4).unsqueeze(0),
            }
        )


@pytest.mark.parametrize(
    "tokens,positions",
    [
        (torch.ones(4, dtype=torch.long), torch.arange(4).unsqueeze(0)),
        (torch.ones((1, 4), dtype=torch.float32), torch.arange(4).unsqueeze(0)),
        (torch.ones((1, 4), dtype=torch.long), torch.arange(3).unsqueeze(0)),
    ],
)
def test_packed_batch_rejects_invalid_shapes_and_dtype(tokens, positions):
    with pytest.raises(ValueError):
        dmodel_forward_kwargs({"tokens": tokens, "position_ids": positions})
