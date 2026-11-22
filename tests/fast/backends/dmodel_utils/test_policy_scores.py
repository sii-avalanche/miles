from argparse import Namespace

import pytest
import torch
from dmodel.training.chunked_loss import ChunkedCrossEntropyOutputHead

from miles.backends.dmodel_utils.policy_scores import DModelPolicyScores
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.utils.ft_utils.process_group_utils import GroupInfo


def _parallel_state() -> ParallelState:
    one = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        intra_dp=one,
        intra_dp_cp=one,
        cp=one,
        tp=one,
        pp=one,
        ep=one,
        etp=one,
        indep_dp=one,
    )


def _cp_parallel_state(rank: int) -> ParallelState:
    one = GroupInfo(rank=0, size=1, group=None)
    cp = GroupInfo(rank=rank, size=2, group=None)
    return ParallelState(
        intra_dp=one,
        intra_dp_cp=one,
        cp=cp,
        tp=one,
        pp=one,
        ep=one,
        etp=one,
        indep_dp=one,
    )


def _args() -> Namespace:
    return Namespace(
        qkv_format="thd",
        true_on_policy_mode=False,
        rollout_temperature=0.8,
        log_probs_chunk_size=-1,
        vocab_size=9,
        debug_unified_grad_fused_logprob=False,
        loss_type="policy_loss",
        allgather_cp=False,
        bf16=False,
        fp16=False,
    )


def _scores(hidden, head):
    return get_log_probs_and_entropy(
        DModelPolicyScores(hidden, head, chunk_tokens=2),
        args=_args(),
        unconcat_tokens=[torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])],
        total_lengths=[4, 3],
        response_lengths=[2, 1],
        with_entropy=True,
    )


def test_chunked_policy_scores_match_dense_logits_and_gradients(monkeypatch):
    set_parallel_state(_parallel_state())

    def pure_scores(logits, tokens, _tp_group, **kwargs):
        scaled = logits.float() / kwargs["temperature"]
        log_probs = torch.log_softmax(scaled, dim=-1)
        selected = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1) if kwargs["with_entropy"] else None
        return selected, entropy

    monkeypatch.setattr(
        "miles.backends.dmodel_utils.policy_scores.calculate_log_probs_and_entropy",
        pure_scores,
    )
    torch.manual_seed(5)
    dense_hidden = torch.randn(1, 7, 4, requires_grad=True)
    chunked_hidden = dense_hidden.detach().clone().requires_grad_(True)
    dense_linear = torch.nn.Linear(4, 9, bias=False)
    chunked_linear = torch.nn.Linear(4, 9, bias=False)
    chunked_linear.load_state_dict(dense_linear.state_dict())
    chunked_head = ChunkedCrossEntropyOutputHead.from_linear(chunked_linear)

    dense_logits = dense_linear(dense_hidden).squeeze(0)
    dense_rows = dense_logits.index_select(0, torch.tensor([1, 2, 5]))
    dense_tokens = torch.tensor([3, 4, 7])
    dense_log_probs = torch.log_softmax(dense_rows.float() / 0.8, dim=-1)
    dense = {
        "log_probs": list(dense_log_probs.gather(-1, dense_tokens.unsqueeze(-1)).squeeze(-1).split([2, 1])),
        "entropy": list((-(dense_log_probs.exp() * dense_log_probs).sum(dim=-1)).split([2, 1])),
    }
    chunked = _scores(chunked_hidden, chunked_head)

    for name in ("log_probs", "entropy"):
        for actual, expected in zip(chunked[name], dense[name], strict=True):
            torch.testing.assert_close(actual, expected)

    sum(x.sum() for x in dense["log_probs"]).backward()
    sum(x.sum() for x in chunked["log_probs"]).backward()
    torch.testing.assert_close(chunked_hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(chunked_head.weight.grad, dense_linear.weight.grad)


def test_chunked_policy_scores_reject_response_at_first_token():
    set_parallel_state(_parallel_state())
    hidden = torch.randn(1, 2, 4)
    head = ChunkedCrossEntropyOutputHead.from_linear(torch.nn.Linear(4, 9, bias=False))
    source = DModelPolicyScores(hidden, head)
    try:
        source.get_log_probs_and_entropy(
            args=_args(),
            unconcat_tokens=[torch.tensor([1, 2])],
            total_lengths=[2],
            response_lengths=[2],
            with_entropy=False,
            entropy_requires_grad=True,
            non_loss_data=True,
            max_seq_lens=None,
            rollout_sampling_mask=None,
        )
    except ValueError as error:
        assert "first packed token" in str(error)
    else:
        raise AssertionError("expected response boundary rejection")


@pytest.mark.parametrize(("rank", "expected_local_rows"), [(0, 1), (1, 3)])
def test_policy_scores_selects_local_zigzag_rows_for_cp(rank, expected_local_rows, monkeypatch):
    set_parallel_state(_cp_parallel_state(rank))

    def pure_scores(logits, tokens, _tp_group, **kwargs):
        scaled = logits.float() / kwargs["temperature"]
        log_probs = torch.log_softmax(scaled, dim=-1)
        selected = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1) if kwargs["with_entropy"] else None
        return selected, entropy

    monkeypatch.setattr(
        "miles.backends.dmodel_utils.policy_scores.calculate_log_probs_and_entropy",
        pure_scores,
    )
    head = ChunkedCrossEntropyOutputHead.from_linear(torch.nn.Linear(4, 9, bias=False))
    scores = get_log_probs_and_entropy(
        DModelPolicyScores(torch.randn(1, 4, 4, requires_grad=True), head),
        args=_args(),
        unconcat_tokens=[torch.arange(8)],
        total_lengths=[8],
        response_lengths=[4],
        with_entropy=False,
    )
    assert scores["log_probs"][0].numel() == expected_local_rows


@pytest.mark.parametrize(("rank", "expected_sizes"), [(0, [1, 0]), (1, [3, 2])])
def test_policy_scores_offsets_packed_cp_samples(rank, expected_sizes, monkeypatch):
    set_parallel_state(_cp_parallel_state(rank))

    def pure_scores(logits, tokens, _tp_group, **kwargs):
        scaled = logits.float() / kwargs["temperature"]
        log_probs = torch.log_softmax(scaled, dim=-1)
        selected = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        return selected, None

    monkeypatch.setattr(
        "miles.backends.dmodel_utils.policy_scores.calculate_log_probs_and_entropy",
        pure_scores,
    )
    head = ChunkedCrossEntropyOutputHead.from_linear(torch.nn.Linear(4, 9, bias=False))
    scores = get_log_probs_and_entropy(
        DModelPolicyScores(torch.randn(1, 8, 4, requires_grad=True), head),
        args=_args(),
        unconcat_tokens=[torch.arange(8), torch.arange(6)],
        total_lengths=[8, 6],
        response_lengths=[4, 2],
        with_entropy=False,
    )
    assert [item.numel() for item in scores["log_probs"]] == expected_sizes


def test_projection_returns_autograd_tensors_and_omits_prompt_rows(monkeypatch):
    set_parallel_state(_parallel_state())
    head = ChunkedCrossEntropyOutputHead.from_linear(torch.nn.Linear(4, 9, bias=False))
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    projected_rows, backward_outputs = [], []

    def pure_scores(logits, tokens, _group, **kwargs):
        projected_rows.append(logits.shape[0])
        values = logits.log_softmax(-1)
        return values.gather(-1, tokens[:, None]).squeeze(-1), -(values.exp() * values).sum(-1)

    def observe_outputs(_module, _inputs, output):
        for results in output:
            for log_prob, entropy in results:
                assert isinstance(log_prob, torch.Tensor)
                log_prob.register_hook(lambda grad: backward_outputs.append(grad.shape))
        # Emulate a module wrapper replacing returned tensors. The scorer must
        # use these outputs, not the original tensors stored in a closure.
        return [[(log_prob * 2, entropy) for log_prob, entropy in results] for results in output]

    monkeypatch.setattr("miles.backends.dmodel_utils.policy_scores.calculate_log_probs_and_entropy", pure_scores)
    handle = head.register_forward_hook(observe_outputs)
    result = _scores(hidden, head)
    expected = (hidden[0, [1, 2, 5]] @ head.weight.T).log_softmax(-1)
    expected = expected.gather(-1, torch.tensor([[3], [4], [7]])).flatten() * 2
    torch.testing.assert_close(torch.cat(result["log_probs"]), expected)
    torch.cat(result["log_probs"]).sum().backward()
    handle.remove()
    assert projected_rows == [2, 1]
    assert len(backward_outputs) == 2
    assert head.weight.grad is not None


@pytest.mark.parametrize("entropy_requires_grad", [False, True])
def test_empty_local_response_retains_output_head_backward_edge(monkeypatch, entropy_requires_grad):
    set_parallel_state(_parallel_state())
    head = ChunkedCrossEntropyOutputHead.from_linear(torch.nn.Linear(4, 9, bias=False))
    hidden = torch.randn(1, 2, 4, requires_grad=True)
    source = DModelPolicyScores(hidden, head)
    seen = []

    def observe_outputs(_module, _inputs, output):
        output[0][0][0].register_hook(lambda grad: seen.append(grad.numel()))

    head.register_forward_hook(observe_outputs)
    result = source.get_log_probs_and_entropy(
        args=_args(),
        unconcat_tokens=[torch.tensor([1, 2])],
        total_lengths=[2],
        response_lengths=[0],
        with_entropy=True,
        entropy_requires_grad=entropy_requires_grad,
        non_loss_data=True,
        max_seq_lens=None,
        rollout_sampling_mask=None,
    )
    assert result["log_probs"][0].numel() == 0
    assert result["entropy"][0].requires_grad == entropy_requires_grad
    source.sum(dtype=torch.float32).backward()
    assert seen == [0]
    torch.testing.assert_close(head.weight.grad, torch.zeros_like(head.weight))
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))


def test_second_sample_cannot_predict_across_document_boundary():
    set_parallel_state(_parallel_state())
    source = DModelPolicyScores(
        torch.randn(1, 4, 4),
        ChunkedCrossEntropyOutputHead.from_linear(torch.nn.Linear(4, 9, bias=False)),
    )
    with pytest.raises(ValueError, match="first packed token of any sample"):
        source.get_log_probs_and_entropy(
            args=_args(),
            unconcat_tokens=[torch.tensor([1, 2]), torch.tensor([3, 4])],
            total_lengths=[2, 2],
            response_lengths=[1, 2],
            with_entropy=False,
            entropy_requires_grad=True,
            non_loss_data=True,
            max_seq_lens=None,
            rollout_sampling_mask=None,
        )


def test_true_on_policy_scales_before_bf16_conversion(monkeypatch):
    set_parallel_state(_parallel_state())
    args = _args()
    args.true_on_policy_mode = True
    args.bf16 = True
    hidden = torch.randn(1, 3, 4, requires_grad=True)
    head = ChunkedCrossEntropyOutputHead.from_linear(torch.nn.Linear(4, 9, bias=False))
    observed = []

    def record(logits, tokens, _group, **kwargs):
        observed.append(logits)
        assert kwargs["temperature"] == 1.0
        return logits.float().log_softmax(-1).gather(-1, tokens[:, None]).flatten(), None

    monkeypatch.setattr("miles.backends.dmodel_utils.policy_scores.calculate_log_probs_and_entropy", record)
    DModelPolicyScores(hidden, head).get_log_probs_and_entropy(
        args=args,
        unconcat_tokens=[torch.tensor([1, 2, 3])],
        total_lengths=[3],
        response_lengths=[2],
        with_entropy=False,
        entropy_requires_grad=False,
        non_loss_data=True,
        max_seq_lens=None,
        rollout_sampling_mask=None,
    )
    torch.testing.assert_close(observed[0], ((hidden[0, :2] @ head.weight.T) / 0.8).bfloat16())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native packed attention uses CUDA")
def test_dense_qwen_forward_and_projection_match_its_native_logits():
    from dmodel.models.attention.metadata import PackedSequenceMetadata
    from dmodel.models.qwen3.config import Qwen3Config
    from dmodel.models.qwen3.model import Qwen3ForCausalLM
    from dmodel.training.chunked_loss import replace_output_head_for_chunked_cross_entropy
    from miles.backends.dmodel_utils.step_runner import DModelStepRunner

    set_parallel_state(_parallel_state())
    config = Qwen3Config(hidden_size=128, intermediate_size=256, n_layers=2, n_heads=2, n_kv_heads=1,
                         max_seq_len=8, vocab_size=16, attention_backend="varlen")
    model = Qwen3ForCausalLM(config).cuda().bfloat16()
    model.init_weights()
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    positions = torch.arange(4, device="cuda").unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(tokens, varlen_metadata=PackedSequenceMetadata(document_positions=positions)).detach()
    head = replace_output_head_for_chunked_cross_entropy(model)
    runner = DModelStepRunner(model, None, None, head, max_grad_norm=1)
    batch = {"tokens": tokens, "position_ids": positions}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        (scores,) = runner.forward_only_step([batch], lambda scores, _: scores)
        (pieces,) = head.chunked_projection((scores.hidden.squeeze(0),), lambda _, __, logits: logits, chunk_tokens=2)
    logits = torch.cat(pieces)
    torch.testing.assert_close(logits, expected.squeeze(0))
    logits[-1].sum().backward()
    assert model.embed_tokens.weight.grad.abs().sum() > 0
