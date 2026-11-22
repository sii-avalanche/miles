"""CPU contracts for optional sample weights, without runtime mocks or process groups."""

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean, slice_log_prob_with_cp
from miles.backends.training_utils.data.rollout import DataIterator, _get_batch_data
from miles.backends.training_utils.loss.objective import loss_function
from miles.utils.ft_utils.process_group_utils import GroupInfo


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("explicit_denominators", [False, True])
def test_sample_mean_values_and_gradients(cp_size, weighted, explicit_denominators):
    state = make_parallel_state()
    masks = [torch.tensor([1, 0, 1]), torch.tensor([1, 1]), torch.tensor([0])]
    lengths = [6, 7, 2]
    responses = [3, 2, 1]
    weights = [0.75, 1.25, 0.0] if weighted else None
    denominators = torch.tensor([4, 5, 0]) if explicit_denominators else None
    values = torch.arange(1, 7, dtype=torch.float64, requires_grad=True)
    expected = sum(
        (1.0 if weights is None else weights[i])
        * (part * mask).sum()
        / (denominators[i] if denominators is not None else mask.sum()).clamp_min(1)
        for i, (part, mask) in enumerate(zip(values.split(responses), masks, strict=True))
    )
    actual = values.new_zeros(())
    for rank in range(cp_size):
        state.cp = GroupInfo(rank=rank, size=cp_size, group=None)
        local = torch.cat(
            [
                slice_log_prob_with_cp(part, total, response)
                for part, total, response in zip(values.split(responses), lengths, responses, strict=True)
            ]
        )
        reducer = get_sum_of_sample_mean(lengths, responses, masks, denominators=denominators, sample_weights=weights)
        actual = actual + reducer(local)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, values)[0], torch.autograd.grad(expected, values)[0])


def test_token_sum_default_is_unchanged_and_weights_are_rejected():
    make_parallel_state()
    masks = [torch.tensor([1, 0]), torch.tensor([1])]
    values = torch.tensor([2.0, 100.0, 3.0], requires_grad=True)
    reducer = get_sum_of_sample_mean([3, 2], [2, 1], masks, calculate_per_token_loss=True)
    assert reducer(values).item() == 5
    with pytest.raises(ValueError, match="require sample-mean"):
        get_sum_of_sample_mean([3, 2], [2, 1], masks, True, sample_weights=[0.5, 0.5])


def test_sample_weights_cannot_silently_truncate_samples():
    make_parallel_state()
    reducer = get_sum_of_sample_mean([2, 2], [1, 1], [torch.ones(1), torch.ones(1)], sample_weights=[1.0])
    with pytest.raises(ValueError, match="zip"):
        reducer(torch.ones(2))


@pytest.mark.parametrize("indexed", [False, True])
def test_batch_selection_preserves_unweighted_defaults(indexed):
    data = {
        "tokens": [[1, 2], [3, 4]],
        "rollout_mask_sums": [2, 7],
        "dynamic_global_batch_size": 8,
        "training_token_counts": {"model_input_tokens": 4},
    }
    iterator = DataIterator(data, micro_batch_indices=[[1, 0]]) if indexed else DataIterator(data, micro_batch_size=2)
    keys = ["tokens"]
    batch = _get_batch_data(iterator, keys)
    assert keys == ["tokens"]
    assert batch == {
        "tokens": list(reversed(data["tokens"])) if indexed else data["tokens"],
        "adapter_slots": None,
        "dynamic_global_batch_size": 8,
    }
    assert data["training_token_counts"] == {"model_input_tokens": 4}


@pytest.mark.parametrize("use_tis", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dummy_response_length", [1, 2])
def test_policy_dispatch_and_tis_keep_sample_mean_normalizer(use_tis, weighted, dummy_response_length):
    make_parallel_state()
    args = make_args(
        global_batch_size=99,
        use_dynamic_global_batch_size=True,
        use_tis=use_tis,
        entropy_coef=0.0,
        observe_training_entropy=False,
        eps_clip_c=None,
        dppo_delta=None,
    )
    responses = [2, 3, dummy_response_length]
    masks = [torch.tensor([1, 1]), torch.tensor([1, 0, 1]), torch.zeros(dummy_response_length)]
    tokens = [
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 0, 1]),
        torch.tensor([0, 2, 1])[: dummy_response_length + 1],
    ]
    total_length = sum(len(sample) for sample in tokens)
    logits = torch.linspace(-0.3, 0.4, total_length * 3).reshape(1, total_length, 3).requires_grad_(True)
    response_rows = [0, 1, 3, 4, 5] + list(range(7, 7 + dummy_response_length))
    labels = torch.cat([sample[1:] for sample in tokens]).unsqueeze(-1)
    selected = logits[0, response_rows].log_softmax(-1).gather(-1, labels).flatten()
    old = selected.detach() + 0.05
    advantages = torch.tensor([1.0, 3.0, 2.0, -1.0, 4.0] + [999.0] * dummy_response_length)
    batch = {
        "unconcat_tokens": tokens,
        "total_lengths": [len(sample) for sample in tokens],
        "response_lengths": responses,
        "loss_masks": masks,
        "rollout_mask_sums": torch.tensor([2, 2, 0]),
        "advantages": list(advantages.split(responses)),
        "log_probs": list(old.split(responses)),
        "dynamic_global_batch_size": 8,
    }
    weights = [0.4, 1.6, 0.0] if weighted else [1.0, 1.0, 1.0]
    if weighted:
        batch["sample_weights"] = weights
    if use_tis:
        batch["rollout_log_probs"] = list((old - 0.1).split(responses))
    expected_tokens = -advantages * (selected - old).exp()
    if use_tis:
        expected_tokens = expected_tokens * torch.exp(torch.tensor(0.1))
    expected = (
        sum(
            weight * (part * mask).sum() / mask.sum().clamp_min(1)
            for weight, part, mask in zip(weights, expected_tokens.split(responses), masks, strict=True)
        )
        / 8
    )
    actual, normalizer, _ = loss_function(args, batch, 3, logits)
    assert normalizer.item() == 1
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits)[0]
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, logits)[0])
    assert torch.count_nonzero(actual_grad[0, 7:]).item() == 0
