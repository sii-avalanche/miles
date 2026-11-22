from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss.hub import logit_processors


class TestGetLogProbsAndEntropy:
    def test_stored_log_probs_forward_the_debug_grad_switch_from_training_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fused_cross_entropy_calls: list[tuple[bool, bool]],
    ) -> None:
        """Stored log probabilities forward the debug grad contract from training arguments."""
        parallel_state = SimpleNamespace(
            tp=SimpleNamespace(rank=0, group=None),
            cp=SimpleNamespace(rank=0, size=1),
        )
        monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
        args = Namespace(
            qkv_format="thd",
            rollout_temperature=1.0,
            true_on_policy_mode=False,
            log_probs_chunk_size=-1,
            allgather_cp=False,
            debug_unified_grad_fused_logprob=True,
        )

        with torch.no_grad():
            result = logit_processors.get_log_probs_and_entropy(
                torch.randn(1, 4, 8),
                args=args,
                unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
                total_lengths=[4],
                response_lengths=[3],
            )

        assert fused_cross_entropy_calls == [(True, True)]
        assert not result["log_probs"][0].requires_grad

    def test_policy_score_source_uses_common_loss_contract_without_dense_logits(self):
        class Source:
            def __init__(self):
                self.device = torch.device("cpu")
                self.weight = torch.tensor(2.0, requires_grad=True)
                self.calls = []

            def sum(self, *, dtype=None):
                return self.weight.to(dtype=dtype).sum()

            def get_log_probs_and_entropy(self, **kwargs):
                self.calls.append(kwargs)
                scores = self.weight * torch.arange(3, dtype=torch.float32)
                return {
                    "log_probs": [scores],
                    "entropy": [scores + 1],
                }

        source = Source()
        args = Namespace(qkv_format="thd")
        sampling_mask = [SimpleNamespace(marker="sampling")]

        result = logit_processors.get_log_probs_and_entropy(
            source,
            args=args,
            unconcat_tokens=[torch.tensor([0, 1, 2, 3])],
            total_lengths=[4],
            response_lengths=[3],
            with_entropy=True,
            entropy_requires_grad=True,
            rollout_sampling_mask=sampling_mask,
        )

        assert len(source.calls) == 1
        assert source.calls[0]["args"] is args
        assert source.calls[0]["rollout_sampling_mask"] is sampling_mask
        torch.testing.assert_close(result["log_probs"][0], torch.tensor([0.0, 2.0, 4.0]))
        result["log_probs"][0].sum().backward()
        assert source.weight.grad.item() == 3.0
