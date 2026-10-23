"""Translate AvaCore feedback into the Miles Sample wire payload after complete judging."""

from dataclasses import asdict

from ava_core.rewards.credit_resolution import CreditResolver, token_fingerprint
from ava_core.rewards.feedback import feedback_from_reward


def fill_credit_assignment(sample, trace, reward, context):
    feedback = feedback_from_reward(reward)
    start = len(sample.tokens) - sample.response_length
    mask = [0] * start + list(sample.loss_mask) + [0] * (len(trace.tokens) - len(sample.tokens))
    resolution = CreditResolver(context).resolve(feedback.credits, trace, loss_mask=mask)
    sample.credit_assignment = {
        "version": 1,
        "reward": reward.score,
        "annotations": [asdict(credit) for credit in feedback.credits],
        "spans": [{**asdict(span), "start": span.start - start, "end": span.end - start} for span in resolution.spans],
        "ignored": [*feedback.metadata.get("credit_errors", []), *resolution.ignored],
        "token_hash": token_fingerprint(sample.tokens),
        "response_length": sample.response_length,
    }
