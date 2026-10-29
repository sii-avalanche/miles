"""Defer native token normalization across backward windows, without owning Adam."""


class DeferredFinalize:
    """Sum unnormalized gradients first; normalize by the full token count once."""

    def __init__(self, config):
        self.config = config
        self.original = config.finalize_model_grads_func
        if self.original is None or not config.calculate_per_token_loss:
            raise ValueError("Streaming requires native token-normalized gradient finalization")
        self.tokens = None
        self.last_call = None
        self.calls = 0

    def capture(self, models, tokens, **kwargs):
        if tokens is None:
            raise ValueError("Missing per-window token normalizer")
        self.tokens = tokens.clone() if self.tokens is None else self.tokens + tokens
        self.last_call = (models, kwargs)
        self.calls += 1

    def __enter__(self):
        self.config.finalize_model_grads_func = self.capture
        return self

    def __exit__(self, *_):
        self.config.finalize_model_grads_func = self.original

    def finish(self):
        if self.last_call is None:
            raise ValueError("No complete window; refuse an optimizer update")
        models, kwargs = self.last_call
        self.original(models, self.tokens, **kwargs)
