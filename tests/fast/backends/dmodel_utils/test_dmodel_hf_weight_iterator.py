"""CPU contract tests; only the CUDA gather primitive is substituted."""

import gc
import json
import weakref
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from dmodel.models.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from dmodel.models.deepseek_v4_flash.model import DeepSeekV4FlashModel
from safetensors import SafetensorError
from safetensors.torch import save_file

from miles.backends.dmodel_utils import hf_weight_iterator as impl
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement


def _public(config, state):
    # Independent public naming oracle, not adapter.to_hf or the code under test.
    public = {}

    def add(name, source):
        public[name] = state[source]

    for name, source in (
        ("model.embed_tokens.weight", "tok_embeddings.weight"),
        ("lm_head.weight", "output.weight"),
        ("model.norm.weight", "norm.weight"),
    ):
        add(name, source)
    for kind in ("base", "fn", "scale"):
        add(f"model.hc_head_{kind}", f"hc_head.hc_{kind}")
    for layer, ratio in enumerate(config.dsv4_compress_ratios):
        h, s = f"model.layers.{layer}", f"layers.{layer}"
        for name, source in (("input_layernorm", "attention_norm"), ("post_attention_layernorm", "ffn_norm")):
            add(f"{h}.{name}.weight", f"{s}.{source}.weight")
        for site in ("attn", "ffn"):
            for kind in ("base", "fn", "scale"):
                add(f"{h}.hc_{site}_{kind}", f"{s}.{site}_hc.{kind}")
        for key in (
            "wq_a.weight",
            "wq_b.weight",
            "q_norm.weight",
            "wkv.weight",
            "kv_norm.weight",
            "wo_b.weight",
            "attn_sink",
        ):
            add(f"{h}.self_attn.{key}", f"{s}.attention.{key}")
        public[f"{h}.self_attn.wo_a.weight"] = state[f"{s}.attention.wo_a"].reshape(
            config.dsv4_o_groups * config.dsv4_o_lora_rank, -1
        )
        if ratio:
            for key in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                add(f"{h}.self_attn.compressor.{key}", f"{s}.attention.compressor.{key}")
        if ratio == 4:
            for key, source in (("wq_b", "linear_wq_b"), ("weights_proj", "linear_weights_proj")):
                add(f"{h}.self_attn.indexer.{key}.weight", f"{s}.attention.indexer.{source}.weight")
            for key in ("ape", "wkv.weight", "wgate.weight", "norm.weight"):
                add(f"{h}.self_attn.indexer.compressor.{key}", f"{s}.attention.indexer.compressor.{key}")
        add(f"{h}.mlp.gate.weight", f"{s}.moe.router.gate.weight")
        if layer < config.num_hash_layers:
            add(f"{h}.mlp.topk.tid2eid", f"{s}.moe.router.tid2eid")
        else:
            add(f"{h}.mlp.gate.e_score_correction_bias", f"{s}.moe.expert_bias")
        for projection, native in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
            add(f"{h}.mlp.shared_experts.{projection}.weight", f"{s}.moe.shared_experts.{projection}.weight")
            for expert in range(config.num_experts):
                public[f"{h}.mlp.experts.{expert}.{projection}.weight"] = state[f"{s}.moe.experts.{native}"][expert]
    return public


@pytest.fixture(scope="module", params=[(43, 3), (4, 1)])
def policy(request):
    depth, hash_layers = request.param
    config = DeepSeekV4FlashConfig(
        max_seq_len=16,
        vocab_size=7,
        dim=8,
        n_layers=depth,
        n_heads=2,
        q_lora_rank=4,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        dsv4_window_size=4,
        dsv4_o_groups=2,
        dsv4_o_lora_rank=2,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=2,
        moe_inter_dim=4,
        num_experts=2,
        num_experts_per_tok=2,
        num_hash_layers=hash_layers,
    )
    with torch.device("meta"):
        schema = DeepSeekV4FlashModel(config).state_dict()
    state = {}
    for index, (name, meta) in enumerate(schema.items()):
        value = torch.arange(meta.numel()).reshape(meta.shape) % 127
        if name.endswith("tid2eid"):
            value = value % config.num_experts
        else:
            keep_fp32 = any(
                part in name for part in ("hc_", "_hc.", "compressor.", "router.gate", "expert_bias", "attn_sink")
            )
            value = (value + index % 11).to(torch.float32 if keep_fp32 else torch.bfloat16)
        state[name] = value
    public = _public(config, state)
    return SimpleNamespace(config=config, state=state, public=public)



def _checkpoint(path, public, *, mtp=True):
    path.mkdir()
    tensors = {name: value.clone().contiguous() for name, value in public.items()}
    if mtp:
        tensors["model.mtp.0.enorm.weight"] = torch.ones(8, dtype=torch.float32)
    save_file(tensors, path / "model-00001-of-00001.safetensors")
    index = {"weight_map": dict.fromkeys(tensors, "model-00001-of-00001.safetensors")}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    return path


@pytest.fixture
def setup(policy, tmp_path):
    with torch.device("meta"):
        model = DeepSeekV4FlashModel(deepcopy(policy.config))
    # Native training keeps FP32 masters even when initial HF payloads are BF16.
    state = {
        name: value.float() if value.is_floating_point() else value.clone() for name, value in policy.state.items()
    }
    model.load_state_dict(state, assign=True)
    args = Namespace(
        hf_checkpoint=str(_checkpoint(tmp_path / "hf", policy.public)),
        update_weight_buffer_size=256,
        dmodel_publish_max_tensor_mb=8192,
    )
    return model, args


def _build(setup, **kwargs):
    model, args = setup
    return impl.DModelHfWeightIterator.build(
        args,
        model,
        required_placement=WeightUpdatePlacement(gather_pp=False),
        model_name="DeepSeekV4Config",
        quantization_config=kwargs.pop("quantization_config", None),
        **kwargs,
    )


def _gather_tracker(monkeypatch):
    refs, calls = [], []

    def gather(value, *, async_op):
        assert async_op is True
        assert sum(ref() is not None for ref in refs) <= 1, "more than two global sources in flight"
        full = value.detach().clone()
        refs.append(weakref.ref(full))
        calls.append((tuple(value.shape), value.dtype))
        return full

    monkeypatch.setattr(impl, "gather_full_param", gather)
    return refs, calls


def test_public_iterator_native_inventory_atomic_pairs_and_preserved_values(setup, policy, monkeypatch):
    refs, calls = _gather_tracker(monkeypatch)
    iterator = _build(setup)
    assert iterator.placement.is_full_gather and iterator.weight_update_selector == "target"
    before = {name: (value.data_ptr(), value.clone()) for name, value in setup[0].state_dict().items()}
    seen, expert_order = set(), {}
    for bucket in iterator.iter_hf_weights(None):
        names = {name for name, _ in bucket}
        for group in iterator._hf_atomic_update_groups():
            for name in names:
                for suffix in group.suffixes:
                    if name.endswith(suffix):
                        assert all(name.removesuffix(suffix) + partner in names for partner in group.suffixes)
        for name, value in bucket:
            assert name not in seen and name in policy.public
            expected = policy.public[name]
            if expected.is_floating_point():
                expected = expected.float()
            assert value.dtype == expected.dtype and torch.equal(value, expected)
            assert value.is_contiguous() and not value.requires_grad
            seen.add(name)
            if ".experts." in name:
                prefix, tail = name.split(".experts.")
                expert, projection = tail.split(".", 1)
                expert_order.setdefault((prefix, projection), []).append(int(expert))
    del bucket, value
    gc.collect()
    assert seen == policy.public.keys()
    assert all(order == list(range(2)) for order in expert_order.values())
    assert len(calls) == len(before) and all(ref() is None for ref in refs)
    for name, value in setup[0].state_dict().items():
        assert value.device.type == "cpu" and value.data_ptr() == before[name][0]
        assert torch.equal(value, before[name][1])


def test_nonsender_joins_every_gather_without_materializing(setup, monkeypatch):
    refs, calls = _gather_tracker(monkeypatch)
    iterator = _build(setup)
    assert list(iterator.iter_hf_weights(None, materialize=False)) == []
    assert len(calls) == len(setup[0].state_dict())
    assert all(ref() is None for ref in refs)


def test_ceiling_before_first_gather(setup, monkeypatch):
    setup[1].dmodel_publish_max_tensor_mb = 0
    monkeypatch.setattr(impl, "gather_full_param", lambda *_: pytest.fail("admission gathered"))
    with pytest.raises(ValueError, match="ceiling"):
        _build(setup)


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "wrong_shard", "traversal", "shape", "integer_dtype", "fp32_demotion"]
)
def test_independent_header_contract_rejects_damage(setup, policy, damage):
    root = impl.Path(setup[1].hf_checkpoint)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = f"model.layers.{policy.config.n_layers - 1}.mlp.experts.1.gate_proj.weight"
    tensors = {name: value.clone().contiguous() for name, value in policy.public.items()}
    if damage == "missing":
        del index["weight_map"][key]
    elif damage == "extra":
        index["weight_map"]["model.extra.weight"] = index["weight_map"][key]
    elif damage == "wrong_shard":
        save_file({"unused": torch.ones(1)}, root / "other.safetensors")
        index["weight_map"][key] = "other.safetensors"
    elif damage == "traversal":
        index["weight_map"][key] = "../other.safetensors"
    else:
        if damage == "shape":
            tensors[key] = torch.zeros(1, 1, dtype=torch.bfloat16)
        elif damage == "integer_dtype":
            tensors["model.layers.0.mlp.topk.tid2eid"] = tensors["model.layers.0.mlp.topk.tid2eid"].int()
        else:
            setup[0].layers["0"].attention.attn_sink.data = setup[0].layers["0"].attention.attn_sink.data.bfloat16()
        save_file(tensors, root / "model-00001-of-00001.safetensors")
        index["weight_map"] = dict.fromkeys(tensors, "model-00001-of-00001.safetensors")
    index_path.write_text(json.dumps(index))
    with pytest.raises((ValueError, KeyError, FileNotFoundError, SafetensorError)):
        _build(setup)


def test_close_releases_current_gather_and_failed_gather_does_not_complete(setup, monkeypatch):
    refs, _ = _gather_tracker(monkeypatch)
    iterator = _build(setup)
    stream = iterator.iter_hf_weights(None)
    bucket = next(stream)
    stream.close()
    assert all(ref() is None for ref in refs)


def test_unsupported_modes_and_empty_publication(setup, monkeypatch):
    with pytest.raises(ValueError, match="unquantized"):
        _build(setup, quantization_config={"quant_method": "fp8"})
    iterator = _build(setup)
    monkeypatch.setattr(impl, "gather_full_param", lambda *_: pytest.fail("empty publication gathered"))
    assert list(iterator.iter_hf_weights(None, include_base=False)) == []
    for args in ({"weights": {}}, {"weights": None, "adapters": [("a", object())]}):
        with pytest.raises(ValueError, match="live native"):
            list(iterator.iter_hf_weights(**args))


def test_mixed_dtype_buckets_support_native_raw_byte_reconstruction(setup, monkeypatch):
    _gather_tracker(monkeypatch)
    setup[1].update_weight_buffer_size = 512 << 20
    iterator = _build(setup)
    for bucket in iterator.iter_hf_weights(None):
        # SGL FlattenedTensorBucket concatenates raw bytes without padding.
        flat = torch.cat([value.flatten().view(torch.uint8) for _, value in bucket])
        offset = 0
        for name, value in bucket:
            restored = flat[offset : offset + value.nbytes].view(value.dtype).reshape(value.shape)
            assert torch.equal(restored, value), name
            offset += value.nbytes

@pytest.mark.parametrize("tied", [False, True])
def test_dense_qwen_checkpoint_roundtrip(tmp_path, monkeypatch, tied):
    from dmodel.models.qwen3.config import Qwen3Config
    from dmodel.models.qwen3.model import Qwen3ForCausalLM
    from dmodel.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

    config = Qwen3Config(hidden_size=8, intermediate_size=16, n_layers=2, n_heads=2, n_kv_heads=1, vocab_size=16,
                         tie_word_embeddings=tied)
    model = Qwen3ForCausalLM(config)
    expected = Qwen3StateDictAdapter(config).to_hf(model.state_dict())
    if tied:
        expected.pop("lm_head.weight", None)
        assert model.embed_tokens.weight is model.lm_head.weight
    root = _checkpoint(tmp_path / "hf", expected, mtp=False)
    args = Namespace(hf_checkpoint=root, update_weight_buffer_size=127, dmodel_publish_max_tensor_mb=1)
    iterator = impl.DModelHfWeightIterator.build(
        args, model, required_placement=WeightUpdatePlacement(gather_pp=False),
        model_name="Qwen3Config", quantization_config=None,
    )
    _gather_tracker(monkeypatch)
    published = dict(pair for bucket in iterator.iter_hf_weights(None) for pair in bucket)
    assert published.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(published[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("overlap", [False, True])
def test_publication_lookahead_respects_byte_limit_and_drains_on_close(setup, monkeypatch, overlap):
    iterator = _build(setup)
    state = impl.get_model_state_dict(iterator.model)
    iterator.sources = iterator.sources[:2]
    first, second = iterator.sources
    iterator.max_source_bytes = state[first].nbytes + state[second].nbytes - int(not overlap)
    handles = []

    class Handle:
        def __init__(self, value):
            self.value = value
            self.waits = 0
            handles.append(self)

        def wait(self):
            self.waits += 1
            return self.value

    monkeypatch.setattr(impl, "AsyncCollectiveTensor", Handle)
    monkeypatch.setattr(impl, "gather_full_param", lambda value, *, async_op: Handle(value))
    stream = iterator._iter_hf_param_units(None, materialize=True)
    next(stream)
    assert [handle.waits for handle in handles] == ([1, 0] if overlap else [1])
    stream.close()
    assert all(handle.waits == 1 for handle in handles)
