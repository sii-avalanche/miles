"""Publish native parameter shards through the model family's HF checkpoint views."""

from pathlib import Path

import torch
from dmodel.models.registry import get_family, get_model_type_for_config
from dmodel.utils.checkpoint.conversion_utils import load_hf_weight_map
from dmodel.utils.checkpoint.hf_sharded import read_hf_weight_views
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import assemble_atomic_update_groups


class DModelHfWeightIterator(HfWeightIteratorBase):
    forced_placement = WeightUpdatePlacement(gather_pp=True)
    weight_update_selector = "target"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = getattr(self.model, "model_args", None) or self.model.config
        if self.quantization_config or getattr(getattr(config, "mtp", None), "enabled", False):
            raise ValueError("DModel publication requires unquantized inference and disabled MTP")
        family = get_model_type_for_config(config)
        adapter = get_family(family).adapter_cls(config)
        directory = Path(self.args.hf_checkpoint)
        headers, mappings = read_hf_weight_views(directory, load_hf_weight_map(directory), adapter)
        state = get_model_state_dict(self.model)
        self.max_source_bytes = self.args.dmodel_publish_max_tensor_mb << 20
        groups = get_hf_atomic_update_groups(family.replace("_", ""), q_lora_rank=getattr(config, "q_lora_rank", None))
        self.plan = {}
        names = {}
        for source, parts in mappings.items():
            param = state[source]
            if param.nbytes > self.max_source_bytes:
                raise ValueError(f"{source} requires {param.nbytes} bytes, ceiling {self.max_source_bytes}")
            experts = None not in parts
            if experts and set(parts) != set(range(param.shape[0])):
                raise ValueError(f"Incomplete expert inventory for {source}")
            self.plan[source] = []
            for expert, (header, mapping) in sorted(parts.items()):
                view = mapping.view
                shape = param.shape[1:] if experts else param.shape
                # A complete permuted or reshaped view can be inverted with one copy.
                axes = sorted(range(view.ndim), key=lambda axis: view.stride(axis), reverse=True)
                if (
                    view.shape != shape
                    or view.numel() != torch.Size(header.shape).numel()
                    or not view.permute(axes).is_contiguous()
                    or header.key in names
                ):
                    raise ValueError(f"Publication requires a complete one-source HF view for {header.key}")
                if param.dtype != view.dtype and (view.dtype, param.dtype) != (torch.bfloat16, torch.float32):
                    raise ValueError(f"Unsupported publication dtype transition for {header.key}")
                names[header.key] = source
                self.plan[source].append((expert, header, view))
        optional = set(adapter.hf_optional_buffers)
        missing = set(state) - self.plan.keys() - optional
        if missing:
            # HF may store a tied embedding and head under a single name.
            params = dict(self.model.named_parameters(remove_duplicate=False))
            represented = {id(params[key]) for key in self.plan.keys() & params.keys()}
            missing = {key for key in missing if id(params.get(key)) not in represented}
        ignored = getattr(adapter, "is_passthrough_hf_key", lambda key: False)
        unexpected = [key for key in headers.keys() - names.keys() if not ignored(key.removeprefix("model."))]
        if missing or unexpected:
            raise ValueError(f"Incomplete publication inventory: native={sorted(missing)}, HF={sorted(unexpected)}")
        # Order gathers on metadata so fused partners arrive together with bounded staging.
        self.groups = [g for g in groups if any(key.endswith(g.suffixes) for key in names)]
        blocks = assemble_atomic_update_groups(([(name, names[name])] for name in sorted(names)), self.groups)
        self.sources = tuple(dict.fromkeys(source for block in blocks for _, source in block))

    def _hf_atomic_update_groups(self):
        return self.groups

    def iter_hf_weights(self, weights, *, include_base=True, adapters=(), materialize=True):
        if weights is not None or adapters:
            raise ValueError("DModel publication reads live native state and does not support adapters")
        for bucket in super().iter_hf_weights(weights, include_base=include_base, materialize=materialize):
            # SGLang concatenates raw bytes without padding. Align larger elements first.
            bucket.sort(key=lambda pair: pair[1].element_size(), reverse=True)
            yield bucket

    def _iter_hf_param_units(self, weights, *, materialize):
        state = get_model_state_dict(self.model)
        pending = None
        try:
            for index, source in enumerate(self.sources):
                full = pending if pending is not None else gather_full_param(state[source], async_op=True)
                pending = None
                if isinstance(full, AsyncCollectiveTensor):
                    full = full.wait()
                if index + 1 < len(self.sources):
                    following = state[self.sources[index + 1]]
                    if state[source].nbytes + following.nbytes <= self.max_source_bytes:
                        pending = gather_full_param(following, async_op=True)
                if materialize:
                    for expert, header, view in self.plan[source]:
                        # Rebuild the HF layout by copying into the same view used for loading.
                        value = full.new_empty(header.shape)
                        value.as_strided(view.shape, view.stride()).copy_(full if expert is None else full[expert])
                        yield [(header.key, value)]
                del full
        finally:
            if isinstance(pending, AsyncCollectiveTensor):
                pending.wait()

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        raise NotImplementedError("DModel publication does not support LoRA")
