"""CPU plans for complete ready groups; use the existing Miles packing primitives."""

from miles.utils.seqlen_balancing import (
    expand_bins_by_splitting,
    first_fit_decreasing_pack,
    first_fit_pack,
    get_seqlen_balanced_partitions,
)


def select_ready(rows, *, budget, remaining):
    # When everything is ready, keep one native-sized window. Otherwise start
    # useful work now; waiting for a fuller bin would recreate the original bubble.
    if len(rows) == remaining:
        return rows
    selected, tokens = [], 0
    for row in rows:
        length = sum(row["lengths"])
        if selected and tokens + length > budget:
            break
        selected.append(row)
        tokens += length
    return selected


def plan_microbatches(lengths, *, capacity, native_layout):
    if not lengths or min(lengths) <= 0 or max(lengths) > capacity:
        raise ValueError("Empty window or sample outside max_tokens_per_gpu")
    count = len(first_fit_pack(lengths, capacity) if native_layout else first_fit_decreasing_pack(lengths, capacity))
    if native_layout:
        bins = get_seqlen_balanced_partitions(lengths, count, equal_size=False)
        if all(sum(lengths[i] for i in b) <= capacity for b in bins):
            return bins
    # Balance the same linear + quadratic length cost used in the replay.
    bins = [[] for _ in range(count)]
    used, cost = [0] * count, [0.0] * count
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        length = lengths[index]
        choices = [j for j in range(count) if used[j] + length <= capacity]
        if not choices:
            bins = first_fit_decreasing_pack(lengths, capacity)
            expand_bins_by_splitting(bins, count, lengths)
            break
        target = min(choices, key=lambda j: (cost[j], j))
        bins[target].append(index)
        used[target] += length
        cost[target] += length + length * length / 4096
    if sorted(i for b in bins for i in b) != list(range(len(lengths))) or not all(bins):
        raise ValueError("Window planner lost or duplicated samples")
    if any(sum(lengths[i] for i in b) > capacity for b in bins):
        raise ValueError("Window exceeds the microbatch token capacity")
    return [sorted(b) for b in bins]
