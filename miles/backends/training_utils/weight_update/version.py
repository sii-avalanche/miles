"""Verify the policy version reported by a set of inference engines."""

import asyncio


async def acknowledge_versions(engines, expected, *, timeout=60):
    """Every engine must acknowledge the completed native publication."""
    if type(expected) is not int or expected <= 0 or not engines:
        raise ValueError("Publication requires engines and a positive integer version")
    versions = await asyncio.wait_for(
        asyncio.gather(*(engine.get_weight_version() for engine in engines)), timeout=timeout
    )
    if any(str(version) != str(expected) for version in versions):
        raise RuntimeError(f"Weight version mismatch: expected {expected}, engines reported {versions}")
    return dict(version=expected, engines=len(versions), versions=[str(v) for v in versions])
