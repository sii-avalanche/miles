import os
import subprocess
import sys


def test_argument_preflight_does_not_import_actor_or_inference_runtime():
    code = """
import logging
import sys
level = logging.getLogger().level
from miles.backends.dmodel_utils.arguments import validate_dmodel_args
assert callable(validate_dmodel_args)
assert 'miles.backends.fsdp_utils.actor' not in sys.modules
assert 'miles.backends.dmodel_utils.actor' not in sys.modules
assert 'sglang' not in sys.modules
assert logging.getLogger().level == level
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
