import numpy as np

from miles.backends.training_utils.data.rollout import _copied_numpy_tensor


def test_copied_numpy_tensor_is_writable_and_does_not_alias_source() -> None:
    source = np.arange(12, dtype=np.int32).reshape(3, 4)
    source.setflags(write=False)

    tensor = _copied_numpy_tensor(source)

    tensor[0, 0] = 99
    assert tensor[0, 0].item() == 99
    assert source[0, 0] == 0
