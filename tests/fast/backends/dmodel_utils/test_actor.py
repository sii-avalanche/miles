from pathlib import Path

from miles.backends.dmodel_utils.actor import configure_worker_cache


def test_worker_cache_isolated_by_host_and_rank(tmp_path, monkeypatch):
    monkeypatch.setattr("socket.gethostname", lambda: "host-a")
    paths = configure_worker_cache(str(tmp_path), 7)

    assert set(paths) == {
        "TILELANG_CACHE_DIR",
        "TILELANG_TMP_DIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    }
    assert all("host-a-rank-007" in value for value in paths.values())
    assert all(Path(value).is_dir() for value in paths.values())
