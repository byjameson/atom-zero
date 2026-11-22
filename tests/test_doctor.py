import socket
from pathlib import Path

from atomzero.doctor import check_cache, check_port, render


def test_cache_exact_sizes(tmp_path: Path):
    (tmp_path / "model.pt").write_bytes(b"x" * 10)
    (tmp_path / "config.json").write_bytes(b"{}")
    checks = check_cache(tmp_path, {"model.pt": 10, "config.json": 2})
    assert [check.status for check in checks] == ["ok", "ok"]


def test_cache_missing_and_wrong_size(tmp_path: Path):
    (tmp_path / "model.pt").write_bytes(b"x" * 3)
    checks = check_cache(tmp_path, {"model.pt": 10, "config.json": 2})
    assert checks[0].status == "fail"
    assert "expected 10" in checks[0].detail
    assert checks[1].status == "fail"
    assert "missing" in checks[1].detail


def test_free_port_is_ok():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert check_port("127.0.0.1", port).status == "ok"


def test_render_summarizes():
    checks = [
        check_cache(Path("does-not-exist"), {"model.pt": 10})[0],
        check_port("127.0.0.1", 0),
    ]
    text = render(checks)
    assert "FAIL" in text
    assert "summary: 1 fail" in text
