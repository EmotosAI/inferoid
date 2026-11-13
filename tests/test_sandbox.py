from pathlib import Path

from inferoid import sandbox


def test_profile_limits_writes_secrets_and_network(tmp_path):
    prof = sandbox.profile(tmp_path / "wt", home=tmp_path / "home")
    assert "(deny network*)" in prof and 'remote ip "localhost:*"' in prof
    assert 'local ip' not in prof  # combined with the unix-socket rule it let internet traffic through
    assert "(deny file-write*)" in prof and str((tmp_path / "wt").resolve()) in prof
    assert str(tmp_path / "home" / ".ssh") in prof and str(tmp_path / "home" / ".aws") in prof


def test_wrap_prefixes_sandbox_exec(tmp_path):
    args = sandbox.wrap(["claude", "-p", "x"], tmp_path)
    assert args[:2] == ["sandbox-exec", "-p"] and args[-3:] == ["claude", "-p", "x"]
