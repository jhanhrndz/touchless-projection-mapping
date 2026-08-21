"""Run the display-lister's own self-test as part of `pytest`.

scripts/ isn't on the import path (conftest only adds src/), so this loads the
script by path. Without this the self-test exists but nothing ever runs it —
and AGENTS.md tells people `uv run pytest` is the whole check.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "list_displays.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("list_displays", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Must be in sys.modules *before* exec: the script uses `from __future__
    # import annotations`, so @dataclass resolves its string annotations by
    # looking its own module up there. Without this it dies on import.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[spec.name]
        raise
    yield module
    del sys.modules[spec.name]


def test_script_exists():
    """AGENTS.md and the README both send people here by exact path."""
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


def test_self_test_passes(mod, capsys):
    assert mod.run_self_test() == 0
    assert "self-test OK" in capsys.readouterr().out


def test_list_displays_never_raises(mod):
    """Whatever platform CI runs on, this must degrade to [] rather than blow up.

    The caller's contract is 'empty means I couldn't tell you' — main() turns
    that into an actionable message and exit 1.
    """
    got = mod.list_displays()
    assert isinstance(got, list)
    for d in got:
        assert isinstance(d.name, str) and d.name


def test_unenumerable_platform_exits_nonzero(mod, monkeypatch, capsys):
    """Negative: an unknown OS must fail loudly with the manual fallback."""
    monkeypatch.setattr(mod.platform, "system", lambda: "Plan9")
    assert mod.main(["list_displays.py"]) == 1
    assert "--projector-size" in capsys.readouterr().err
