import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_collection_modifyitems(config, items):
    import pytest

    if config.getoption("--hardware", default=False):
        return
    skip_hardware = pytest.mark.skip(reason="needs --hardware to run (camera/device required)")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)


def pytest_addoption(parser):
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="run tests that require a real camera or physical device",
    )
