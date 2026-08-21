from pathlib import Path

import pytest
import yaml

from runtime.config import load_registry
from runtime.devices import Call, Device, MenuItem, MockDevice
from runtime.devices.registry import DeviceRegistry


@pytest.mark.asyncio
async def test_mock_device_records_hover_and_click():
    dev = MockDevice("lamp")

    await dev.on_hover(True)
    await dev.on_click("left")
    await dev.on_click("right")
    await dev.on_hover(False)

    assert dev.calls == [
        Call("hover", (True,)),
        Call("click", ("left",)),
        Call("click", ("right",)),
        Call("hover", (False,)),
    ]


@pytest.mark.asyncio
async def test_mock_device_context_menu_and_action():
    menu = [MenuItem("Color…", "color"), MenuItem("Brightness…", "bright")]
    dev = MockDevice("lamp", menu=menu)

    items = await dev.on_context_menu()
    await dev.invoke_menu_action("color", "#ff0000")

    assert items == menu
    assert dev.calls[-1] == Call("menu", ("color", "#ff0000"))


@pytest.mark.asyncio
async def test_registry_dispatches_to_correct_device():
    registry = DeviceRegistry()
    a = MockDevice("a")
    b = MockDevice("b")
    registry.add(a)
    registry.add(b)

    await registry.dispatch_click("a", "left")
    await registry.dispatch_click("b", "right")
    await registry.dispatch_hover("a", True)

    assert [c.kind for c in a.calls] == ["click", "hover"]
    assert [c.kind for c in b.calls] == ["click"]
    assert b.calls[0].args == ("right",)


def test_registry_rejects_duplicate_object_id():
    registry = DeviceRegistry()
    registry.add(MockDevice("dup"))
    with pytest.raises(ValueError, match="duplicate"):
        registry.add(MockDevice("dup"))


@pytest.mark.asyncio
async def test_registry_unknown_object_id():
    registry = DeviceRegistry()
    with pytest.raises(KeyError, match="missing"):
        await registry.dispatch_click("missing", "left")


def test_registry_unknown_factory_type():
    registry = DeviceRegistry()
    with pytest.raises(ValueError, match="unknown device type"):
        registry.build("x", "nonexistent_type", {})


def test_load_registry_missing_file_returns_empty(tmp_path: Path):
    registry = load_registry(tmp_path / "nope.yaml")
    assert len(registry) == 0


def test_load_registry_none_returns_empty():
    assert len(load_registry(None)) == 0


@pytest.mark.asyncio
async def test_load_registry_parses_mock_devices(tmp_path: Path):
    cfg = tmp_path / "devices.yaml"
    cfg.write_text(
        yaml.safe_dump(
            [
                {"object_id": "switch_left", "type": "mock", "label": "Left switch"},
                {"object_id": "motor", "type": "mock"},
            ]
        )
    )
    registry = load_registry(cfg)

    assert len(registry) == 2
    assert "switch_left" in registry
    assert registry.get("switch_left").label == "Left switch"
    assert registry.get("motor").label == "motor"  # falls back to object_id

    await registry.dispatch_click("switch_left", "left")
    dev = registry.get("switch_left")
    assert isinstance(dev, MockDevice)
    assert dev.calls[0].kind == "click"


@pytest.mark.parametrize(
    "raw,match",
    [
        # A top-level scalar is neither a list nor the {devices: [...]} mapping.
        ("42", "expected a list of device entries or a mapping"),
        # Mapping form is now valid, but `devices` must be a list.
        ("devices: 42", "'devices' must be a list"),
        ("- type: mock", "missing required field 'object_id'"),
        ("- object_id: x", "missing required field 'type'"),
        ("- 42", "must be a mapping"),
    ],
)
def test_load_registry_validation_errors(tmp_path: Path, raw: str, match: str):
    cfg = tmp_path / "devices.yaml"
    cfg.write_text(raw)
    with pytest.raises(ValueError, match=match):
        load_registry(cfg)


def test_load_registry_accepts_mapping_form(tmp_path: Path):
    """The {scene: {...}, devices: [...]} mapping form builds devices the same
    as a bare list, and load_scene_settings reads the scene block."""
    from src.runtime.config import load_scene_settings

    cfg = tmp_path / "devices.yaml"
    cfg.write_text(
        "scene:\n"
        "  flame_carry: true\n"
        "devices:\n"
        "  - object_id: switch_left\n"
        "    type: mock\n"
    )
    reg = load_registry(cfg)
    assert [d.object_id for d in reg] == ["switch_left"]
    assert load_scene_settings(cfg) == {"flame_carry": True}


def test_load_scene_settings_legacy_list_is_empty(tmp_path: Path):
    """A bare-list devices.yaml (no scene block) yields no scene flags."""
    from src.runtime.config import load_scene_settings

    cfg = tmp_path / "devices.yaml"
    cfg.write_text("- object_id: x\n  type: mock\n")
    assert load_scene_settings(cfg) == {}


def test_device_is_abstract():
    with pytest.raises(TypeError):
        Device("x")  # type: ignore[abstract]


# ---------- --mock flag (load_registry mock=True) ----------

@pytest.mark.asyncio
async def test_mock_flag_rewrites_real_types_to_mock(tmp_path: Path):
    """The whole point: a YAML that would otherwise build a UDP/Kasa/HTTP
    device gets every entry rewritten to MockDevice. Same object_ids,
    no real hardware fired."""
    cfg = tmp_path / "devices.yaml"
    cfg.write_text(
        yaml.safe_dump(
            [
                {
                    "object_id": "water_pump",
                    "type": "udp",
                    "label": "Water pump",
                    "config": {"host": "10.0.0.51", "port": 5006, "fill_cup_s": 24.0},
                },
                {
                    "object_id": "lamp_switch",
                    "type": "kasa",
                    "label": "Lamp switch",
                    "config": {"host_env": "X", "username_env": "Y", "password_env": "Z"},
                },
            ]
        )
    )
    registry = load_registry(cfg, mock=True)

    assert len(registry) == 2
    assert "water_pump" in registry and "lamp_switch" in registry

    pump = registry.get("water_pump")
    switch = registry.get("lamp_switch")
    assert isinstance(pump, MockDevice), f"expected MockDevice, got {type(pump).__name__}"
    assert isinstance(switch, MockDevice), f"expected MockDevice, got {type(switch).__name__}"
    # Labels survive the rewrite.
    assert pump.label == "Water pump"
    assert switch.label == "Lamp switch"

    # Clicks must be recorded but not actually sent anywhere.
    await registry.dispatch_click("water_pump", "left")
    assert pump.calls[0].kind == "click"


def test_mock_flag_off_still_builds_real_types(tmp_path: Path):
    """Default (mock=False) path is unchanged: real factories run."""
    cfg = tmp_path / "devices.yaml"
    cfg.write_text(
        yaml.safe_dump(
            [
                {
                    "object_id": "water_pump",
                    "type": "udp",
                    "config": {"host": "127.0.0.1"},
                },
            ]
        )
    )
    registry = load_registry(cfg, mock=False)
    from runtime.devices.udp_device import UdpDevice

    assert isinstance(registry.get("water_pump"), UdpDevice)


def test_mock_flag_skips_factory_validation(tmp_path: Path):
    """A YAML missing required real-type config (e.g. http with no base_url)
    would normally error from the factory. With mock=True the type is
    rewritten before the factory sees it, so the YAML stays loadable —
    handy if you want to dry-run a half-configured devices file."""
    cfg = tmp_path / "devices.yaml"
    cfg.write_text(
        yaml.safe_dump(
            [
                {"object_id": "x", "type": "http", "config": {}},  # missing base_url
            ]
        )
    )
    # Without mock: factory rejects.
    with pytest.raises(ValueError, match="missing required config.base_url"):
        load_registry(cfg, mock=False)
    # With mock: builds fine as MockDevice.
    registry = load_registry(cfg, mock=True)
    assert isinstance(registry.get("x"), MockDevice)


def test_mock_flag_still_requires_object_id_and_type(tmp_path: Path):
    """Even in mock mode, structural YAML errors must still raise — we're
    only rewriting `type`, not skipping schema validation."""
    cfg = tmp_path / "devices.yaml"
    cfg.write_text("- type: udp\n")  # missing object_id
    with pytest.raises(ValueError, match="missing required field 'object_id'"):
        load_registry(cfg, mock=True)


@pytest.mark.asyncio
async def test_default_invoke_menu_action_raises():
    class Bare(Device):
        async def on_click(self, button):
            return None

    dev = Bare("x")
    with pytest.raises(NotImplementedError):
        await dev.invoke_menu_action("anything")


@pytest.mark.asyncio
async def test_close_all_calls_each_device():
    closed: list[str] = []

    class Recorder(Device):
        async def on_click(self, button):
            return None

        async def close(self):
            closed.append(self.object_id)

    registry = DeviceRegistry()
    registry.add(Recorder("a"))
    registry.add(Recorder("b"))
    await registry.close_all()

    assert sorted(closed) == ["a", "b"]
