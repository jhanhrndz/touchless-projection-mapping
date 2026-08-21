"""Tests for KasaDevice — discovery is mocked so tests run without hardware.

Negative cases per project convention: missing host env, unknown menu action,
toggle on already-on / already-off paths covered. Plus the MAC-fallback paths:
cache hit / cache miss / MAC not found on LAN / MAC mismatch at cached IP.
"""

import sys
import types

import pytest

from runtime.devices.kasa_device import KasaDevice, _normalize_mac


class FakeKasaDevice:
    def __init__(self, is_on: bool = False, mac: str = ""):
        self.is_on = is_on
        self.mac = mac
        self.calls: list[str] = []

    async def update(self):
        self.calls.append("update")

    async def turn_on(self):
        self.calls.append("turn_on")
        self.is_on = True

    async def turn_off(self):
        self.calls.append("turn_off")
        self.is_on = False

    async def disconnect(self):
        self.calls.append("disconnect")


class _FakeDiscover:
    """Stub for kasa.Discover. Returns a FakeKasaDevice on discover_single.

    - `single_responses` maps host → FakeKasaDevice (or Exception). Default: returns
      the original `fake` for any host.
    - `discover_result` is what `discover()` (broadcast) returns. Defaults to
      {}; tests that want broadcast to find something set this explicitly."""

    def __init__(self, fake: FakeKasaDevice):
        self._fake = fake
        self.single_responses: dict = {}
        self.discover_result: dict = {}
        self.discover_calls: list[tuple] = []
        self.broadcast_calls: list[dict] = []

    async def discover_single(self, host, username=None, password=None):
        self.discover_calls.append((host, username, password))
        resp = self.single_responses.get(host, self._fake)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def discover(self, username=None, password=None, timeout=None):
        self.broadcast_calls.append(
            {"username": username, "password": password, "timeout": timeout}
        )
        return self.discover_result


@pytest.fixture
def fake_kasa(monkeypatch):
    """Replace `kasa` module's Discover with a FakeDiscover that returns a FakeKasaDevice."""
    fake = FakeKasaDevice(is_on=False)
    discover = _FakeDiscover(fake)
    fake_module = types.SimpleNamespace(Discover=discover)
    monkeypatch.setitem(sys.modules, "kasa", fake_module)
    monkeypatch.setenv("SMART_SWITCH_HOST", "10.0.0.42")
    monkeypatch.setenv("KASA_USERNAME", "u")
    monkeypatch.setenv("KASA_PASSWORD", "p")
    return fake, discover


@pytest.mark.asyncio
async def test_kasa_click_turns_on_when_off(fake_kasa):
    fake, discover = fake_kasa
    switch = KasaDevice("lamp_switch")
    await switch.on_click("left")
    assert fake.is_on is True
    assert "turn_on" in fake.calls
    assert discover.discover_calls == [("10.0.0.42", "u", "p")]


@pytest.mark.asyncio
async def test_kasa_click_turns_off_when_on(fake_kasa):
    fake, _ = fake_kasa
    fake.is_on = True
    switch = KasaDevice("lamp_switch")
    await switch.on_click("left")
    assert fake.is_on is False
    assert "turn_off" in fake.calls


@pytest.mark.asyncio
async def test_kasa_discovery_only_runs_once(fake_kasa):
    fake, discover = fake_kasa
    switch = KasaDevice("lamp_switch")
    await switch.on_click("left")
    await switch.on_click("left")
    await switch.on_click("left")
    assert len(discover.discover_calls) == 1


@pytest.mark.asyncio
async def test_kasa_context_menu_actions(fake_kasa):
    fake, _ = fake_kasa
    switch = KasaDevice("lamp_switch")
    items = await switch.on_context_menu()
    assert {i.action_id for i in items} == {"on", "off"}

    await switch.invoke_menu_action("on")
    assert fake.is_on is True
    await switch.invoke_menu_action("off")
    assert fake.is_on is False


@pytest.mark.asyncio
async def test_kasa_unknown_menu_action_raises(fake_kasa):
    switch = KasaDevice("lamp_switch")
    with pytest.raises(NotImplementedError, match="unknown action"):
        await switch.invoke_menu_action("explode")


@pytest.mark.asyncio
async def test_kasa_missing_host_env_raises(monkeypatch):
    monkeypatch.delenv("SMART_SWITCH_HOST", raising=False)
    switch = KasaDevice("lamp_switch")
    with pytest.raises(RuntimeError, match="not set"):
        await switch.on_click("left")


@pytest.mark.asyncio
async def test_kasa_close_disconnects_if_connected(fake_kasa):
    fake, _ = fake_kasa
    switch = KasaDevice("lamp_switch")
    await switch.on_click("left")
    await switch.close()
    assert "disconnect" in fake.calls


@pytest.mark.asyncio
async def test_kasa_close_no_op_when_never_connected(fake_kasa):
    """Closing without ever clicking shouldn't try to discover or connect."""
    fake, discover = fake_kasa
    switch = KasaDevice("lamp_switch")
    await switch.close()
    assert discover.discover_calls == []
    assert "disconnect" not in fake.calls


def test_kasa_factory_in_registry(fake_kasa):
    from runtime.devices.registry import DeviceRegistry

    registry = DeviceRegistry()
    spec = {"label": "Test switch", "config": {"host_env": "SMART_SWITCH_HOST"}}
    dev = registry.build("lamp_switch", "kasa", spec)
    assert isinstance(dev, KasaDevice)
    assert dev.label == "Test switch"
    assert "lamp_switch" in registry


# ---------- MAC normalization ----------

def test_normalize_mac_strips_separators_and_uppercases():
    assert _normalize_mac("aa:bb:cc:DD:ee:ff") == "AABBCCDDEEFF"
    assert _normalize_mac("AA-BB-CC-DD-EE-FF") == "AABBCCDDEEFF"
    assert _normalize_mac("AABB.CCDD.EEFF") == "AABBCCDDEEFF"
    assert _normalize_mac("AABBCCDDEEFF") == "AABBCCDDEEFF"


# ---------- MAC fallback: broadcast-discover when static IP is dead ----------

@pytest.mark.asyncio
async def test_kasa_broadcast_discovers_by_mac_when_static_ip_fails(fake_kasa, tmp_path):
    fake, discover = fake_kasa
    fake.mac = "AA:BB:CC:DD:EE:FF"
    # Static IP is dead (raises) — only broadcast finds the switch.
    discover.single_responses["10.0.0.42"] = ConnectionError("timeout")
    discover.discover_result = {"10.0.0.99": fake}
    discover.single_responses["10.0.0.99"] = fake  # so the cache-hit path on retry also works

    cache_path = tmp_path / ".kasa_cache.json"
    switch = KasaDevice("lamp_switch", mac="AA:BB:CC:DD:EE:FF", cache_path=cache_path)
    await switch.on_click("left")

    assert fake.is_on is True
    assert discover.broadcast_calls, "broadcast discover() should have been called"
    # New IP cached under the normalized MAC.
    import json
    cached = json.loads(cache_path.read_text())
    assert cached == {"AABBCCDDEEFF": "10.0.0.99"}


@pytest.mark.asyncio
async def test_kasa_cache_hit_avoids_broadcast(fake_kasa, tmp_path):
    """If the cached IP works on first try, broadcast discovery is NOT called."""
    fake, discover = fake_kasa
    fake.mac = "AA:BB:CC:DD:EE:FF"

    cache_path = tmp_path / ".kasa_cache.json"
    cache_path.write_text('{"AABBCCDDEEFF": "10.0.0.99"}')
    # Cached IP responds.
    discover.single_responses["10.0.0.99"] = fake

    switch = KasaDevice("lamp_switch", mac="AA:BB:CC:DD:EE:FF", cache_path=cache_path)
    await switch.on_click("left")

    assert fake.is_on is True
    assert discover.broadcast_calls == []  # no broadcast needed
    # Cached IP should be tried first (before env host).
    assert discover.discover_calls[0][0] == "10.0.0.99"


@pytest.mark.asyncio
async def test_kasa_mac_mismatch_at_cached_ip_triggers_rediscovery(fake_kasa, tmp_path):
    """If the cached IP responds but with a DIFFERENT MAC (DHCP shuffle landed
    another Kasa device there), reject the connection and broadcast-discover."""
    other_dev = FakeKasaDevice(mac="00:00:00:00:00:00")  # wrong MAC
    target_dev = FakeKasaDevice(mac="AA:BB:CC:DD:EE:FF")

    fake, discover = fake_kasa
    # Cached IP returns the wrong device; env IP also wrong; broadcast finds target.
    discover.single_responses["10.0.0.99"] = other_dev
    discover.single_responses["10.0.0.42"] = ConnectionError("dead")
    discover.discover_result = {"10.0.0.50": target_dev}

    cache_path = tmp_path / ".kasa_cache.json"
    cache_path.write_text('{"AABBCCDDEEFF": "10.0.0.99"}')

    switch = KasaDevice("lamp_switch", mac="AA:BB:CC:DD:EE:FF", cache_path=cache_path)
    await switch.on_click("left")

    assert target_dev.is_on is True
    assert other_dev.is_on is False  # we never toggled the imposter
    import json
    assert json.loads(cache_path.read_text()) == {"AABBCCDDEEFF": "10.0.0.50"}


@pytest.mark.asyncio
async def test_kasa_mac_not_found_on_lan_raises(fake_kasa, tmp_path):
    """Static IP dead + broadcast returns nothing matching MAC → clear error."""
    fake, discover = fake_kasa
    discover.single_responses["10.0.0.42"] = ConnectionError("dead")
    discover.discover_result = {}  # nothing on LAN

    cache_path = tmp_path / ".kasa_cache.json"
    switch = KasaDevice("lamp_switch", mac="AA:BB:CC:DD:EE:FF", cache_path=cache_path)
    with pytest.raises(RuntimeError, match="not found on the LAN"):
        await switch.on_click("left")


@pytest.mark.asyncio
async def test_kasa_no_mac_no_fallback_raises_helpful_error(fake_kasa, tmp_path):
    """Without a MAC, an unreachable switch gets a 'set mac:' hint, not a silent retry."""
    fake, discover = fake_kasa
    discover.single_responses["10.0.0.42"] = ConnectionError("dead")

    switch = KasaDevice("lamp_switch", cache_path=tmp_path / ".kasa_cache.json")
    with pytest.raises(RuntimeError, match="mac:"):
        await switch.on_click("left")
    assert discover.broadcast_calls == []  # no broadcast attempted


# ---------- Boot-time prewarm ----------

@pytest.mark.asyncio
async def test_kasa_prewarm_returns_reachable_string_on_success(fake_kasa, tmp_path):
    fake, _ = fake_kasa
    fake.host = "10.0.0.42"
    fake.mac = "AA:BB:CC:DD:EE:FF"
    switch = KasaDevice(
        "lamp_switch",
        mac="AA:BB:CC:DD:EE:FF",
        cache_path=tmp_path / ".kasa_cache.json",
    )
    status = await switch.prewarm()
    assert status is not None
    assert "reachable at" in status


@pytest.mark.asyncio
async def test_kasa_prewarm_returns_unreachable_instead_of_raising(fake_kasa, tmp_path):
    """Prewarm must NEVER raise — the runtime continues to boot even if a switch is offline."""
    fake, discover = fake_kasa
    discover.single_responses["10.0.0.42"] = ConnectionError("dead")
    discover.discover_result = {}  # broadcast finds nothing

    switch = KasaDevice(
        "lamp_switch",
        mac="AA:BB:CC:DD:EE:FF",
        cache_path=tmp_path / ".kasa_cache.json",
    )
    status = await switch.prewarm()
    assert status is not None
    assert "unreachable" in status
