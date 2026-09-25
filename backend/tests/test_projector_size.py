"""Tests for --projector-size parsing.

This flag replaces calibration entirely: it is the only thing that tells the
editor how big the wall canvas is. A silently-wrong value puts every polygon in
the wrong place on the wall with no error to go on, so the negative cases here
matter more than the positive ones — bad input must raise, never fall back.
"""

import pytest

from runtime.__main__ import parse_projector_size


# ---------------- accepted forms ----------------


def test_parses_standard_form():
    assert parse_projector_size("1920x1080") == (1920, 1080)


def test_parses_uppercase_x():
    assert parse_projector_size("1280X800") == (1280, 800)


def test_parses_unicode_multiplication_sign():
    """Copy-pasting '1920×1080' out of a display-settings panel is common."""
    assert parse_projector_size("1920×1080") == (1920, 1080)


def test_parses_non_16_9_projector():
    assert parse_projector_size("1024x768") == (1024, 768)


# ---------------- rejected forms ----------------


@pytest.mark.parametrize(
    "bad, reason",
    [
        ("1920", "no separator"),
        ("1920x", "missing height"),
        ("x1080", "missing width"),
        ("1920x1080x60", "three components — refresh rate pasted in"),
        ("1920*1080", "wrong separator"),
        ("1920 1080", "space-separated"),
        ("1920.5x1080", "non-integer width"),
        ("hdx1080", "non-numeric width"),
        ("", "empty string"),
        ("0x1080", "zero width"),
        ("1920x0", "zero height"),
        ("-1920x1080", "negative width"),
        ("1920x-1080", "negative height"),
    ],
)
def test_rejects_malformed(bad, reason):
    with pytest.raises(ValueError):
        parse_projector_size(bad)


def test_error_message_names_the_expected_shape():
    """The message is the only guidance the user gets — it must show the form."""
    with pytest.raises(ValueError, match="1920x1080"):
        parse_projector_size("1920")


def test_zero_is_rejected_as_positive_not_as_format():
    """'0x1080' parses as two ints, so it must be caught by the range check."""
    with pytest.raises(ValueError, match="positive"):
        parse_projector_size("0x1080")
