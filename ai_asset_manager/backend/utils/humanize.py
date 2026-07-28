"""Human-readable formatting and its inverse.

The parsing half is used by the search query language (``size:>10GB``) and by CLI flags,
so it must accept the same spellings the formatter produces.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

_BINARY_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
_DECIMAL_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")

#: Accepts "10GB", "10 GiB", "1.5tb", "512k", or a bare byte count.
_SIZE_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[kmgtpe]?i?b?)\s*$",
    re.IGNORECASE,
)

_SIZE_MULTIPLIERS: dict[str, int] = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "ki": 1024,
    "kib": 1024,
    "m": 1000**2,
    "mb": 1000**2,
    "mi": 1024**2,
    "mib": 1024**2,
    "g": 1000**3,
    "gb": 1000**3,
    "gi": 1024**3,
    "gib": 1024**3,
    "t": 1000**4,
    "tb": 1000**4,
    "ti": 1024**4,
    "tib": 1024**4,
    "p": 1000**5,
    "pb": 1000**5,
    "pi": 1024**5,
    "pib": 1024**5,
}


def format_bytes(num_bytes: float, *, binary: bool = True, precision: int = 1) -> str:
    """Render a byte count as a human-readable string.

    Args:
        num_bytes: Size in bytes.
        binary: Use 1024-based units (``GiB``) rather than 1000-based (``GB``).
        precision: Decimal places for non-byte units.

    Returns:
        A string such as ``"3.2 GiB"``. Exact bytes are rendered without decimals.

    Examples:
        >>> format_bytes(0)
        '0 B'
        >>> format_bytes(1536)
        '1.5 KiB'
        >>> format_bytes(1_000_000_000, binary=False)
        '1.0 GB'
    """
    if num_bytes < 0:
        return f"-{format_bytes(-num_bytes, binary=binary, precision=precision)}"

    units = _BINARY_UNITS if binary else _DECIMAL_UNITS
    step = 1024.0 if binary else 1000.0

    value = float(num_bytes)
    for unit in units:
        if value < step or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.{precision}f} {unit}"
        value /= step
    return f"{value:.{precision}f} {units[-1]}"  # pragma: no cover - loop always returns


def parse_size(text: str) -> int:
    """Parse a human-written size into bytes.

    Accepts both binary and decimal spellings; a bare ``GB`` means 1000³ while ``GiB``
    means 1024³, matching what :func:`format_bytes` emits.

    Args:
        text: A string such as ``"10GB"``, ``"1.5 GiB"`` or ``"2048"``.

    Returns:
        The size in bytes.

    Raises:
        ValueError: If the string is not a recognisable size.

    Examples:
        >>> parse_size("1KiB")
        1024
        >>> parse_size("1kb")
        1000
        >>> parse_size("512")
        512
    """
    match = _SIZE_RE.match(text)
    if not match:
        raise ValueError(f"Unrecognised size: {text!r}")

    unit = match.group("unit").lower()
    if unit not in _SIZE_MULTIPLIERS:
        raise ValueError(f"Unrecognised size unit in {text!r}")
    return int(float(match.group("value")) * _SIZE_MULTIPLIERS[unit])


def format_count(value: int) -> str:
    """Render a large integer compactly.

    Used for parameter counts, where ``494032768`` is far less legible than ``494.0M``.

    Examples:
        >>> format_count(494_032_768)
        '494.0M'
        >>> format_count(8_030_261_248)
        '8.0B'
        >>> format_count(512)
        '512'
    """
    if value < 0:
        return f"-{format_count(-value)}"
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if value >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return str(value)


def format_duration(seconds: float) -> str:
    """Render a duration compactly.

    Examples:
        >>> format_duration(0.42)
        '0.4s'
        >>> format_duration(95)
        '1m 35s'
        >>> format_duration(3725)
        '1h 2m'
    """
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m {secs}s"
    hours, remainder = divmod(int(seconds), 3600)
    return f"{hours}h {remainder // 60}m"


def format_relative_time(moment: datetime, *, now: datetime | None = None) -> str:
    """Render a timestamp as an age such as ``"3 days ago"``.

    Args:
        moment: The timestamp to describe. Naive values are assumed to be UTC.
        now: Reference point, defaulting to the current time.

    Returns:
        A short relative description.
    """
    now = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    delta: timedelta = now - moment
    total = int(delta.total_seconds())

    if total < 0:
        return "in the future"
    if total < 60:
        return "just now"
    for unit_seconds, label in (
        (31_536_000, "year"),
        (2_592_000, "month"),
        (604_800, "week"),
        (86_400, "day"),
        (3_600, "hour"),
        (60, "minute"),
    ):
        count = total // unit_seconds
        if count >= 1:
            return f"{count} {label}{'s' if count != 1 else ''} ago"
    return "just now"  # pragma: no cover - unreachable given the 60s guard above
