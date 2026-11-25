"""Shared helpers for the Vast.ai experiment infrastructure.

The helpers here are deliberately pure so they can be unit-tested without a
Vast.ai account, a GPU, or network access. Operational scripts import them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class Offer:
    """A normalized Vast.ai offer."""

    offer_id: int
    gpu_name: str
    num_gpus: int
    gpu_ram_gb: float
    dph: float
    reliability: float
    inet_down_mbps: float
    inet_up_mbps: float
    cuda_max: float
    location: str = ""

    @property
    def vram_gb(self) -> float:
        return self.gpu_ram_gb * max(1, self.num_gpus)


def parse_ssh_url(value: str) -> tuple[str, str, int]:
    """Return ``(user, host, port)`` from a Vast.ai SSH URL.

    Accepts ``ssh://root@host:12345`` and ``root@host:12345``.
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("empty SSH URL")
    if "://" not in raw:
        raw = "ssh://" + raw
    parsed = urlparse(raw)
    if not parsed.hostname:
        raise ValueError(f"no host in SSH URL: {value!r}")
    user = parsed.username or "root"
    port = parsed.port or 22
    return user, parsed.hostname, int(port)


def estimate_cost(dph: float, hours: float) -> float:
    """Cost in USD for ``hours`` at ``dph`` dollars per hour."""
    if dph < 0 or not math.isfinite(dph):
        raise ValueError("dph must be finite and >= 0")
    if hours < 0 or not math.isfinite(hours):
        raise ValueError("hours must be finite and >= 0")
    return round(dph * hours, 4)


def select_offers(
    offers: list[Offer],
    *,
    max_price: float,
    min_vram_gb: float = 24.0,
    min_reliability: float = 0.95,
    min_down_mbps: float = 100.0,
    allowed_gpus: tuple[str, ...] = ("RTX 4090", "RTX 3090", "A100 SXM4", "A100 PCIE"),
) -> list[Offer]:
    """Filter and sort offers cheapest-first.

    The filter is intentionally strict: a cheap but unreliable host costs more
    than it saves when a multi-hour run dies at hour two.
    """
    allowed = {name.lower() for name in allowed_gpus}
    chosen: list[Offer] = []
    for offer in offers:
        if offer.num_gpus < 1:
            continue
        if offer.dph <= 0 or offer.dph > max_price:
            continue
        if offer.vram_gb < min_vram_gb:
            continue
        if offer.reliability < min_reliability:
            continue
        if offer.inet_down_mbps < min_down_mbps:
            continue
        if allowed and offer.gpu_name.lower() not in allowed:
            continue
        chosen.append(offer)
    return sorted(chosen, key=lambda o: (o.dph, -o.reliability, -o.vram_gb))


def planned_minutes_ok(minutes: float, *, minimum: float = 30.0) -> bool:
    """Guard against renting a GPU for a trivial job."""
    return math.isfinite(minutes) and minutes >= minimum


def format_offer_table(offers: list[Offer], limit: int = 10) -> str:
    """Render a compact, paste-friendly offer table."""
    if not offers:
        return "no matching offers"
    lines = [
        "id       gpu            vram  $/hr   rel   down   cuda  location",
        "-------  -------------  ----  -----  ----  -----  ----  --------",
    ]
    for o in offers[:limit]:
        lines.append(
            f"{o.offer_id:<7}  {o.gpu_name:<13}  {o.vram_gb:>4.0f}  "
            f"{o.dph:>5.2f}  {o.reliability:>4.2f}  {o.inet_down_mbps:>5.0f}  "
            f"{o.cuda_max:>4.1f}  {o.location[:24]}"
        )
    return "\n".join(lines)
