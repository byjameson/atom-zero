"""Tests for the Vast.ai infrastructure helpers (no network, no GPU)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vastai"))

from common import (  # noqa: E402
    Offer,
    estimate_cost,
    format_offer_table,
    parse_ssh_url,
    planned_minutes_ok,
    select_offers,
)


def _offer(**kwargs) -> Offer:
    base = dict(
        offer_id=1,
        gpu_name="RTX 3090",
        num_gpus=1,
        gpu_ram_gb=24.0,
        dph=0.15,
        reliability=0.99,
        inet_down_mbps=500.0,
        inet_up_mbps=200.0,
        cuda_max=12.4,
        location="Test",
    )
    base.update(kwargs)
    return Offer(**base)


def test_parse_ssh_url_with_scheme():
    assert parse_ssh_url("ssh://root@203.0.113.5:41234") == ("root", "203.0.113.5", 41234)


def test_parse_ssh_url_without_scheme():
    assert parse_ssh_url("root@example.com:2222") == ("root", "example.com", 2222)


def test_parse_ssh_url_rejects_empty():
    try:
        parse_ssh_url("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty SSH URL should fail")


def test_estimate_cost_rounds_and_validates():
    assert estimate_cost(0.15, 3.0) == 0.45
    assert estimate_cost(0.0, 10.0) == 0.0
    for bad in (-1.0, float("inf")):
        try:
            estimate_cost(bad, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("bad dph should fail")


def test_planned_minutes_guard():
    assert planned_minutes_ok(30) is True
    assert planned_minutes_ok(29.9) is False
    assert planned_minutes_ok(float("nan")) is False


def test_select_offers_filters_price_vram_reliability_and_network():
    offers = [
        _offer(offer_id=1, dph=0.10),
        _offer(offer_id=2, dph=0.60),
        _offer(offer_id=3, gpu_ram_gb=16.0),
        _offer(offer_id=4, reliability=0.50),
        _offer(offer_id=5, inet_down_mbps=10.0),
        _offer(offer_id=6, gpu_name="Tesla P40"),
        _offer(offer_id=7, dph=0.20, reliability=0.995),
    ]
    chosen = select_offers(offers, max_price=0.5)
    assert [o.offer_id for o in chosen] == [1, 7]


def test_select_offers_sorts_by_price_then_reliability():
    a = _offer(offer_id=1, dph=0.20, reliability=0.96)
    b = _offer(offer_id=2, dph=0.20, reliability=0.999)
    c = _offer(offer_id=3, dph=0.19, reliability=0.95)
    assert [o.offer_id for o in select_offers([a, b, c], max_price=0.5)] == [3, 2, 1]


def test_multi_gpu_offer_vram_is_total():
    assert _offer(num_gpus=2, gpu_ram_gb=24.0).vram_gb == 48.0


def test_format_offer_table_empty():
    assert format_offer_table([]) == "no matching offers"


def test_format_offer_table_contains_id_and_gpu():
    text = format_offer_table([_offer(offer_id=4242)])
    assert "4242" in text
    assert "RTX 3090" in text


def test_preflight_parses_raw_offers():
    spec = importlib.util.spec_from_file_location(
        "vastai_preflight", ROOT / "scripts" / "vastai" / "preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    payload = (
        '[{"id": 7, "gpu_name": "RTX 4090", "num_gpus": 1, "gpu_ram": 24564, '
        '"dph_total": 0.37, "reliability2": 0.99, "inet_down": 900, '
        '"inet_up": 400, "cuda_max_good": 12.4, "geolocation": "US"}]'
    )
    offers = module.parse_offers(payload)
    assert len(offers) == 1
    assert offers[0].offer_id == 7
    assert offers[0].vram_gb == 24564 / 1024
    assert offers[0].dph == 0.37
