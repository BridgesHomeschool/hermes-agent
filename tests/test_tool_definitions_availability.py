"""Availability drift must not be hidden forever by the outer schema memo."""

import importlib
import json
from unittest.mock import Mock

import pytest

import model_tools
from tools.registry import ToolRegistry


@pytest.fixture
def catalog(monkeypatch):
    registry_module = importlib.import_module("tools.registry")
    registry = ToolRegistry()
    monkeypatch.setattr(model_tools, "registry", registry)
    monkeypatch.setattr(registry_module, "registry", registry)
    monkeypatch.setattr(model_tools, "_tool_defs_cache", {})
    monkeypatch.setattr(registry_module, "_check_fn_cache", {})
    monkeypatch.setattr(registry_module, "_check_fn_last_good", {})
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(model_tools.time, "monotonic", clock)
    state = {"available": False}
    probe = Mock(side_effect=lambda: state["available"])
    name = "mcp_synthetic_recovery_read"
    registry.register(
        name=name, toolset="synthetic_recovery",
        schema={"name": name, "description": "Synthetic read", "parameters": {"type": "object"}},
        handler=lambda args: '{}', check_fn=probe,
    )
    monkeypatch.setattr(model_tools, "validate_toolset", lambda name: name == "synthetic_recovery")
    monkeypatch.setattr(model_tools, "resolve_toolset",
                        lambda name: ["mcp_synthetic_recovery_read"] if name == "synthetic_recovery" else [])
    return registry, registry_module, clock, state, probe, name


@pytest.mark.parametrize("raw_catalog", [False, True])
def test_availability_recovers_without_registry_mutation(catalog, monkeypatch, raw_catalog):
    registry, module, clock, state, probe, name = catalog
    compute = Mock(wraps=model_tools._compute_tool_definitions)
    monkeypatch.setattr(model_tools, "_compute_tool_definitions", compute)
    generation = registry._generation
    kwargs = dict(enabled_toolsets=["synthetic_recovery"], quiet_mode=True,
                  skip_tool_search_assembly=raw_catalog)
    frozen = model_tools.get_tool_definitions(**kwargs)
    assert frozen == []
    state["available"] = True
    clock.return_value = module._CHECK_FN_TTL_SECONDS / 2
    assert model_tools.get_tool_definitions(**kwargs) == []
    assert compute.call_count == probe.call_count == 1
    # Both availability caches have expired, with no reload or registration.
    clock.return_value = module._CHECK_FN_TTL_SECONDS * 2
    recovered = model_tools.get_tool_definitions(**kwargs)
    assert name in {td["function"]["name"] for td in recovered} if raw_catalog else recovered
    assert name in model_tools._last_resolved_tool_names
    assert registry._generation == generation
    assert frozen == []  # An existing agent's schema list is not rewritten.
    assert compute.call_count == probe.call_count == 2
    assert model_tools.get_tool_definitions(**kwargs) == recovered
    assert compute.call_count == probe.call_count == 2
    # Persistent loss must also become visible after the existing grace period.
    state["available"] = False
    clock.return_value += module._CHECK_FN_FAILURE_GRACE_SECONDS + module._CHECK_FN_TTL_SECONDS * 2
    assert model_tools.get_tool_definitions(**kwargs) == []


def test_recovered_catalog_keeps_session_toolset_restrictions(catalog):
    _, module, clock, state, _, name = catalog
    allowed = dict(enabled_toolsets=["synthetic_recovery"], quiet_mode=True,
                   skip_tool_search_assembly=True)
    restricted = dict(**allowed, disabled_toolsets=["synthetic_recovery"])
    assert model_tools.get_tool_definitions(**allowed) == []
    assert model_tools.get_tool_definitions(**restricted) == []
    state["available"] = True
    clock.return_value = module._CHECK_FN_TTL_SECONDS * 2
    assert name in {td["function"]["name"] for td in model_tools.get_tool_definitions(**allowed)}
    assert model_tools.get_tool_definitions(**restricted) == []
    assert name in {td["function"]["name"] for td in model_tools.get_tool_definitions(**allowed)}
    # The real bridge must read the refreshed, session-filtered catalog too.
    for disabled, expected in [(None, True), (["synthetic_recovery"], False), (None, True)]:
        result, dispatch = model_tools._dispatch_bridge_tool(
            "tool_search", {"queries": [name], "limit": 5},
            ["synthetic_recovery"], disabled,
        )
        assert dispatch is None
        assert (name in json.loads(result)["tools"]) is expected
