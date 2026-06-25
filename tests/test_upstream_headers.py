"""Tests for the --upstream-header flag and UPSTREAM_HEADERS env var"""

import re

import pytest
import typer
from typer import BadParameter
from typer.testing import CliRunner
from unittest.mock import patch

from ollama_mcp_bridge.utils import parse_upstream_headers, parse_bool_env

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _invoke(args):
    """Run cli_app via CliRunner with server startup mocked out."""
    import ollama_mcp_bridge.main as main_module

    app = typer.Typer()
    app.command()(main_module.cli_app)
    with (
        patch("ollama_mcp_bridge.main.is_port_in_use", return_value=(False, "")),
        patch("ollama_mcp_bridge.main.check_ollama_health", return_value=True),
        patch("ollama_mcp_bridge.main.check_for_updates"),
        patch("ollama_mcp_bridge.main.uvicorn.run") as mock_uvicorn,
    ):
        result = CliRunner().invoke(app, args)
    return result, mock_uvicorn


# --- parse_upstream_headers unit tests ---------------------------------------


def test_parse_none_when_no_input():
    assert parse_upstream_headers(None, []) is None
    assert parse_upstream_headers("", None) is None


def test_parse_env_json():
    assert parse_upstream_headers('{"Authorization": "Bearer xxx"}', []) == {"Authorization": "Bearer xxx"}


def test_parse_repeatable_flags():
    headers = parse_upstream_headers(None, ["Authorization: Bearer xxx", "X-API-Key: yyy"])
    assert headers == {"Authorization": "Bearer xxx", "X-API-Key": "yyy"}


def test_parse_flags_override_env_per_key():
    headers = parse_upstream_headers('{"X-API-Key": "env", "Keep": "1"}', ["X-API-Key: cli"])
    assert headers == {"X-API-Key": "cli", "Keep": "1"}


def test_parse_strips_flag_whitespace():
    assert parse_upstream_headers(None, ["Authorization:  Bearer xyz "]) == {"Authorization": "Bearer xyz"}


def test_parse_keeps_value_colons():
    assert parse_upstream_headers(None, ["X-Url: http://h:11434"]) == {"X-Url": "http://h:11434"}


def test_parse_rejects_flag_without_colon():
    with pytest.raises(BadParameter):
        parse_upstream_headers(None, ["missing-colon"])


def test_parse_rejects_empty_flag_name():
    with pytest.raises(BadParameter):
        parse_upstream_headers(None, [": value"])


def test_parse_rejects_invalid_json():
    with pytest.raises(BadParameter):
        parse_upstream_headers("not json", None)


def test_parse_rejects_non_object_json():
    with pytest.raises(BadParameter):
        parse_upstream_headers('["a", "b"]', None)


# --- CLI integration tests ----------------------------------------------------


def test_cli_headers_from_env_json(monkeypatch):
    monkeypatch.setenv("UPSTREAM_HEADERS", '{"Authorization": "Bearer test-token"}')
    from ollama_mcp_bridge.api import app as fastapi_app

    result, mock_uvicorn = _invoke(["--config", "mcp-config.json"])

    assert result.exit_code == 0, result.output
    mock_uvicorn.assert_called_once()
    assert fastapi_app.state.ollama_headers == {"Authorization": "Bearer test-token"}


def test_cli_headers_from_repeatable_flags(monkeypatch):
    monkeypatch.delenv("UPSTREAM_HEADERS", raising=False)
    from ollama_mcp_bridge.api import app as fastapi_app

    result, mock_uvicorn = _invoke(
        [
            "--config",
            "mcp-config.json",
            "--upstream-header",
            "Authorization: Bearer xxx",
            "--upstream-header",
            "X-API-Key: yyy",
        ]
    )

    assert result.exit_code == 0, result.output
    mock_uvicorn.assert_called_once()
    assert fastapi_app.state.ollama_headers == {"Authorization": "Bearer xxx", "X-API-Key": "yyy"}


def test_cli_flags_override_env(monkeypatch):
    monkeypatch.setenv("UPSTREAM_HEADERS", '{"Authorization": "Bearer env", "X-API-Key": "keep"}')
    from ollama_mcp_bridge.api import app as fastapi_app

    result, _ = _invoke(["--config", "mcp-config.json", "--upstream-header", "Authorization: Bearer cli"])

    assert result.exit_code == 0, result.output
    assert fastapi_app.state.ollama_headers == {"Authorization": "Bearer cli", "X-API-Key": "keep"}


def test_cli_headers_unset(monkeypatch):
    monkeypatch.delenv("UPSTREAM_HEADERS", raising=False)
    from ollama_mcp_bridge.api import app as fastapi_app

    result, _ = _invoke(["--config", "mcp-config.json"])

    assert result.exit_code == 0, result.output
    assert fastapi_app.state.ollama_headers is None


def test_cli_invalid_flag_format(monkeypatch):
    monkeypatch.delenv("UPSTREAM_HEADERS", raising=False)

    result, _ = _invoke(["--config", "mcp-config.json", "--upstream-header", "missing-colon"])

    assert result.exit_code == 2
    output = _ANSI_ESCAPE_RE.sub("", result.output)
    assert "--upstream-header" in output


def test_cli_invalid_env_json(monkeypatch):
    monkeypatch.setenv("UPSTREAM_HEADERS", "not json")

    result, _ = _invoke(["--config", "mcp-config.json"])

    assert result.exit_code == 2
    output = _ANSI_ESCAPE_RE.sub("", result.output)
    assert "UPSTREAM_HEADERS" in output


# --- parse_bool_env unit tests ----------------------------------------------


def test_parse_bool_env_unset_returns_default(monkeypatch):
    monkeypatch.delenv("TEST_BOOL_VAR", raising=False)
    assert parse_bool_env("TEST_BOOL_VAR", True) is True
    assert parse_bool_env("TEST_BOOL_VAR", False) is False


def test_parse_bool_env_truthy_values(monkeypatch):
    for value in ["1", "true", "TRUE", "yes", "Y", "on", "t"]:
        monkeypatch.setenv("TEST_BOOL_VAR", value)
        assert parse_bool_env("TEST_BOOL_VAR", False) is True, value


def test_parse_bool_env_falsy_values(monkeypatch):
    for value in ["0", "false", "FALSE", "no", "N", "off", "f"]:
        monkeypatch.setenv("TEST_BOOL_VAR", value)
        assert parse_bool_env("TEST_BOOL_VAR", True) is False, value


def test_parse_bool_env_empty_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_BOOL_VAR", "   ")
    assert parse_bool_env("TEST_BOOL_VAR", True) is True


def test_parse_bool_env_unknown_returns_default(monkeypatch):
    monkeypatch.setenv("TEST_BOOL_VAR", "maybe")
    assert parse_bool_env("TEST_BOOL_VAR", False) is False


# --- Forwarding flag wiring (CLI + state) -----------------------------------


def test_cli_forward_client_headers_default_true(monkeypatch):
    monkeypatch.delenv("FORWARD_CLIENT_HEADERS", raising=False)
    from ollama_mcp_bridge.api import app as fastapi_app

    result, _ = _invoke(["--config", "mcp-config.json"])

    assert result.exit_code == 0, result.output
    assert fastapi_app.state.forward_client_headers is True


def test_cli_no_forward_client_headers_flag(monkeypatch):
    monkeypatch.delenv("FORWARD_CLIENT_HEADERS", raising=False)
    from ollama_mcp_bridge.api import app as fastapi_app

    result, _ = _invoke(["--config", "mcp-config.json", "--no-forward-client-headers"])

    assert result.exit_code == 0, result.output
    assert fastapi_app.state.forward_client_headers is False


def test_cli_forward_client_headers_env_false(monkeypatch):
    monkeypatch.setenv("FORWARD_CLIENT_HEADERS", "false")
    from ollama_mcp_bridge.api import app as fastapi_app

    result, _ = _invoke(["--config", "mcp-config.json"])

    assert result.exit_code == 0, result.output
    assert fastapi_app.state.forward_client_headers is False


def test_cli_forward_client_headers_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("FORWARD_CLIENT_HEADERS", "false")
    from ollama_mcp_bridge.api import app as fastapi_app

    result, _ = _invoke(["--config", "mcp-config.json", "--forward-client-headers"])

    assert result.exit_code == 0, result.output
    assert fastapi_app.state.forward_client_headers is True


# --- Forwarding behavior (ProxyService unit tests) --------------------------


class _DummyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": "ok"}}


class _DummyAsyncClient:
    def __init__(self):
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _DummyResponse()

    async def aclose(self):
        return None


class _DummyRequest:
    def __init__(self, headers):
        self.headers = headers


def _make_proxy_service(forward_client_headers, ollama_headers=None):
    from ollama_mcp_bridge.mcp_manager import MCPManager
    from ollama_mcp_bridge.proxy_service import ProxyService

    mgr = MCPManager(
        ollama_url="http://localhost:11434",
        ollama_headers=ollama_headers,
        forward_client_headers=forward_client_headers,
    )
    ps = ProxyService(mgr)
    return ps, mgr


@pytest.mark.anyio
async def test_forwarding_disabled_does_not_forward_client_headers():
    ps, mgr = _make_proxy_service(forward_client_headers=False)
    original = ps.http_client
    ps.http_client = _DummyAsyncClient()
    try:
        request = _DummyRequest({"authorization": "Bearer client", "X-Custom": "v"})
        await ps._proxy_with_tools_non_streaming(
            "/api/chat",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            request=request,
        )
        assert ps.http_client.calls[0]["headers"] == {}
    finally:
        dummy = ps.http_client
        ps.http_client = original
        await dummy.aclose()
        await original.aclose()
        await mgr.http_client.aclose()


@pytest.mark.anyio
async def test_forwarding_enabled_forwards_authorization():
    ps, mgr = _make_proxy_service(forward_client_headers=True)
    original = ps.http_client
    ps.http_client = _DummyAsyncClient()
    try:
        request = _DummyRequest({"authorization": "Bearer client-token", "X-Custom": "v"})
        await ps._proxy_with_tools_non_streaming(
            "/api/chat",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            request=request,
        )
        sent = ps.http_client.calls[0]["headers"]
        assert sent.get("authorization") == "Bearer client-token"
        assert sent.get("X-Custom") == "v"
        assert "host" not in {k.lower() for k in sent}
    finally:
        dummy = ps.http_client
        ps.http_client = original
        await dummy.aclose()
        await original.aclose()
        await mgr.http_client.aclose()


@pytest.mark.anyio
async def test_forwarding_excludes_host_header():
    ps, mgr = _make_proxy_service(forward_client_headers=True)
    original = ps.http_client
    ps.http_client = _DummyAsyncClient()
    try:
        request = _DummyRequest({"host": "evil.example", "authorization": "Bearer x"})
        await ps._proxy_with_tools_non_streaming(
            "/api/chat",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            request=request,
        )
        sent = ps.http_client.calls[0]["headers"]
        assert "host" not in sent
        assert "Host" not in sent
        assert sent.get("authorization") == "Bearer x"
    finally:
        dummy = ps.http_client
        ps.http_client = original
        await dummy.aclose()
        await original.aclose()
        await mgr.http_client.aclose()


@pytest.mark.anyio
async def test_configured_static_header_overrides_forwarded_client_header():
    ps, mgr = _make_proxy_service(forward_client_headers=True, ollama_headers={"Authorization": "Bearer static"})
    original = ps.http_client
    ps.http_client = _DummyAsyncClient()
    try:
        request = _DummyRequest({"authorization": "Bearer client"})
        await ps._proxy_with_tools_non_streaming(
            "/api/chat",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            request=request,
        )
        sent = ps.http_client.calls[0]["headers"]
        assert sent.get("Authorization") == "Bearer static"
        assert "authorization" not in sent  # lower-cased client key is filtered
    finally:
        dummy = ps.http_client
        ps.http_client = original
        await dummy.aclose()
        await original.aclose()
        await mgr.http_client.aclose()


@pytest.mark.anyio
async def test_forwarding_disabled_only_sends_static_headers():
    ps, mgr = _make_proxy_service(forward_client_headers=False, ollama_headers={"X-API-Key": "k"})
    original = ps.http_client
    ps.http_client = _DummyAsyncClient()
    try:
        request = _DummyRequest({"authorization": "Bearer client", "X-Custom": "v"})
        await ps._proxy_with_tools_non_streaming(
            "/api/chat",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            request=request,
        )
        assert ps.http_client.calls[0]["headers"] == {"X-API-Key": "k"}
    finally:
        dummy = ps.http_client
        ps.http_client = original
        await dummy.aclose()
        await original.aclose()
        await mgr.http_client.aclose()
