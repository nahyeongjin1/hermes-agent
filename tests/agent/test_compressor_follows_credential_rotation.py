"""A same-route credential change must move every holder of the old main key.

The compressor hands ``api_key`` to the auxiliary router as ``main_runtime``, which pins the summary
call to it. When a credential-pool rotation (``_swap_credential``) or a token refresh updates only the
agent, every later summary reuses the dead key and compaction aborts (401) for the rest of the session
while the main loop keeps working. Holders are identity-matched on the old key, as the Anthropic refresh
path already does, so a compressor routed elsewhere keeps its own credential.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.context_compressor import ContextCompressor
from run_agent import AIAgent

ANTHROPIC = "https://api.anthropic.com"


def _compressor(*, api_key, base_url):
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.api_key, compressor.base_url = api_key, base_url
    return compressor


def _anthropic_agent(compressor):
    return SimpleNamespace(
        api_mode="anthropic_messages", provider="anthropic", model="claude-opus-4-6",
        api_key="revoked", base_url=ANTHROPIC,
        _anthropic_client=MagicMock(), _anthropic_api_key="revoked", _anthropic_base_url=ANTHROPIC,
        _build_direct_anthropic_client=MagicMock(return_value=MagicMock()),
        _anthropic_oauth_flag=MagicMock(return_value=True),
        context_compressor=compressor,
    )


def _openai_agent(compressor, *, key="old", base="https://a.example/v1"):
    return SimpleNamespace(
        api_mode="chat_completions", provider="custom", model="shared-model", api_key=key, base_url=base,
        _client_kwargs={"api_key": key, "base_url": base},
        _reapply_route_client_config=MagicMock(), _replace_primary_openai_client=MagicMock(return_value=True),
        _apply_client_headers_for_base_url=MagicMock(),
        context_compressor=compressor,
    )


def _entry(key, base=ANTHROPIC):
    return SimpleNamespace(id="next", runtime_api_key=key, access_token="", runtime_base_url=base, base_url=base)


def test_anthropic_pool_rotation_moves_compressor_key():
    compressor = _compressor(api_key="revoked", base_url=ANTHROPIC)
    agent = _anthropic_agent(compressor)

    assert AIAgent._swap_credential(agent, _entry("fresh")) is True

    assert agent.api_key == "fresh"
    assert (compressor.api_key, compressor.base_url) == ("fresh", ANTHROPIC)


def test_openai_style_pool_rotation_moves_compressor_key_and_route():
    compressor = _compressor(api_key="old", base_url="https://a.example/v1")
    agent = _openai_agent(compressor)

    AIAgent._swap_credential(agent, _entry("new", "https://b.example/v1"))

    assert (compressor.api_key, compressor.base_url) == ("new", "https://b.example/v1")


def test_compressor_on_its_own_credential_is_left_alone():
    """An explicitly routed compressor never held the main key, so a rotation does not touch it."""
    compressor = _compressor(api_key="aux-key", base_url="https://openrouter.ai/api/v1")
    agent = _anthropic_agent(compressor)

    AIAgent._swap_credential(agent, _entry("fresh"))

    assert (compressor.api_key, compressor.base_url) == ("aux-key", "https://openrouter.ai/api/v1")


def test_refused_swap_leaves_compressor_unchanged(monkeypatch):
    compressor = _compressor(api_key="revoked", base_url=ANTHROPIC)
    agent = _anthropic_agent(compressor)
    monkeypatch.setattr("hermes_cli.anon_auth.route_can_serve_model", lambda *a, **k: False)

    assert AIAgent._swap_credential(agent, _entry("fresh")) is False

    assert compressor.api_key == "revoked"


def test_agent_without_compressor_still_swaps():
    agent = _anthropic_agent(None)
    del agent.context_compressor

    assert AIAgent._swap_credential(agent, _entry("fresh")) is True
    assert agent.api_key == "fresh"


def test_pool_rotation_rotates_the_published_aux_runtime_key():
    """Same-turn ``auto`` aux calls read the published runtime, not the compressor."""
    from agent.auxiliary_client import _RUNTIME_MAIN_CONTEXT

    runtime = {"provider": "anthropic", "model": "claude-opus-4-6", "base_url": ANTHROPIC, "api_key": "revoked"}
    token = _RUNTIME_MAIN_CONTEXT.set(runtime)
    try:
        AIAgent._swap_credential(_anthropic_agent(_compressor(api_key="revoked", base_url=ANTHROPIC)), _entry("fresh"))
    finally:
        _RUNTIME_MAIN_CONTEXT.reset(token)

    assert runtime["api_key"] == "fresh"


def test_openai_credential_refresh_moves_compressor_key():
    compressor = _compressor(api_key="old", base_url="https://n.example/v1")
    agent = _openai_agent(compressor, base="https://n.example/v1")
    agent._sync_client_kwargs_credentials = lambda: AIAgent._sync_client_kwargs_credentials(agent)

    assert AIAgent._adopt_openai_credentials(agent, "new", "https://n.example/v1", reason="t") is True

    assert compressor.api_key == "new"


def test_copilot_token_refresh_moves_compressor_key():
    compressor = _compressor(api_key="old", base_url="https://a.example/v1")
    agent = _openai_agent(compressor)
    agent._sync_client_kwargs_credentials = lambda: AIAgent._sync_client_kwargs_credentials(agent)

    AIAgent._apply_copilot_token(agent, "new", None, reason="t")

    assert compressor.api_key == "new"


def test_route_changing_rotation_resets_the_aux_ceiling_and_keeps_the_published_runtime():
    """A route move invalidates the probed aux ceiling; the published runtime keeps its old host+key pair
    for the rest of the turn rather than sending the new key to the old host."""
    from agent.auxiliary_client import _RUNTIME_MAIN_CONTEXT

    compressor = _compressor(api_key="old", base_url="https://a.example/v1")
    compressor._aux_context_ceiling = 123_456
    runtime = {"provider": "custom", "model": "shared-model", "base_url": "https://a.example/v1", "api_key": "old"}
    token = _RUNTIME_MAIN_CONTEXT.set(runtime)
    try:
        AIAgent._swap_credential(_openai_agent(compressor), _entry("new", "https://b.example/v1"))
    finally:
        _RUNTIME_MAIN_CONTEXT.reset(token)

    assert compressor._aux_context_ceiling is None
    assert runtime["api_key"] == "old"


def _env_agent(compressor, *, replace_ok):
    agent = _openai_agent(compressor)
    agent._replace_primary_openai_client = MagicMock(return_value=replace_ok)
    agent._resolve_env_credentials = lambda: ("new", "https://a.example/v1", "https://a.example/v1")
    agent._should_adopt_env_credentials = lambda *a: True
    agent._sync_client_kwargs_credentials = lambda: AIAgent._sync_client_kwargs_credentials(agent)
    return agent


def test_env_credential_refresh_moves_compressor_key():
    compressor = _compressor(api_key="old", base_url="https://a.example/v1")

    assert AIAgent._try_refresh_env_client_credentials(_env_agent(compressor, replace_ok=True)) is True

    assert compressor.api_key == "new"


def test_rolled_back_env_refresh_leaves_compressor_unchanged():
    compressor = _compressor(api_key="old", base_url="https://a.example/v1")
    agent = _env_agent(compressor, replace_ok=False)

    assert AIAgent._try_refresh_env_client_credentials(agent) is False

    assert (agent.api_key, compressor.api_key) == ("old", "old")
