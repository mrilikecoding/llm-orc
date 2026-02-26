# Ollama Options Pass-Through Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow Ollama provider-specific parameters (num_ctx, top_k, top_p, repeat_penalty, seed, etc.) to flow from profile YAML and agent config through to the Ollama API.

**Architecture:** A generic `options` dict field is added to `LlmAgentConfig` and threaded through `ModelFactory` to `OllamaModel`. Profile-level and agent-level options are deep-merged. Explicit `temperature`/`max_tokens` always win.

**Tech Stack:** Python, Pydantic, ollama-python, pytest

---

### Task 1: Add `options` field to LlmAgentConfig

**Files:**
- Modify: `src/llm_orc/schemas/agent_config.py:39-42`
- Test: `tests/unit/schemas/test_agent_config.py`

**Step 1: Write the failing test**

Add to `tests/unit/schemas/test_agent_config.py`:

```python
class TestOptionsFieldAccepted:
    """Scenario: options dict accepted on LLM agent config."""

    def test_llm_agent_with_options(self) -> None:
        data: dict[str, Any] = {
            "name": "analyzer",
            "model_profile": "local-qwen",
            "options": {"num_ctx": 8192, "top_k": 20},
        }
        config = parse_agent_config(data)
        assert isinstance(config, LlmAgentConfig)
        assert config.options == {"num_ctx": 8192, "top_k": 20}

    def test_llm_agent_without_options(self) -> None:
        data: dict[str, Any] = {"name": "analyzer", "model_profile": "gpt4"}
        config = parse_agent_config(data)
        assert isinstance(config, LlmAgentConfig)
        assert config.options is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schemas/test_agent_config.py::TestOptionsFieldAccepted -v`
Expected: FAIL — `options` is rejected by `extra="forbid"`

**Step 3: Write minimal implementation**

In `src/llm_orc/schemas/agent_config.py`, add after `max_tokens` (line 40):

```python
    options: dict[str, Any] | None = None
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/schemas/test_agent_config.py::TestOptionsFieldAccepted -v`
Expected: PASS

**Step 5: Run full suite and lint**

Run: `make test && make lint`
Expected: All green

**Step 6: Commit**

```bash
git add src/llm_orc/schemas/agent_config.py tests/unit/schemas/test_agent_config.py
git commit -m "feat: add options field to LlmAgentConfig"
```

---

### Task 2: Accept `options` in OllamaModel and merge into API call

**Files:**
- Modify: `src/llm_orc/models/ollama.py:13-39`
- Test: `tests/unit/models/test_ollama_options.py` (new)

**Step 1: Write the failing test**

Create `tests/unit/models/test_ollama_options.py`:

```python
"""Tests for OllamaModel options pass-through."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_orc.models.ollama import OllamaModel


class TestOllamaOptionsPassThrough:
    """Scenario: provider options forwarded to Ollama API."""

    def test_init_stores_options(self) -> None:
        model = OllamaModel(
            model_name="qwen3:8b",
            options={"num_ctx": 8192, "top_k": 20},
        )
        assert model._options == {"num_ctx": 8192, "top_k": 20}

    def test_init_default_options_is_none(self) -> None:
        model = OllamaModel(model_name="qwen3:8b")
        assert model._options is None

    @pytest.mark.asyncio
    async def test_options_merged_into_api_call(self) -> None:
        model = OllamaModel(
            model_name="qwen3:8b",
            temperature=0.6,
            max_tokens=2000,
            options={"num_ctx": 8192, "top_k": 20},
        )

        mock_response: dict[str, Any] = {
            "message": {"content": "test response"},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        assert options["num_ctx"] == 8192
        assert options["top_k"] == 20
        assert options["temperature"] == 0.6
        assert options["num_predict"] == 2000

    @pytest.mark.asyncio
    async def test_explicit_temperature_wins_over_options(self) -> None:
        """Explicit temperature field overlays options temperature."""
        model = OllamaModel(
            model_name="qwen3:8b",
            temperature=0.3,
            options={"temperature": 0.9, "num_ctx": 4096},
        )

        mock_response: dict[str, Any] = {
            "message": {"content": "test"},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        assert options["temperature"] == 0.3
        assert options["num_ctx"] == 4096

    @pytest.mark.asyncio
    async def test_no_options_works_unchanged(self) -> None:
        """Backward compat: no options behaves exactly as before."""
        model = OllamaModel(model_name="qwen3:8b")

        mock_response: dict[str, Any] = {
            "message": {"content": "test"},
        }
        model.client = AsyncMock()
        model.client.chat = AsyncMock(return_value=mock_response)

        await model.generate_response("hello", "system prompt")

        call_kwargs = model.client.chat.call_args
        options = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        assert options is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/models/test_ollama_options.py -v`
Expected: FAIL — `OllamaModel.__init__` doesn't accept `options`

**Step 3: Write minimal implementation**

Replace `OllamaModel.__init__` and `generate_response` in `src/llm_orc/models/ollama.py`:

```python
class OllamaModel(ModelInterface):
    """Ollama model implementation."""

    def __init__(
        self,
        model_name: str = "llama2",
        host: str = "http://localhost:11434",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(temperature=temperature, max_tokens=max_tokens)
        self.model_name = model_name
        self.host = host
        self.client = ollama.AsyncClient(host=host)
        self._options = options

    # ... name property unchanged ...

    async def generate_response(self, message: str, role_prompt: str) -> str:
        """Generate response using Ollama API."""
        start_time = time.time()

        # Build options: generic options underlay, explicit fields overlay
        options: dict[str, float | int] = {}
        if self._options:
            options.update(self._options)
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens

        # ... rest unchanged ...
```

Add `from typing import Any` to imports.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/models/test_ollama_options.py -v`
Expected: PASS

**Step 5: Run full suite and lint**

Run: `make test && make lint`
Expected: All green

**Step 6: Commit**

```bash
git add src/llm_orc/models/ollama.py tests/unit/models/test_ollama_options.py
git commit -m "feat: accept provider options in OllamaModel"
```

---

### Task 3: Thread `options` through ModelFactory

**Files:**
- Modify: `src/llm_orc/core/models/model_factory.py:37-134, 307-350`
- Test: `tests/unit/core/models/test_model_factory.py`

**Step 1: Write the failing tests**

Add to `tests/unit/core/models/test_model_factory.py`:

```python
class TestOptionsPassThrough:
    """Scenario: options threaded from config to OllamaModel."""

    @pytest.fixture
    def factory(self) -> ModelFactory:
        config_manager = Mock(spec=ConfigurationManager)
        credential_storage = Mock(spec=CredentialStorage)
        credential_storage.get_auth_method.return_value = None
        return ModelFactory(config_manager, credential_storage)

    async def test_load_model_from_agent_config_passes_options(
        self, factory: ModelFactory
    ) -> None:
        """Options from agent config dict reach load_model."""
        agent_config = {
            "model": "qwen3:8b",
            "provider": "ollama",
            "options": {"num_ctx": 8192, "top_k": 20},
        }

        with patch.object(
            factory, "load_model", return_value=AsyncMock()
        ) as mock_load:
            await factory.load_model_from_agent_config(agent_config)
            mock_load.assert_called_once_with(
                "qwen3:8b",
                "ollama",
                temperature=None,
                max_tokens=None,
                options={"num_ctx": 8192, "top_k": 20},
            )

    async def test_load_model_from_agent_config_no_options(
        self, factory: ModelFactory
    ) -> None:
        """No options field passes None (backward compat)."""
        agent_config = {"model": "llama3", "provider": "ollama"}

        with patch.object(
            factory, "load_model", return_value=AsyncMock()
        ) as mock_load:
            await factory.load_model_from_agent_config(agent_config)
            mock_load.assert_called_once_with(
                "llama3",
                "ollama",
                temperature=None,
                max_tokens=None,
                options=None,
            )

    async def test_load_model_from_agent_config_merges_profile_options(
        self, factory: ModelFactory
    ) -> None:
        """Profile options merged with agent options, agent wins."""
        factory._config_manager.resolve_model_profile.return_value = (
            "qwen3:8b",
            "ollama",
        )
        factory._config_manager.get_model_profile.return_value = {
            "model": "qwen3:8b",
            "provider": "ollama",
            "options": {"num_ctx": 8192, "top_k": 40},
        }

        agent_config = {
            "model_profile": "analyst-qwen",
            "options": {"top_k": 20, "top_p": 0.8},
        }

        with patch.object(
            factory, "load_model", return_value=AsyncMock()
        ) as mock_load:
            await factory.load_model_from_agent_config(agent_config)
            mock_load.assert_called_once_with(
                "qwen3:8b",
                "ollama",
                temperature=None,
                max_tokens=None,
                options={"num_ctx": 8192, "top_k": 20, "top_p": 0.8},
            )

    async def test_load_model_forwards_options_to_ollama(
        self, factory: ModelFactory
    ) -> None:
        """load_model passes options to OllamaModel constructor."""
        model = await factory.load_model(
            "qwen3:8b",
            "ollama",
            options={"num_ctx": 8192},
        )

        assert isinstance(model, OllamaModel)
        assert model._options == {"num_ctx": 8192}

    async def test_load_model_non_ollama_ignores_options(
        self, factory: ModelFactory
    ) -> None:
        """Non-Ollama providers don't break when options is passed."""
        factory._credential_storage.get_auth_method.return_value = "api_key"
        factory._credential_storage.get_api_key.return_value = "test-key"

        model = await factory.load_model(
            "claude-3-sonnet",
            "anthropic",
            options={"num_ctx": 8192},
        )

        assert isinstance(model, ClaudeModel)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/core/models/test_model_factory.py::TestOptionsPassThrough -v`
Expected: FAIL — `load_model` doesn't accept `options`

**Step 3: Write minimal implementation**

Changes to `src/llm_orc/core/models/model_factory.py`:

a) `load_model_from_agent_config` — extract options, merge with profile options:

```python
async def load_model_from_agent_config(
    self, agent_config: dict[str, Any]
) -> ModelInterface:
    temperature: float | None = agent_config.get("temperature")
    max_tokens: int | None = agent_config.get("max_tokens")
    agent_options: dict[str, Any] | None = agent_config.get("options")

    if agent_config.get("model_profile"):
        profile_name = agent_config["model_profile"]
        resolved_model, resolved_provider = (
            self._config_manager.resolve_model_profile(profile_name)
        )
        # Merge profile options with agent options (agent wins)
        profile = self._config_manager.get_model_profile(profile_name)
        profile_options = (profile or {}).get("options")
        merged_options = _merge_options(profile_options, agent_options)
        return await self.load_model(
            resolved_model,
            resolved_provider,
            temperature=temperature,
            max_tokens=max_tokens,
            options=merged_options,
        )

    model: str | None = agent_config.get("model")
    provider: str | None = agent_config.get("provider")

    if not model:
        raise ValueError(
            "Agent configuration must specify either 'model_profile' or 'model'"
        )

    return await self.load_model(
        model,
        provider,
        temperature=temperature,
        max_tokens=max_tokens,
        options=agent_options,
    )
```

b) `load_model` — accept and thread `options`:

```python
async def load_model(
    self,
    model_name: str,
    provider: str | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    options: dict[str, Any] | None = None,
) -> ModelInterface:
    if model_name.startswith("mock"):
        return MockModel(model_name)

    storage = self._credential_storage
    auth_method = _resolve_authentication_method(model_name, provider, storage)

    if not auth_method:
        return _handle_no_authentication(
            model_name,
            provider,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
        )

    return _create_authenticated_model(
        model_name,
        provider,
        auth_method,
        storage,
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

c) `_handle_no_authentication` — accept and forward `options`:

```python
def _handle_no_authentication(
    model_name: str,
    provider: str | None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    options: dict[str, Any] | None = None,
) -> ModelInterface:
    if provider == "ollama":
        return OllamaModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
        )
    elif provider:
        raise ValueError(...)
    else:
        logger.info(...)
        return OllamaModel(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            options=options,
        )
```

d) Add module-level helper:

```python
def _merge_options(
    profile_options: dict[str, Any] | None,
    agent_options: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge profile and agent options dicts. Agent keys win."""
    if not profile_options and not agent_options:
        return None
    return {**(profile_options or {}), **(agent_options or {})}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/core/models/test_model_factory.py::TestOptionsPassThrough -v`
Expected: PASS

**Step 5: Run full suite and lint**

Run: `make test && make lint`
Expected: All green. Existing tests still pass — they don't assert on `options=` so the new kwarg with default `None` is backward-compatible.

**Step 6: Commit**

```bash
git add src/llm_orc/core/models/model_factory.py tests/unit/core/models/test_model_factory.py
git commit -m "feat: thread options through ModelFactory to OllamaModel"
```

---

### Task 4: Verify end-to-end with profile integration test

**Files:**
- Test: `tests/unit/schemas/test_agent_config.py`

**Step 1: Write integration test**

Add to `tests/unit/schemas/test_agent_config.py`:

```python
class TestOptionsFlowFromProfileToModel:
    """Scenario: options from profile YAML reach OllamaModel."""

    @pytest.mark.asyncio
    async def test_profile_options_reach_ollama_model(self) -> None:
        """End-to-end: profile with options -> factory -> OllamaModel."""
        from unittest.mock import Mock

        from llm_orc.core.models.model_factory import ModelFactory

        config_manager = Mock()
        config_manager.resolve_model_profile.return_value = (
            "qwen3:8b",
            "ollama",
        )
        config_manager.get_model_profile.return_value = {
            "model": "qwen3:8b",
            "provider": "ollama",
            "options": {"num_ctx": 8192, "top_k": 40},
        }

        credential_storage = Mock()
        credential_storage.get_auth_method.return_value = None

        factory = ModelFactory(config_manager, credential_storage)

        agent = LlmAgentConfig(
            name="analyzer",
            model_profile="analyst-qwen",
            options={"top_k": 20, "top_p": 0.8},
        )
        config_dict = agent.model_dump()

        model = await factory.load_model_from_agent_config(config_dict)

        from llm_orc.models.ollama import OllamaModel

        assert isinstance(model, OllamaModel)
        assert model._options == {"num_ctx": 8192, "top_k": 20, "top_p": 0.8}
```

**Step 2: Run test**

Run: `uv run pytest tests/unit/schemas/test_agent_config.py::TestOptionsFlowFromProfileToModel -v`
Expected: PASS (all pieces wired together)

**Step 3: Run full suite and lint**

Run: `make test && make lint`
Expected: All green

**Step 4: Commit**

```bash
git add tests/unit/schemas/test_agent_config.py
git commit -m "test: add end-to-end integration test for options pass-through"
```
