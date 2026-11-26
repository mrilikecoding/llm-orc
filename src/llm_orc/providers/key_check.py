"""Check that a cloud provider accepts a stored API key.

One list-models request per check: it names no model and spends no tokens.
The request is bounded by ``KEY_CHECK_TIMEOUT_SECONDS`` and never retried.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import anthropic
import httpx
from anthropic import AsyncAnthropic
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

if TYPE_CHECKING:
    from llm_orc.core.auth.authentication import CredentialStorage

KEY_CHECK_TIMEOUT_SECONDS = 5.0

KeyState = Literal["accepted", "rejected", "inconclusive"]

# Google documents no status for a bad key on this call (nothing in the
# google-genai package says a bad key is a 400), so only 401 and 403, which
# mean the credential was refused, are "rejected".
_GOOGLE_REJECTED_CODES = frozenset({401, 403})


@dataclass(frozen=True)
class KeyCheck:
    """What one list-models request said about a key.

    ``rejected`` only when the provider refused the credential itself;
    ``inconclusive`` for anything else that is not a success.
    """

    state: KeyState
    reason: str


def checkable_api_key(storage: "CredentialStorage", provider: str) -> str | None:
    """The stored API key worth sending to ``provider``'s list endpoint.

    None for an OAuth credential, or the claude-cli path the model factory
    also accepts in the api_key slot: there is nothing to send.
    """
    if storage.get_auth_method(provider) != "api_key":
        return None
    api_key = storage.get_api_key(provider)
    if api_key is None or api_key.startswith("/"):
        return None
    return api_key


async def check_api_key(provider: str, api_key: str) -> KeyCheck:
    """Ask ``provider`` to list its models with ``api_key``; never raises."""
    if not _is_header_safe(api_key):
        return KeyCheck("rejected", f"malformed key: cannot be sent to {provider}")
    try:
        if provider == "anthropic-api":
            return await _check_anthropic(api_key)
        return await _check_google(api_key)
    except Exception as e:
        return _inconclusive(provider, type(e).__name__)


def _is_header_safe(api_key: str) -> bool:
    """Printable ASCII only: the key travels in a request header."""
    return bool(api_key) and all(0x20 <= ord(c) < 0x7F for c in api_key)


def _accepted(provider: str) -> KeyCheck:
    return KeyCheck("accepted", f"{provider} accepted the API key")


def _rejected(provider: str, detail: str) -> KeyCheck:
    return KeyCheck("rejected", f"{provider} rejected the API key ({detail})")


def _inconclusive(provider: str, detail: str) -> KeyCheck:
    return KeyCheck(
        "inconclusive", f"could not verify the key with {provider}: {detail}"
    )


async def _check_anthropic(api_key: str) -> KeyCheck:
    provider = "anthropic-api"
    client = AsyncAnthropic(
        api_key=api_key, timeout=KEY_CHECK_TIMEOUT_SECONDS, max_retries=0
    )
    try:
        await client.models.list(limit=1)
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        return _rejected(provider, f"HTTP {e.status_code}")
    except anthropic.APITimeoutError:
        return _inconclusive(
            provider, f"APITimeoutError after {KEY_CHECK_TIMEOUT_SECONDS}s"
        )
    except anthropic.APIConnectionError as e:
        return _inconclusive(provider, type(e).__name__)
    except anthropic.APIStatusError as e:
        return _inconclusive(provider, f"HTTP {e.status_code}")
    finally:
        await client.close()
    return _accepted(provider)


async def _check_google(api_key: str) -> KeyCheck:
    provider = "google-gemini"
    client = genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(
            timeout=int(KEY_CHECK_TIMEOUT_SECONDS * 1000)
        ),
    )
    try:
        await client.aio.models.list(config={"page_size": 1})
    except genai_errors.APIError as e:
        if e.code in _GOOGLE_REJECTED_CODES:
            return _rejected(provider, f"HTTP {e.code}")
        return _inconclusive(provider, f"HTTP {e.code}")
    except httpx.TimeoutException:
        return _inconclusive(provider, f"timed out after {KEY_CHECK_TIMEOUT_SECONDS}s")
    except httpx.HTTPError as e:
        return _inconclusive(provider, type(e).__name__)
    finally:
        await client.aio.aclose()
    return _accepted(provider)
