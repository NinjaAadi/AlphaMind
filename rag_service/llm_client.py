"""Ollama client for free local LLM (Llama)."""
import logging
from typing import Optional

import httpx

from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


def ollama_generate(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
) -> str:
    """
    Call Ollama generate API. Requires Ollama running locally with a model pulled:
      ollama run llama3.2
    """
    model = model or OLLAMA_MODEL
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False}
    if system:
        body["system"] = system
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama at {OLLAMA_HOST}. Is Ollama running? Run: ollama run {model}")
        raise RuntimeError(f"Ollama not available: {e}") from e
    except httpx.TimeoutException as e:
        logger.warning(f"Ollama generate timed out after {OLLAMA_TIMEOUT}s")
        raise RuntimeError(
            f"LLM request timed out after {OLLAMA_TIMEOUT}s. Try a shorter question or try again (first run loads the model)."
        ) from e
    except Exception as e:
        logger.exception("Ollama generate failed")
        raise


def ollama_health() -> bool:
    """Check if Ollama is reachable."""
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
