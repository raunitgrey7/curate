"""Optional rewording of explanations with an Anthropic model.

Disabled unless ``CURATE_LLM_EXPLAIN`` is on and ``ANTHROPIC_API_KEY`` is set. The deterministic text is
the input and the fallback: any SDK error, refusal or empty reply returns the original sentence, so the
service never depends on the network. The model id comes from ``CURATE_LLM_MODEL`` (``claude-opus-5-5`` by
default; ``claude-haiku-5-5`` is the cheaper choice for this one-sentence task).
"""

from __future__ import annotations

import logging
import os

from curate.config import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You rewrite one recommendation explanation for a streaming or shopping app. Keep every number and every "
    "title exactly as given, keep it to one sentence, do not add claims, and answer with the sentence only."
)


def llm_available(settings: Settings) -> bool:
    """True when the flag is on, a key is present and the optional dependency is installed."""
    if not settings.llm_explanations or not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def reword_explanation(text: str, settings: Settings) -> str:
    """Return a reworded explanation, or ``text`` unchanged when the LLM path is off or fails."""
    if not llm_available(settings):
        return text
    import anthropic

    try:
        response = anthropic.Anthropic().messages.create(
            model=settings.llm_model,
            max_tokens=200,
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
    except anthropic.RateLimitError:
        logger.warning("llm explanation rate limited, using deterministic text")
        return text
    except anthropic.APIStatusError as exc:
        logger.warning("llm explanation failed", extra={"status": exc.status_code})
        return text
    except anthropic.APIConnectionError:
        logger.warning("llm explanation unreachable, using deterministic text")
        return text
    if response.stop_reason == "refusal":
        return text
    reworded = " ".join(block.text for block in response.content if block.type == "text").strip()
    return reworded or text
