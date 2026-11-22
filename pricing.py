"""
pricing.py — the one Gemini Flash model id and its token prices, shared by the
eval and the agent so both report cost the same way.
"""

# Gemini 2.5 Flash for every LLM call (answers, judges, the agent)
FLASH_MODEL_ID = "gemini-2.5-flash"

# Gemini 2.5 Flash paid-tier list price, USD per token (text input / output).
# Output includes thinking tokens — langchain-google-genai folds them into output_tokens.
PRICE_IN_PER_TOKEN  = 0.30 / 1_000_000
PRICE_OUT_PER_TOKEN = 2.50 / 1_000_000


def cost(input_tokens: float, output_tokens: float) -> float:
    return input_tokens * PRICE_IN_PER_TOKEN + output_tokens * PRICE_OUT_PER_TOKEN


def usage(message) -> tuple:
    """(input_tokens, output_tokens) from a LangChain AIMessage, 0s if absent."""
    um = getattr(message, "usage_metadata", None) or {}
    return int(um.get("input_tokens", 0)), int(um.get("output_tokens", 0))
