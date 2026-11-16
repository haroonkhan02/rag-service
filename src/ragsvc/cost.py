"""What an answer costs: list prices and the arithmetic.

Prices are Anthropic's list prices in USD per million tokens (input, output) for requests
on the Claude API. Output includes the model's thinking tokens. A local model has no price
per token; its cost is the time it keeps the GPU busy.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def cost_usd(model: str, input_tokens: float, output_tokens: float) -> float:
    """0.0 for models without a list price (local ones)."""
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1e6
