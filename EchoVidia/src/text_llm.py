from __future__ import annotations

import os
import time
from typing import Optional

from openai import OpenAI


def t2t_generate(
    prompt: str,
    *,
    model: str = "openai/gpt-4.1-mini",
    max_tokens: int = 4095,
    temperature: float = 0.0,
    sleep_s: float = 5.0,
    openrouter_api_key: Optional[str] = None,
) -> str:
    """
    Text-only LLM generation via OpenRouter.

    Note: kept in a dedicated module so importing it does not eagerly load audio models.
    """

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key or os.getenv("OPENROUTER_API"),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if sleep_s:
        time.sleep(sleep_s)

    return response.choices[0].message.content

