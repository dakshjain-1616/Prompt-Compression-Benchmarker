"""Drop-in SDK wrappers that compress prompts before every API call.

Usage — Anthropic:
    from pcb.middleware import CompressingAnthropic
    client = CompressingAnthropic(compressor="llmlingua2", rate=0.45)
    response = client.messages.create(model="claude-opus-4-7", messages=[...], max_tokens=1024)
    print(client.stats)   # tokens saved, calls made, reduction %

Usage — OpenAI:
    from pcb.middleware import CompressingOpenAI
    client = CompressingOpenAI(compressor="tfidf", rate=0.4)
    response = client.chat.completions.create(model="gpt-4.1", messages=[...])
"""

from pcb.middleware.anthropic_client import CompressingAnthropic
from pcb.middleware.openai_client import CompressingOpenAI

__all__ = ["CompressingAnthropic", "CompressingOpenAI"]
