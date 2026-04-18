"""LLM-as-judge evaluator via OpenRouter — supports all major 2026 models.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    pcb run --llm-judge
    pcb run --llm-judge --judge-model google/gemini-2.5-pro
"""

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Optional


def _key_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

import httpx

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Live-verified April 2026 model list — IDs pulled directly from OpenRouter API
SUPPORTED_MODELS: Dict[str, str] = {
    # ── Anthropic Claude 4.x (live on OpenRouter) ─────────────────────────
    "claude-opus-4.7":          "anthropic/claude-opus-4.7",       # Latest Opus
    "claude-opus-4.6":          "anthropic/claude-opus-4.6",       # #6 by token volume
    "claude-opus-4.6-fast":     "anthropic/claude-opus-4.6-fast",
    "claude-opus-4.5":          "anthropic/claude-opus-4.5",
    "claude-sonnet-4.6":        "anthropic/claude-sonnet-4.6",     # #2 globally by usage
    "claude-sonnet-4.5":        "anthropic/claude-sonnet-4.5",
    "claude-haiku-4.5":         "anthropic/claude-haiku-4.5",
    # ── OpenAI GPT-5.x (live) ────────────────────────────────────────────
    "gpt-5.4":                  "openai/gpt-5.4",                  # #7 by volume
    "gpt-5.4-pro":              "openai/gpt-5.4-pro",
    "gpt-5.4-mini":             "openai/gpt-5.4-mini",
    "gpt-5.1":                  "openai/gpt-5.1",
    "gpt-5":                    "openai/gpt-5",
    "gpt-4.1":                  "openai/gpt-4.1",
    "gpt-4.1-mini":             "openai/gpt-4.1-mini",
    "gpt-4.1-nano":             "openai/gpt-4.1-nano",
    "o4-mini":                  "openai/o4-mini",
    "o4-mini-high":             "openai/o4-mini-high",
    "o3":                       "openai/o3",
    "o3-pro":                   "openai/o3-pro",
    # ── Google Gemini 3.x (live) ──────────────────────────────────────────
    "gemini-3.1-pro":           "google/gemini-3.1-pro-preview",   # #8 by volume
    "gemini-3.1-flash-lite":    "google/gemini-3.1-flash-lite-preview",
    "gemini-3-flash":           "google/gemini-3-flash-preview",
    "gemini-2.5-pro":           "google/gemini-2.5-pro",
    "gemini-2.5-flash":         "google/gemini-2.5-flash",
    "gemini-2.5-flash-lite":    "google/gemini-2.5-flash-lite",
    "gemma-4-31b":              "google/gemma-4-31b-it",
    # ── xAI Grok 4.x (live) ──────────────────────────────────────────────
    "grok-4.20":                "x-ai/grok-4.20",                  # Latest Grok
    "grok-4.1-fast":            "x-ai/grok-4.1-fast",
    "grok-4":                   "x-ai/grok-4",
    "grok-4-fast":              "x-ai/grok-4-fast",
    "grok-3":                   "x-ai/grok-3",
    "grok-3-mini":              "x-ai/grok-3-mini",
    # ── Qwen 3.x / 3.5 / 3.6 (live) ──────────────────────────────────────
    "qwen3.6-plus":             "qwen/qwen3.6-plus",               # #5 by volume
    "qwen3.5-397b":             "qwen/qwen3.5-397b-a17b",
    "qwen3.5-plus":             "qwen/qwen3.5-plus-02-15",
    "qwen3-235b":               "qwen/qwen3-235b-a22b",
    "qwen3-coder":              "qwen/qwen3-coder",
    "qwen3-max":                "qwen/qwen3-max",
    "qwen3-max-thinking":       "qwen/qwen3-max-thinking",
    # ── DeepSeek V3.x / R1 (live) ────────────────────────────────────────
    "deepseek-v3.2":            "deepseek/deepseek-v3.2",          # #4 by volume
    "deepseek-v3.2-exp":        "deepseek/deepseek-v3.2-exp",
    "deepseek-v3.1":            "deepseek/deepseek-chat-v3.1",
    "deepseek-r1-0528":         "deepseek/deepseek-r1-0528",
    "deepseek-r1":              "deepseek/deepseek-r1",
    # ── MiniMax M2.x (live) — #3 globally ────────────────────────────────
    "minimax-m2.7":             "minimax/minimax-m2.7",
    "minimax-m2.5":             "minimax/minimax-m2.5",
    "minimax-m2.1":             "minimax/minimax-m2.1",
    # ── Xiaomi MiMo-V2 (live) — #1 globally ──────────────────────────────
    "mimo-v2-pro":              "xiaomi/mimo-v2-pro",
    "mimo-v2-flash":            "xiaomi/mimo-v2-flash",
    # ── Moonshot Kimi K2 (live) — #9 by volume ───────────────────────────
    "kimi-k2.5":                "moonshotai/kimi-k2.5",
    "kimi-k2-thinking":         "moonshotai/kimi-k2-thinking",
    "kimi-k2":                  "moonshotai/kimi-k2",
    # ── Meta Llama 4 (live) ───────────────────────────────────────────────
    "llama-4-maverick":         "meta-llama/llama-4-maverick",
    "llama-4-scout":            "meta-llama/llama-4-scout",
    # ── Mistral 2026 (live) ───────────────────────────────────────────────
    "mistral-large-3":          "mistralai/mistral-large-2512",
    "mistral-medium-3.1":       "mistralai/mistral-medium-3.1",
    "devstral-medium":          "mistralai/devstral-medium",
    "codestral-2508":           "mistralai/codestral-2508",
    # ── NVIDIA Nemotron 3 (live) ──────────────────────────────────────────
    "nemotron-super-120b":      "nvidia/nemotron-3-super-120b-a12b",
    "nemotron-nano-30b":        "nvidia/nemotron-3-nano-30b-a3b",
    # ── Z.ai GLM 5.x (live) ──────────────────────────────────────────────
    "glm-5.1":                  "z-ai/glm-5.1",
    "glm-5":                    "z-ai/glm-5",
    "glm-4.7":                  "z-ai/glm-4.7",
}

DEFAULT_MODEL = "claude-sonnet-4.6"

# ── Prompts per task type ──────────────────────────────────────────────────────

_RAG_PROMPT = """\
You are an expert evaluator assessing information preservation after text compression.

TASK: Determine whether the compressed context retains enough information to correctly answer the question.

Question: {question}
Expected answer: {answer}
Compressed context:
\"\"\"
{compressed_context}
\"\"\"

Scoring rubric:
- 1.0: The compressed context clearly contains all information needed to answer correctly.
- 0.7-0.9: The answer can be inferred with minor gaps.
- 0.4-0.6: Some relevant information present but answer is uncertain.
- 0.1-0.3: Little relevant information; answer likely wrong.
- 0.0: The compressed context is useless for answering the question.

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""

_SUMMARIZATION_PROMPT = """\
You are an expert evaluator assessing summarization quality after text compression.

TASK: Rate how well the generated summary (from compressed text) covers the same key information as the reference summary.

Reference summary: {reference_summary}
Generated summary from compressed text: {generated_summary}

Scoring rubric:
- 1.0: All key facts from the reference are present in the generated summary.
- 0.7-0.9: Most key facts preserved, minor omissions.
- 0.4-0.6: About half the key facts preserved.
- 0.1-0.3: Few key facts preserved.
- 0.0: No overlap with reference summary.

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""

_CODING_PROMPT = """\
You are an expert software engineer evaluating context compression for coding tasks.

TASK: Determine whether the compressed context retains enough information (imports, type definitions, helper functions, constraints) to correctly implement the required function.

Task description: {docstring}
Expected solution:
\"\"\"
{solution}
\"\"\"
Compressed context:
\"\"\"
{compressed_context}
\"\"\"

Scoring rubric:
- 1.0: All necessary context is present — imports, types, helpers, constraints.
- 0.7-0.9: Minor context gaps; solution likely correct but may miss edge cases.
- 0.4-0.6: Key helpers or types are missing; implementation would be incomplete.
- 0.1-0.3: Most context lost; solution would be incorrect or non-compilable.
- 0.0: Context is useless for the implementation task.

Respond with ONLY a single decimal number between 0.0 and 1.0. No explanation."""


def _resolve_model(model_alias: str) -> str:
    """Resolve a short alias or pass-through a full model ID."""
    return SUPPORTED_MODELS.get(model_alias, model_alias)


def _parse_score(text: str) -> Optional[float]:
    """Extract a 0.0-1.0 float from model response."""
    text = text.strip()
    # Try direct parse
    try:
        val = float(text)
        return max(0.0, min(1.0, val))
    except ValueError:
        pass
    # Extract first decimal/integer in range
    match = re.search(r'\b(1\.0|0\.\d+|[01])\b', text)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass
    return None


class LLMJudge:
    """Calls OpenRouter to score compressed output quality."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.model_id = _resolve_model(model)
        self.model_alias = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache: Dict[str, float] = {}

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass --openrouter-key to pcb run."
            )

    def _call(self, prompt: str) -> Optional[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/prompt-compression-benchmarker",
            "X-Title": "Prompt Compression Benchmarker",
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16,
            "temperature": 0.0,
        }

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = httpx.post(
                    OPENROUTER_BASE,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                score = _parse_score(content)
                return score
            except (httpx.HTTPStatusError, httpx.RequestError, KeyError, IndexError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
        print(f"  [judge warn] {self.model_id} failed after {self.max_retries} attempts: {last_err}")
        return None

    def score_rag(
        self,
        compressed_context: str,
        question: str,
        answer: str,
        sample_id: str = "",
    ) -> Optional[float]:
        key = f"rag:{sample_id}:{_key_hash(compressed_context)}"
        if key in self._cache:
            return self._cache[key]
        prompt = _RAG_PROMPT.format(
            question=question,
            answer=answer,
            compressed_context=compressed_context[:3000],
        )
        score = self._call(prompt)
        if score is not None:
            self._cache[key] = score
        return score

    def score_summarization(
        self,
        generated_summary: str,
        reference_summary: str,
        sample_id: str = "",
    ) -> Optional[float]:
        key = f"sum:{sample_id}:{_key_hash(generated_summary)}"
        if key in self._cache:
            return self._cache[key]
        prompt = _SUMMARIZATION_PROMPT.format(
            reference_summary=reference_summary,
            generated_summary=generated_summary,
        )
        score = self._call(prompt)
        if score is not None:
            self._cache[key] = score
        return score

    def score_coding(
        self,
        compressed_context: str,
        docstring: str,
        solution: str,
        sample_id: str = "",
    ) -> Optional[float]:
        key = f"code:{sample_id}:{_key_hash(compressed_context)}"
        if key in self._cache:
            return self._cache[key]
        prompt = _CODING_PROMPT.format(
            docstring=docstring,
            solution=solution[:1000],
            compressed_context=compressed_context[:3000],
        )
        score = self._call(prompt)
        if score is not None:
            self._cache[key] = score
        return score

    @property
    def name(self) -> str:
        return f"llm_judge({self.model_alias})"
