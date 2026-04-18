"""Two-pass pipeline compressor: sentence selection then word-level pruning.

Pass 1 drops low-value or query-irrelevant sentences.
Pass 2 prunes stopwords and filler tokens from surviving sentences.
The two passes target different types of redundancy so the reductions compound.
"""

import re
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pcb.compressors.base import BaseCompressor, CompressionResult
from pcb.utils.stopwords import STOPWORDS as _STOPWORDS
from pcb.utils.token_counter import TokenCounter

_FILLER_PHRASES = re.compile(
    r"\b(it is worth noting that|as we can see|in order to|the fact that|"
    r"it should be noted that|needless to say|as a matter of fact|"
    r"at the end of the day|in terms of|with respect to|due to the fact that|"
    r"for the purpose of|in the event that|in light of the fact that|"
    r"it is important to note that|as mentioned (above|earlier|previously)|"
    r"based on the (above|foregoing))\b",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _score_by_query(sentences: list[str], query: str) -> np.ndarray:
    """Score each sentence by cosine similarity to the query.

    Query-aware scoring must refit because the query can introduce vocabulary
    that isn't present in the sentence corpus alone.
    """
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        matrix = vectorizer.fit_transform([query] + sentences)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except Exception:
        scores = np.ones(len(sentences))
    return scores


def _score_from_matrix(matrix) -> np.ndarray:
    """Score each sentence by its aggregate TF-IDF weight using a pre-fit matrix."""
    try:
        return np.array(matrix.sum(axis=1)).flatten()
    except Exception:
        return np.ones(matrix.shape[0])


def _dedup_sentences_with_matrix(sentences: list[str], threshold: float = 0.90):
    """Remove near-duplicate sentences. Returns (kept_sentences, kept_matrix).

    The returned matrix is the TF-IDF encoding of the surviving sentences so
    downstream scoring can reuse the fit instead of refitting a second vectorizer.
    """
    if len(sentences) <= 2:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
        try:
            matrix = vectorizer.fit_transform(sentences)
            return sentences, matrix
        except Exception:
            return sentences, None

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_df=0.95)
    try:
        matrix = vectorizer.fit_transform(sentences)
    except Exception:
        return sentences, None

    kept = [0]
    for i in range(1, len(sentences)):
        sims = cosine_similarity(matrix[i : i + 1], matrix[np.array(kept)]).flatten()
        if sims.max() < threshold:
            kept.append(i)

    if len(kept) == len(sentences):
        return sentences, matrix

    kept_arr = np.array(kept)
    return [sentences[i] for i in kept], matrix[kept_arr]


def _prune_words(text: str, word_rate: float) -> str:
    """Remove stopwords and low-content tokens to hit word_rate token reduction."""
    words = text.split()
    if not words:
        return text

    target_keep = max(1, int(len(words) * (1.0 - word_rate)))
    word_freq: dict[str, int] = {}
    for w in words:
        clean = re.sub(r"[^a-z0-9]", "", w.lower())
        word_freq[clean] = word_freq.get(clean, 0) + 1

    scored = []
    for i, w in enumerate(words):
        clean = re.sub(r"[^a-z0-9]", "", w.lower())
        is_stop = clean in _STOPWORDS
        len_score = min(1.0, len(clean) / 8)
        freq_score = 1.0 / (word_freq.get(clean, 1) + 1)
        score = (0.0 if is_stop else 0.5) + 0.3 * len_score + 0.2 * freq_score
        scored.append((i, w, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    keep = {idx for idx, _, _ in scored[:target_keep]}
    return " ".join(w for i, w, _ in sorted(scored, key=lambda x: x[0]) if i in keep) or words[0]


class PipelineCompressor(BaseCompressor):
    """Two-pass compressor: sentence selection then word-level pruning.

    Pass 1 (sentence level) removes 65% of the overall reduction budget.
    Pass 2 (word level) removes the remaining 35%.
    When a query is provided, Pass 1 scores by relevance to the query instead
    of by corpus-wide TF-IDF, which makes it safe to be much more aggressive.
    """

    supports_query = True

    def __init__(self, config: Optional[dict] = None):
        super().__init__(name="smart", config=config)
        self.token_counter = TokenCounter()

    def initialize(self) -> None:
        self._is_initialized = True

    def compress(self, text: str, **kwargs: Any) -> CompressionResult:
        """Compress text in two passes.

        Kwargs:
            rate  (float): fraction of tokens to remove, default 0.5.
            query (str):   optional query for relevance-based sentence scoring.
        """
        self._ensure_initialized()
        self._validate_kwargs(kwargs)

        rate: float = kwargs.get("rate", 0.5)
        query: Optional[str] = kwargs.get("query", None)

        original_tokens = self.token_counter.count_tokens(text)

        # --- pre-pass: strip known filler phrases ---
        cleaned = _FILLER_PHRASES.sub("", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        sentences = _split_sentences(cleaned)
        if len(sentences) <= 1:
            return CompressionResult(
                original_text=text,
                compressed_text=cleaned,
                original_tokens=original_tokens,
                compressed_tokens=self.token_counter.count_tokens(cleaned),
                compression_ratio=max(0.0, 1.0 - self.token_counter.count_tokens(cleaned) / max(original_tokens, 1)),
                metadata={"method": "smart", "passes": "filler_only"},
            )

        # --- pass 1: deduplicate then sentence selection ---
        # Dedup fits TF-IDF once; the resulting matrix is reused for corpus
        # scoring to avoid a second fit on the same data.
        sentences, kept_matrix = _dedup_sentences_with_matrix(sentences)

        # Word pass will remove word_rate of remaining tokens.
        # Sentence pass must therefore keep sentence_keep fraction so that:
        #   sentence_keep * (1 - word_rate) = (1 - rate)
        word_rate = min(0.35, rate * 0.40)
        sentence_keep = (1.0 - rate) / max(0.01, 1.0 - word_rate)
        # Upper bound 1.0 — earlier 0.85 cap caused low-rate requests
        # (e.g. rate=0.1) to over-prune sentences regardless of target.
        sentence_keep = max(0.20, min(1.0, sentence_keep))
        n_keep = max(1, int(len(sentences) * sentence_keep))

        if query:
            scores = _score_by_query(sentences, query)
        elif kept_matrix is not None:
            scores = _score_from_matrix(kept_matrix)
        else:
            scores = np.ones(len(sentences))

        top_idx = sorted(np.argsort(scores)[-n_keep:])
        selected_text = " ".join(sentences[i] for i in top_idx)

        # --- pass 2: word-level pruning on surviving sentences ---
        if word_rate > 0.05:
            compressed_text = _prune_words(selected_text, word_rate)
        else:
            compressed_text = selected_text

        compressed_tokens = self.token_counter.count_tokens(compressed_text)
        ratio = max(0.0, 1.0 - compressed_tokens / max(original_tokens, 1))

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            metadata={
                "method": "smart",
                "query_aware": query is not None,
                "sentences_in": len(_split_sentences(cleaned)),
                "sentences_kept": n_keep,
                "word_prune_rate": round(word_rate, 3),
                "passes": "dedup+sentence+word",
            },
        )
