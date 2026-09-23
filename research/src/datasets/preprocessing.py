"""
Text Preprocessing for Legal NLP Research Framework.
Handles tokenization, cleaning, sliding window chunking.
"""

import re
import logging
import unicodedata
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Configurable text preprocessor for legal documents.
    Integrates with HuggingFace tokenizers.
    """

    def __init__(self, config: dict):
        prep_cfg = config.get("preprocessing", {})
        self.max_length = prep_cfg.get("max_length", 512)
        self.truncation = prep_cfg.get("truncation", True)
        self.padding = prep_cfg.get("padding", "max_length")
        self.lowercase = prep_cfg.get("lowercase", False)
        self.remove_stopwords = prep_cfg.get("remove_stopwords", False)
        self.unicode_normalize = prep_cfg.get("unicode_normalize", True)
        sw_cfg = prep_cfg.get("sliding_window", {})
        self.use_sliding_window = sw_cfg.get("enabled", True)
        self.sw_stride = sw_cfg.get("stride", 256)
        self.sw_aggregate = sw_cfg.get("aggregate", "mean")

        self._stopwords: Optional[set] = None
        self._tokenizer = None

    def set_tokenizer(self, tokenizer):
        """Attach a HuggingFace tokenizer for token-level operations."""
        self._tokenizer = tokenizer

    def clean_text(self, text: str) -> str:
        """Apply text cleaning pipeline."""
        if self.unicode_normalize:
            text = unicodedata.normalize("NFKC", text)

        # Remove null bytes and control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        if self.lowercase:
            text = text.lower()

        if self.remove_stopwords:
            text = self._strip_stopwords(text)

        return text

    def tokenize(self, text: str, add_special_tokens: bool = True) -> dict:
        """
        Tokenize a single text using the attached HuggingFace tokenizer.
        If the text exceeds max_length and sliding_window is enabled,
        returns multiple chunks.
        """
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer attached. Call set_tokenizer() first.")

        text = self.clean_text(text)
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        effective_max = self.max_length - (2 if add_special_tokens else 0)

        if len(tokens) <= effective_max or not self.use_sliding_window:
            return self._tokenizer(
                text,
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                return_tensors="pt",
            )

        # Sliding window chunking for long documents
        chunks = []
        for start in range(0, len(tokens), self.sw_stride):
            chunk = tokens[start : start + effective_max]
            if len(chunk) < 10:
                break
            chunk_text = self._tokenizer.decode(chunk)
            enc = self._tokenizer(
                chunk_text,
                max_length=self.max_length,
                truncation=True,
                padding=self.padding,
                return_tensors="pt",
            )
            chunks.append(enc)

        return {"chunks": chunks, "n_chunks": len(chunks)}

    def batch_clean(self, texts: list[str]) -> list[str]:
        """Clean a batch of texts."""
        return [self.clean_text(t) for t in texts]

    def get_token_lengths(self, texts: list[str]) -> list[int]:
        """Return token length for each text (requires tokenizer)."""
        if self._tokenizer is None:
            # Approximate by whitespace tokens
            return [len(t.split()) for t in texts]
        cleaned = self.batch_clean(texts)
        return [
            len(self._tokenizer.encode(t, add_special_tokens=False))
            for t in cleaned
        ]

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _strip_stopwords(self, text: str) -> str:
        if self._stopwords is None:
            try:
                import nltk

                nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords

                self._stopwords = set(stopwords.words("english"))
            except Exception:
                self._stopwords = set()
        words = text.split()
        return " ".join(w for w in words if w.lower() not in self._stopwords)
