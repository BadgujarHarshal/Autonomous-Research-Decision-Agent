#  app/tools/summarizer.py

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

from app.tools.base import BaseTool


class SummarizerTool(BaseTool):
    name = "summarize"
    description = (
        "Creates a deterministic extractive summary from supplied text "
        "by selecting sentences containing the strongest recurring terms."
    )

    _stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "from",
        "that",
        "this",
        "it",
        "be",
        "at",
        "which",
        "has",
        "have",
        "had",
    }

    @property
    def argument_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to summarize.",
                },
                "max_sentences": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
            },
            "required": ["text"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        self.validate_arguments(arguments)

        text = arguments.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")

        max_sentences = arguments.get("max_sentences", 3)

        if not isinstance(max_sentences, int):
            raise ValueError("max_sentences must be an integer.")

        if not 1 <= max_sentences <= 10:
            raise ValueError(
                "max_sentences must be between 1 and 10."
            )

        sentences = self._split_sentences(text)

        if len(sentences) <= max_sentences:
            return {
                "summary": " ".join(sentences),
                "selected_sentences": len(sentences),
                "total_sentences": len(sentences),
            }

        frequencies = Counter()

        for sentence in sentences:
            words = self._words(sentence)

            for word in words:
                if word not in self._stopwords:
                    frequencies[word] += 1

        scored = []

        for index, sentence in enumerate(sentences):
            words = self._words(sentence)

            score = sum(
                frequencies[word]
                for word in words
                if word not in self._stopwords
            )

            scored.append(
                (
                    score,
                    index,
                    sentence,
                )
            )

        selected = sorted(
            scored,
            key=lambda item: item[0],
            reverse=True,
        )[:max_sentences]

        selected.sort(key=lambda item: item[1])

        summary = " ".join(
            item[2]
            for item in selected
        )

        return {
            "summary": summary,
            "selected_sentences": len(selected),
            "total_sentences": len(sentences),
        }

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _words(sentence: str) -> List[str]:
        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            sentence.lower(),
        )