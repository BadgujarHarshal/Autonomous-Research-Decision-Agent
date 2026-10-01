# app/tools/knowledge_search.py

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from app.tools.base import BaseTool


class KnowledgeSearchTool(BaseTool):
    """
    Knowledge retrieval adapter.

    A production retrieval function can be injected through `search_fn`.
    If no retrieval backend is configured, the tool can perform a simple
    keyword fallback over supplied documents.
    """

    name = "search_knowledge"
    description = (
        "Searches the available knowledge base for information relevant "
        "to a user question."
    )

    def __init__(
        self,
        search_fn: Optional[Callable[[str, int], Any]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.search_fn = search_fn
        self.documents = documents or []

    @property
    def argument_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question or search query.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def execute(self, arguments: Dict[str, Any]) -> Any:
        self.validate_arguments(arguments)

        query = arguments.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        top_k = arguments.get("top_k", 5)

        if not isinstance(top_k, int):
            raise ValueError("top_k must be an integer.")

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        query = query.strip()

        if self.search_fn is not None:
            return self.search_fn(query, top_k)

        return self._keyword_search(query, top_k)

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        query_terms = {
            token.lower()
            for token in query.split()
            if token.strip()
        }

        results = []

        for document in self.documents:
            content = str(
                document.get("content")
                or document.get("text")
                or ""
            )

            title = str(document.get("title") or "")

            document_terms = {
                token.lower()
                for token in f"{title} {content}".split()
                if token.strip()
            }

            overlap = len(query_terms.intersection(document_terms))

            if overlap > 0:
                results.append(
                    {
                        "title": title,
                        "content": content,
                        "score": overlap,
                        "metadata": document.get("metadata", {}),
                    }
                )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]