import logging
import httpx
from typing import Any

from mcp_server_qdrant.qdrant import Entry
from mcp_server_qdrant.settings import RerankerSettings

logger = logging.getLogger(__name__)


class RerankerClient:
    """Client for reranking search results via BGE reranker API."""

    def __init__(self, settings: RerankerSettings):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=settings.timeout)

    async def rerank(self, query: str, entries: list[Entry], top_k: int | None = None) -> list[Entry]:
        """
        Rerank entries by relevance to query.

        :param query: User query
        :param entries: List of entries from Qdrant search
        :param top_k: Number of top results to return (overrides default)
        :return: Reranked list of entries (limited to top_k)
        """
        if not entries:
            return entries

        top_k = top_k or self.settings.top_k

        # Extract text content from entries
        documents = [entry.content for entry in entries]

        try:
            logger.info(f"Reranking {len(documents)} candidates → top {top_k}")

            headers = {"Content-Type": "application/json"}
            if self.settings.api_key:
                headers["Authorization"] = f"Bearer {self.settings.api_key}"

            response = await self.client.post(
                self.settings.url,
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": top_k
                },
                headers=headers
            )

            response.raise_for_status()
            data = response.json()

            # Validate response structure
            if not data or "results" not in data or not isinstance(data["results"], list):
                raise ValueError(f"Invalid reranker response structure: {data}")

            # Map reranked results back to original entries
            # API returns: {results: [{index: N, relevance_score: X}, ...]}
            reranked_entries = []
            for item in data["results"][:top_k]:  # Force limit to top_k
                index = item.get("index")
                score = item.get("relevance_score") or item.get("score")

                if index is None or index < 0 or index >= len(entries):
                    logger.warning(f"Invalid index from reranker: {index}")
                    continue

                # Add reranked entry with score metadata
                entry = entries[index]
                if entry.metadata is None:
                    entry.metadata = {}
                entry.metadata["rerank_score"] = score
                entry.metadata["reranked"] = True

                reranked_entries.append(entry)

            logger.info(f"Reranked {len(entries)} → {len(reranked_entries)} results")
            return reranked_entries

        except httpx.HTTPError as e:
            logger.error(f"Reranker HTTP error: {e}")
            raise RuntimeError(f"Reranker API failed: {e}")
        except Exception as e:
            logger.error(f"Reranker error: {e}")
            raise RuntimeError(f"Reranker failed: {e}")

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
