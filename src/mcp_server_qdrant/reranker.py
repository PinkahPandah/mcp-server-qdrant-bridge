import logging
import httpx
from datetime import datetime, timezone
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
            for item in data["results"]:  # Process all results, will re-sort and trim
                index = item.get("index")
                score = item.get("relevance_score") or item.get("score")

                if index is None or index < 0 or index >= len(entries):
                    logger.warning(f"Invalid index from reranker: {index}")
                    continue

                # Add reranked entry with score metadata
                entry = entries[index]
                if entry.metadata is None:
                    entry.metadata = {}

                # Apply TTL decay for expired working memories
                adjusted_score = self._apply_ttl_decay(entry, score)

                entry.metadata["rerank_score"] = adjusted_score
                entry.metadata["reranked"] = True

                reranked_entries.append(entry)

            # Re-sort by adjusted score (TTL decay may have changed order)
            reranked_entries.sort(key=lambda e: e.metadata.get("rerank_score", 0), reverse=True)

            # Trim to top_k after re-sorting
            reranked_entries = reranked_entries[:top_k]

            logger.info(f"Reranked {len(entries)} → {len(reranked_entries)} results (with TTL decay)")
            return reranked_entries

        except httpx.HTTPError as e:
            logger.error(f"Reranker HTTP error: {e}")
            raise RuntimeError(f"Reranker API failed: {e}")
        except Exception as e:
            logger.error(f"Reranker error: {e}")
            raise RuntimeError(f"Reranker failed: {e}")

    def _apply_ttl_decay(self, entry: Entry, score: float) -> float:
        """
        Apply decay multiplier to scores for expired or near-expired working memory.

        Long-term memories and unclassified entries pass through unchanged.
        Working memories get decayed based on TTL expiration status.
        """
        metadata = entry.metadata or {}
        memory_type = metadata.get("memory_type")
        ttl_days = metadata.get("ttl_days")
        timestamp = metadata.get("timestamp")

        # Only decay working memory with valid TTL info
        if memory_type != "working" or not ttl_days or not timestamp:
            return score

        try:
            # Parse timestamp (handle ISO format with Z or +00:00)
            ts = timestamp.replace("Z", "+00:00") if isinstance(timestamp, str) else str(timestamp)
            created = datetime.fromisoformat(ts)
            age_days = (datetime.now(timezone.utc) - created).days

            if age_days > ttl_days:
                # Expired - heavy decay
                decay = 0.3
                logger.debug(f"TTL expired ({age_days}d > {ttl_days}d): score {score:.3f} → {score * decay:.3f}")
                return score * decay
            elif age_days > ttl_days * 0.8:
                # Near expiry (>80% of TTL) - mild decay
                decay = 0.7
                logger.debug(f"TTL near expiry ({age_days}d / {ttl_days}d): score {score:.3f} → {score * decay:.3f}")
                return score * decay

            return score

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse TTL metadata: {e}")
            return score

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
