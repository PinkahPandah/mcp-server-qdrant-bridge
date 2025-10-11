import httpx

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class OpenAICompatibleProvider(EmbeddingProvider):
    """
    OpenAI-compatible API implementation of the embedding provider.
    Supports any embedding server with OpenAI-compatible /embeddings endpoint.

    :param base_url: Base URL of the embedding server (e.g., http://192.168.45.12:7997)
    :param api_key: API key for authentication
    :param model_name: Model name to use (e.g., BAAI/bge-large-en-v1.5)
    :param vector_size: Size of the embedding vectors (e.g., 1024 for BGE-large)
    """

    def __init__(self, base_url: str, api_key: str, model_name: str, vector_size: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.vector_size = vector_size
        self.client = httpx.AsyncClient(timeout=30.0)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        response = await self.client.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": documents,
                "model": self.model_name,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Extract embeddings from OpenAI-compatible response format
        # Response: {"data": [{"embedding": [...]}, ...]}
        return [item["embedding"] for item in data["data"]]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        response = await self.client.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": query,
                "model": self.model_name,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Extract first embedding from response
        return data["data"][0]["embedding"]

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Returns empty string to use the default unnamed vector.
        """
        # Return empty string for unnamed (default) vector
        return ""

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.vector_size

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
