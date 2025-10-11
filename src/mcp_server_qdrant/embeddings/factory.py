from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(settings.model_name)
    elif settings.provider_type == EmbeddingProviderType.OPENAI_COMPATIBLE:
        from mcp_server_qdrant.embeddings.openai_compatible import OpenAICompatibleProvider

        if not settings.base_url or not settings.api_key or not settings.vector_size:
            raise ValueError(
                "OpenAI-compatible provider requires base_url, api_key, and vector_size to be set"
            )

        return OpenAICompatibleProvider(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model_name=settings.model_name,
            vector_size=settings.vector_size,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
