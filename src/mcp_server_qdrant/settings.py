from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in Qdrant. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user"
)
DEFAULT_TOOL_DELETE_DESCRIPTION = (
    "Delete points from Qdrant by point IDs or metadata filters. "
    "ID-based deletion (via point_ids parameter) is preferred as it's more reliable and accurate. "
    "Use filter-based deletion (via query_filter parameter) when you don't have specific IDs. "
    "Provide either point_ids OR query_filter, not both."
)

METADATA_PATH = "metadata"


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    tool_delete_description: str = Field(
        default=DEFAULT_TOOL_DELETE_DESCRIPTION,
        validation_alias="TOOL_DELETE_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    # OpenAI-compatible provider settings
    base_url: str | None = Field(
        default=None,
        validation_alias="EMBEDDING_BASE_URL",
    )
    api_key: str | None = Field(
        default=None,
        validation_alias="EMBEDDING_API_KEY",
    )
    vector_size: int | None = Field(
        default=None,
        validation_alias="EMBEDDING_VECTOR_SIZE",
    )


class RerankerSettings(BaseSettings):
    """Configuration for optional reranking service."""

    enabled: bool = Field(
        default=False,
        validation_alias="RERANKER_ENABLED",
        description="Enable reranker globally (can be overridden per query)",
    )
    url: str = Field(
        default="https://reranker.example.com/rerank",
        validation_alias="RERANKER_URL",
        description="Reranker API endpoint URL",
    )
    api_key: str | None = Field(
        default="be2094c3bcfe4215ee286fab7780b0d812612f021dc476e77ee32d1b6483651a",
        validation_alias="RERANKER_API_KEY",
        description="Bearer token for reranker API",
    )
    candidate_pool_size: int = Field(
        default=30,
        validation_alias="RERANKER_CANDIDATE_POOL_SIZE",
        description="Number of candidates to retrieve before reranking",
    )
    top_k: int = Field(
        default=8,
        validation_alias="RERANKER_TOP_K",
        description="Number of top results to return after reranking",
    )
    timeout: int = Field(
        default=10,
        validation_alias="RERANKER_TIMEOUT",
        description="HTTP timeout in seconds",
    )


class FilterableField(BaseModel):
    name: str = Field(description="The name of the field payload field to filter on")
    description: str = Field(
        description="A description for the field used in the tool description"
    )
    field_type: Literal["keyword", "integer", "float", "boolean"] = Field(
        description="The type of the field"
    )
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"] | None = (
        Field(
            default=None,
            description=(
                "The condition to use for the filter. If not provided, the field will be indexed, but no "
                "filter argument will be exposed to MCP tool."
            ),
        )
    )
    required: bool = Field(
        default=False,
        description="Whether the field is required for the filter.",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: str | None = Field(default=None, validation_alias="QDRANT_URL")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str | None = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    local_path: str | None = Field(default=None, validation_alias="QDRANT_LOCAL_PATH")
    search_limit: int = Field(default=5, validation_alias="QDRANT_SEARCH_LIMIT")
    multi_collection_limit_multiplier: int = Field(
        default=3,
        validation_alias="QDRANT_MULTI_COLLECTION_LIMIT_MULTIPLIER",
        description="Multiplier for result limit when searching multiple collections (e.g., 3x means limit*3 total results)",
    )
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")

    filterable_fields: list[FilterableField] | None = Field(default=None)

    allow_arbitrary_filter: bool = Field(
        default=False, validation_alias="QDRANT_ALLOW_ARBITRARY_FILTER"
    )

    def filterable_fields_dict(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {field.name: field for field in self.filterable_fields}

    def filterable_fields_dict_with_conditions(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {
            field.name: field
            for field in self.filterable_fields
            if field.condition is not None
        }

    @model_validator(mode="after")
    def check_local_path_conflict(self) -> "QdrantSettings":
        if self.local_path:
            if self.location is not None or self.api_key is not None:
                raise ValueError(
                    "If 'local_path' is set, 'location' and 'api_key' must be None."
                )
        return self
