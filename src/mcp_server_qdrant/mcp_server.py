import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.reranker import RerankerClient
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    RerankerSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        reranker_settings: Optional[RerankerSettings] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        # Initialize reranker client if enabled
        self.reranker_settings = reranker_settings or RerankerSettings()
        self.reranker_client = (
            RerankerClient(self.reranker_settings)
            if self.reranker_settings.enabled
            else None
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Return full entry with content and metadata in readable format.
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_id = entry.id if entry.id else "unknown"
        
        output = []
        output.append("---")
        output.append(f"ID: {entry_id}")
        
        if entry.score is not None:
            output.append(f"Score: {entry.score:.4f}")
        
        if entry.metadata:
            key_fields = ["file_name", "source", "doc_type", "chunk_id"]
            for field in key_fields:
                if field in entry.metadata and entry.metadata[field]:
                    output.append(f"{field}: {entry.metadata[field]}")
        
        output.append("")
        output.append(entry.content)
        output.append("---")
        
        return "\n".join(output)

    def format_entry_minimal(self, entry: Entry) -> str:
        """
        Return only key metadata without chunk content for token-efficient responses.
        """
        entry_id = entry.id if entry.id else "unknown"

        if not entry.metadata:
            return f"📄 ID: {entry_id} | ⚠️ Keine Metadaten verfügbar"

        # Extract commonly queried factual fields
        factual_fields = {
            "ip_addresses": entry.metadata.get("ip_addresses"),
            "hostnames": entry.metadata.get("hostnames"),
            "ports": entry.metadata.get("ports"),
            "domains": entry.metadata.get("domains"),
            "urls": entry.metadata.get("urls"),
            "service_name": entry.metadata.get("service_name"),
            "stack_name": entry.metadata.get("stack_name"),
            "file_name": entry.metadata.get("file_name"),
            "source": entry.metadata.get("source"),
            "doc_type": entry.metadata.get("doc_type"),
        }

        # Filter out None values
        factual_fields = {k: v for k, v in factual_fields.items() if v}

        if not factual_fields:
            return f"📄 ID: {entry_id} | ⚠️ Keine relevanten Metadaten"
        
        metadata_str = " | ".join([f"{k}: {v}" for k, v in factual_fields.items()])
        return f"📄 ID: {entry_id} | {metadata_str}"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            point_id = await self.qdrant_connector.store(
                entry, collection_name=collection_name
            )
            if collection_name:
                return f"Remembered: {information} in collection {collection_name} (point_id: {point_id})"
            return f"Remembered: {information} (point_id: {point_id})"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str | None,
                Field(
                    description="(DEPRECATED: Use collections parameter) The collection to search in"
                ),
            ] = None,
            collections: Annotated[
                list[str] | None,
                Field(
                    description="List of collections to search. Use ['*'] to search all collections. If not provided, uses collection_name or default collection."
                ),
            ] = None,
            mode: Annotated[
                str,
                Field(
                    description="Response mode: 'minimal' returns only metadata for token efficiency, 'full' (default) returns complete chunks when you need more context"
                ),
            ] = "full",
            limit: Annotated[
                int | None,
                Field(
                    description="Maximum number of results to return. If not specified, uses default (5). Increase for deeper searches."
                ),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            rerank: Annotated[
                bool,
                Field(
                    description="Enable reranking for improved relevance and token efficiency. Retrieves more candidates and reranks to return best results. Set to false to disable. Default: True"
                ),
            ] = True,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: (DEPRECATED, Optional) The name of the collection to search in. Use collections parameter instead.
            :param collections: (Optional) List of collections to search. Use ['*'] for all collections. If None, uses collection_name or default.
            :param mode: Response mode - 'minimal' for token-efficient metadata, 'full' (default) for complete chunks when needed.
            :param limit: Maximum number of results to return per collection. If not specified, uses default (5).
            :param query_filter: The filter to apply to the query.
            :param rerank: Enable reranking (default: True). Set to False to disable reranking and return raw vector search results.
            :return: A list of entries found or None.

            Note: Either collections, collection_name, or a default collection must be configured.
            """

            # Validate that we have a way to determine which collection(s) to search
            if (
                collections is None
                and collection_name is None
                and self.qdrant_settings.collection_name is None
            ):
                raise ValueError(
                    "Must provide either 'collections' parameter, 'collection_name' parameter, or configure a default collection"
                )

            # Use provided limit or fall back to default
            search_limit = (
                limit if limit is not None else self.qdrant_settings.search_limit
            )

            # Log query_filter, mode, and limit
            await ctx.debug(f"Query filter: {query_filter}")
            await ctx.debug(f"Response mode: {mode}")
            await ctx.debug(f"Search limit: {search_limit}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            # Handle backward compatibility: collections parameter vs collection_name
            if collections is not None:
                # New multi-collection mode
                await ctx.debug(f"Multi-collection search: {collections}")
                entries = await self.qdrant_connector.search_multi(
                    query,
                    collections=collections,
                    limit=search_limit,
                    query_filter=query_filter,
                    limit_multiplier=self.qdrant_settings.multi_collection_limit_multiplier,
                )
            else:
                # Legacy single-collection mode
                await ctx.debug(f"Single-collection search: {collection_name}")
                entries = await self.qdrant_connector.search(
                    query,
                    collection_name=collection_name,
                    limit=search_limit,
                    query_filter=query_filter,
                )

            # Reranking logic
            if rerank and self.reranker_client:
                try:
                    await ctx.debug(f"Reranking {len(entries)} candidates")
                    # Use configured top_k or fall back to search_limit
                    top_k = min(search_limit, self.reranker_settings.top_k)
                    entries = await self.reranker_client.rerank(
                        query, entries, top_k=top_k
                    )
                    await ctx.debug(f"Reranked to {len(entries)} results")
                except Exception as e:
                    await ctx.debug(
                        f"Reranker failed: {e}, returning unreranked results"
                    )
                    # On reranker failure, return original results (fail gracefully)
                    logger.warning(f"Reranker failed, using unreranked results: {e}")

            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                if mode == "minimal":
                    content.append(self.format_entry_minimal(entry))
                else:
                    content.append(self.format_entry(entry))
            return content

        async def delete(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to delete from")
            ],
            query_filter: ArbitraryFilter | None = None,
            point_ids: Annotated[
                list[str] | None,
                Field(
                    description="List of point IDs (UUIDs) to delete. Preferred method for reliable deletion."
                ),
            ] = None,
        ) -> str:
            """
            Delete points from Qdrant by IDs or metadata filters.

            ID-based deletion is preferred as it's more reliable. Provide either point_ids OR query_filter, not both.

            :param ctx: The context for the request.
            :param collection_name: The name of the collection to delete from.
            :param query_filter: Filter conditions to identify points to delete (optional).
            :param point_ids: List of point IDs (UUIDs) to delete (optional, recommended).
            :return: Confirmation message with operation details.
            """
            if point_ids is not None:
                await ctx.debug(
                    f"Deleting {len(point_ids)} points by ID: {point_ids[:3]}{'...' if len(point_ids) > 3 else ''}"
                )
                result = await self.qdrant_connector.delete(
                    point_ids=point_ids, collection_name=collection_name
                )
            elif query_filter is not None:
                await ctx.debug(f"Deleting points with filter: {query_filter}")
                filter_obj = models.Filter(**query_filter)
                result = await self.qdrant_connector.delete(
                    filter_obj, collection_name=collection_name
                )
            else:
                raise ValueError("Must provide either point_ids or query_filter")

            return f"Delete operation completed for {collection_name}: status={result['status']}, operation_id={result.get('operation_id', 'N/A')}"

        async def retrieve(
            ctx: Context,
            point_id: Annotated[
                str, Field(description="The exact point ID (UUID) to retrieve")
            ],
            collection_name: Annotated[
                str, Field(description="The collection to retrieve from")
            ],
        ) -> str:
            """
            Retrieve a point by exact ID from Qdrant.

            This is faster than semantic search for exact ID lookups (no embedding, no vector computation).

            :param ctx: The context for the request.
            :param point_id: The exact point ID (UUID) to retrieve.
            :param collection_name: The name of the collection to retrieve from.
            :return: The entry if found, error message otherwise.
            """
            await ctx.debug(f"Retrieving point {point_id} from {collection_name}")

            entry = await self.qdrant_connector.retrieve(
                point_id=point_id, collection_name=collection_name
            )

            if not entry:
                return f"Point {point_id} not found in collection {collection_name}"

            # Format same as find tool for consistency
            metadata_json = (
                json.dumps(entry.metadata, indent=2) if entry.metadata else "{}"
            )
            return f"<entry><id>{entry.id}</id><content>{entry.content}</content><metadata>{metadata_json}</metadata></entry>"

        find_foo = find
        store_foo = store
        delete_foo = delete
        retrieve_foo = retrieve

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
            delete_foo = wrap_filters(delete_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})
            delete_foo = make_partial_function(delete_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            delete_foo = make_partial_function(
                delete_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            retrieve_foo = make_partial_function(
                retrieve_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            self.tool(
                delete_foo,
                name="qdrant-delete",
                description=self.tool_settings.tool_delete_description,
            )
            self.tool(
                retrieve_foo,
                name="qdrant-retrieve",
                description="Retrieve a point by exact ID from Qdrant. Faster than semantic search for known point IDs (no embedding, no vector computation). Use when you have the exact point ID from a previous search or log.",
            )
