# Multi-Collection Search Design for mcp-server-qdrant

## Current State Analysis

### qdrant-search-ui Implementation (Working Reference)

**Key Features:**
- ✅ Searches multiple collections in parallel via `Promise.all()`
- ✅ Merges results from all collections
- ✅ Optional collection filtering (user can select specific collections)
- ✅ Priority boosting strategy (boost scores based on collection order)
- ✅ Smart result distribution (returns 3x results for multi-collection searches)
- ✅ Auto-discover collections from Qdrant API

**Architecture:**
```javascript
async function searchQdrant(vector, limit, collections, metadataFilters) {
    // 1. Determine collections to search
    let collectionsToSearch;
    if (collections && collections.length > 0) {
        collectionsToSearch = collections;  // User specified
    } else if (CONFIG.qdrant.autoDiscover) {
        collectionsToSearch = await fetchQdrantCollections();  // Auto-discover
    } else {
        collectionsToSearch = CONFIG.qdrant.collections;  // Configured list
    }

    // 2. Search all collections in parallel
    const searchPromises = collectionsToSearch.map(async (collection) => {
        const url = `${baseUrl}/collections/${collection}/points/search`;
        return await fetch(url, { /* search params */ });
    });

    const results = await Promise.all(searchPromises);

    // 3. Merge and sort results
    let allResults = results.flatMap(r => r.result || []);

    // 4. Apply priority boosting (optional)
    if (priorityStrategy !== 'natural') {
        allResults = applyPriorityBoosting(allResults, priorityStrategy);
    }

    // 5. Sort by score and limit
    allResults.sort((a, b) => b.score - a.score);
    return allResults.slice(0, totalLimit);
}
```

**Priority Strategies:**
- `natural` - No boosting, pure vector similarity scores
- `soft` - Small boost (1.1x) to preferred collections
- `medium` - Moderate boost (1.25x)
- `hard` - Strong boost (1.5x)

### mcp-server-qdrant Current Architecture

**Limitations:**
- ❌ Only searches ONE collection per query
- ❌ No parallel collection search
- ❌ No result merging from multiple sources
- ❌ No priority boosting
- ❌ No collection auto-discovery

**Current Flow:**
```python
async def search(query, collection_name, limit, query_filter):
    # Only searches single collection
    collection_name = collection_name or self._default_collection_name

    query_vector = await self._embedding_provider.embed_query(query)

    search_results = await self._client.query_points(
        collection_name=collection_name,  # Single collection
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
    )

    return [Entry(...) for result in search_results.points]
```

## Design Goals

1. **Backward Compatibility** - Existing single-collection behavior must work unchanged
2. **Flexible Collection Selection** - Support:
   - Single collection (current behavior)
   - Multiple specific collections (["homelab-docs-bge", "docker-stacks-bge"])
   - All collections (null/None = search everything)
3. **Performance** - Parallel searches with asyncio.gather()
4. **Result Quality** - Merge, sort, and optionally boost scores
5. **Token Efficiency** - Smart result distribution and limiting

## Proposed API Changes

### Option 1: Collections Parameter (Simple, Recommended)

**qdrant-find tool signature:**
```python
async def find(
    ctx: Context,
    query: str,
    collection_name: str | None = None,  # DEPRECATED (backward compat)
    collections: list[str] | None = None,  # NEW - replaces collection_name
    mode: str = "minimal",
    limit: int | None = None,
    query_filter: ArbitraryFilter | None = None,
    rerank: bool = False,
) -> list[str] | None:
```

**Behavior:**
- If `collections` provided → search those collections
- If `collection_name` provided (legacy) → search single collection
- If neither provided → search default collection
- If `collections=["*"]` → search ALL collections (auto-discover)

**Examples:**
```python
# Single collection (backward compatible)
qdrant-find(query="server IP", collection_name="homelab-docs-bge")

# Multiple specific collections
qdrant-find(query="server IP", collections=["homelab-docs-bge", "docker-stacks-bge"])

# All collections
qdrant-find(query="server IP", collections=["*"])

# With reranking (retrieve more from each collection, rerank combined results)
qdrant-find(query="ZFS architecture", collections=["homelab-docs-bge", "chad-brain-bge"],
            limit=30, rerank=true)
```

### Option 2: Search Mode Parameter (More Complex)

**qdrant-find tool signature:**
```python
async def find(
    ctx: Context,
    query: str,
    collection_name: str | None = None,
    mode: str = "minimal",
    limit: int | None = None,
    query_filter: ArbitraryFilter | None = None,
    rerank: bool = False,
    search_mode: str = "single",  # NEW: "single", "multi", "all"
) -> list[str] | None:
```

**Not recommended** - less intuitive than explicit collection list.

## Implementation Plan

### Phase 1: Core Multi-Collection Search

**Files to modify:**

1. **`src/mcp_server_qdrant/qdrant.py`** - Add multi-collection search method
   ```python
   async def search_multi(
       self,
       query: str,
       *,
       collections: list[str] | None = None,
       limit: int = 10,
       query_filter: models.Filter | None = None,
   ) -> list[Entry]:
       """
       Search multiple collections in parallel and merge results.

       :param collections: List of collection names, or None for default, or ["*"] for all
       """
       # Determine which collections to search
       if collections is None:
           collections_to_search = [self._default_collection_name]
       elif collections == ["*"]:
           collections_to_search = await self.get_collection_names()
       else:
           collections_to_search = collections

       # Embed query once (reuse for all collections)
       query_vector = await self._embedding_provider.embed_query(query)
       vector_name = self._embedding_provider.get_vector_name()

       # Search all collections in parallel
       search_tasks = [
           self._search_single_collection(
               collection, query_vector, vector_name, limit, query_filter
           )
           for collection in collections_to_search
       ]

       results = await asyncio.gather(*search_tasks, return_exceptions=True)

       # Merge results, filter errors, add collection metadata
       all_entries = []
       for collection, result in zip(collections_to_search, results):
           if isinstance(result, Exception):
               logger.warning(f"Collection {collection} search failed: {result}")
               continue

           # Add collection name to metadata
           for entry in result:
               if entry.metadata is None:
                   entry.metadata = {}
               entry.metadata["collection"] = collection
               all_entries.append(entry)

       # Sort by score (Entry needs score attribute)
       all_entries.sort(key=lambda e: e.score, reverse=True)

       # Return top N (smart limiting for multi-collection)
       total_limit = limit * min(len(collections_to_search), 3) if len(collections_to_search) > 1 else limit
       return all_entries[:total_limit]

   async def _search_single_collection(
       self, collection, query_vector, vector_name, limit, query_filter
   ) -> list[Entry]:
       """Helper to search a single collection."""
       if not await self._client.collection_exists(collection):
           return []

       search_results = await self._client.query_points(
           collection_name=collection,
           query=query_vector,
           using=vector_name,
           limit=limit,
           query_filter=query_filter,
           with_payload=True,
       )

       return [
           Entry(
               content=result.payload.get("page_content") or result.payload.get("document", ""),
               metadata=result.payload.get("metadata"),
               id=str(result.id),
               score=result.score,  # IMPORTANT: Capture score for sorting
           )
           for result in search_results.points
       ]
   ```

2. **`src/mcp_server_qdrant/qdrant.py`** - Update Entry model to include score
   ```python
   class Entry(BaseModel):
       content: str
       metadata: Metadata | None = None
       id: str | None = None
       score: float | None = None  # NEW - for multi-collection sorting
   ```

3. **`src/mcp_server_qdrant/mcp_server.py`** - Update find() tool
   ```python
   async def find(
       ctx: Context,
       query: str,
       collection_name: str | None = None,
       collections: list[str] | None = None,  # NEW
       mode: str = "minimal",
       limit: int | None = None,
       query_filter: ArbitraryFilter | None = None,
       rerank: bool = False,
   ) -> list[str] | None:
       # Handle backward compatibility
       if collections is not None:
           # New multi-collection mode
           await ctx.debug(f"Multi-collection search: {collections}")
           search_limit = limit if limit is not None else self.qdrant_settings.search_limit

           entries = await self.qdrant_connector.search_multi(
               query,
               collections=collections,
               limit=search_limit,
               query_filter=query_filter,
           )
       else:
           # Legacy single-collection mode
           collection = collection_name or self.qdrant_settings.collection_name
           await ctx.debug(f"Single-collection search: {collection}")
           search_limit = limit if limit is not None else self.qdrant_settings.search_limit

           entries = await self.qdrant_connector.search(
               query,
               collection_name=collection,
               limit=search_limit,
               query_filter=query_filter,
           )

       # Reranking logic (works with both single and multi-collection)
       if rerank and self.reranker_client:
           # ... existing reranking code ...

       # Format results
       # ... existing formatting code ...
   ```

4. **`src/mcp_server_qdrant/settings.py`** - Optional: Add multi-collection settings
   ```python
   class QdrantSettings(BaseSettings):
       # ... existing settings ...

       multi_collection_limit_multiplier: int = Field(
           default=3,
           validation_alias="QDRANT_MULTI_COLLECTION_LIMIT_MULTIPLIER",
           description="Multiplier for result limit when searching multiple collections"
       )
   ```

### Phase 2: Priority Boosting (Optional Enhancement)

**Add after Phase 1 is working:**

1. **`src/mcp_server_qdrant/priority.py`** - New module for priority strategies
   ```python
   from enum import Enum
   from mcp_server_qdrant.qdrant import Entry

   class PriorityStrategy(str, Enum):
       NATURAL = "natural"
       SOFT = "soft"
       MEDIUM = "medium"
       HARD = "hard"

   BOOST_FACTORS = {
       PriorityStrategy.NATURAL: 1.0,
       PriorityStrategy.SOFT: 1.1,
       PriorityStrategy.MEDIUM: 1.25,
       PriorityStrategy.HARD: 1.5,
   }

   def apply_priority_boosting(
       entries: list[Entry],
       collection_order: list[str],
       strategy: PriorityStrategy = PriorityStrategy.NATURAL
   ) -> list[Entry]:
       """Apply priority boosting based on collection order."""
       if strategy == PriorityStrategy.NATURAL:
           return entries

       boost_factor = BOOST_FACTORS[strategy]

       for entry in entries:
           collection = entry.metadata.get("collection")
           if collection in collection_order:
               priority_index = collection_order.index(collection)
               # Higher priority (lower index) = higher boost
               entry.score *= boost_factor ** (len(collection_order) - priority_index)

       return entries
   ```

2. **Add to settings:**
   ```python
   priority_strategy: PriorityStrategy = Field(
       default=PriorityStrategy.NATURAL,
       validation_alias="QDRANT_PRIORITY_STRATEGY"
   )
   collection_priority_order: list[str] | None = Field(
       default=None,
       validation_alias="QDRANT_COLLECTION_PRIORITY_ORDER"
   )
   ```

### Phase 3: Reranker Integration

**Reranker works naturally with multi-collection:**
- Retrieve 30 results from each collection (e.g., 2 collections = 60 candidates)
- Merge all candidates
- Rerank combined pool
- Return top 8

No special changes needed - existing reranker logic handles this automatically!

## Testing Strategy

### Unit Tests
```python
async def test_multi_collection_search():
    # Test searching multiple collections
    results = await connector.search_multi(
        query="test",
        collections=["collection1", "collection2"],
        limit=5
    )
    assert len(results) <= 15  # 5 * 3 (multiplier)

async def test_all_collections_search():
    # Test wildcard search
    results = await connector.search_multi(
        query="test",
        collections=["*"],
        limit=5
    )
    assert all(r.metadata.get("collection") for r in results)

async def test_backward_compatibility():
    # Test that old single-collection searches still work
    results = await connector.search(
        query="test",
        collection_name="homelab-docs-bge",
        limit=5
    )
    assert len(results) <= 5
```

### Integration Tests
1. Search single collection (verify backward compat)
2. Search 2 specific collections (verify parallel + merge)
3. Search all collections (verify auto-discovery)
4. Search with reranking (verify combined results reranked)
5. Search with metadata filters (verify filters applied to all collections)

## Benefits

✅ **Powerful** - Search across all knowledge bases in one query
✅ **Flexible** - Single collection, multi-collection, or all collections
✅ **Fast** - Parallel searches with asyncio
✅ **Backward Compatible** - Existing code works unchanged
✅ **Token Efficient** - Smart result distribution prevents overwhelming output
✅ **Reranker Ready** - Works seamlessly with existing reranker feature

## Example Use Cases

### Chad's Brain Use Case
```python
# Search across all brain collections for troubleshooting
qdrant-find(
    query="Frigate camera issues",
    collections=["homelab-docs-bge", "docker-stacks-bge", "chad-brain-bge"],
    mode="full",
    limit=10,
    rerank=true
)
```

### Comprehensive Architecture Query
```python
# Search everything for architecture understanding
qdrant-find(
    query="ZFS replication architecture",
    collections=["*"],  # Search ALL collections
    mode="full",
    limit=30,
    rerank=true
)
```

### Targeted Multi-Source Search
```python
# Search docs + stacks for service configuration
qdrant-find(
    query="Traefik middleware configuration",
    collections=["homelab-docs-bge", "docker-stacks-bge"],
    mode="full",
    limit=15
)
```

## Implementation Priority

**Recommended approach:**
1. ✅ Implement Phase 1 (core multi-collection search) - ~2-3 hours
2. ✅ Test thoroughly with real queries - ~1 hour
3. ✅ Update README and documentation - ~30 mins
4. 🔜 Phase 2 (priority boosting) - Optional, implement if needed
5. ✅ Phase 3 works automatically - no additional work needed!

## Open Questions

1. **Should we deprecate `collection_name` parameter?**
   - Recommend: Keep for backward compatibility, but mark as deprecated in docs
   - Eventually migrate to `collections` for all use cases

2. **Should wildcard search be explicit or implicit?**
   - Recommend: Explicit `collections=["*"]` (clearer intent)
   - Alternative: `collections=None` means "all" (but breaks backward compat)

3. **How to handle collection-specific errors?**
   - Recommend: Log warning, continue with successful collections
   - Return partial results rather than failing entire query

4. **Should we add collection weights/priorities now?**
   - Recommend: Phase 2 (after core functionality proven)
   - Most use cases work fine with natural score sorting
