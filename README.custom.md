# MCP Server Qdrant - Token-Optimized Fork

Fork of [mcp-server-qdrant](https://github.com/modelcontextprotocol/servers/tree/main/src/qdrant) with 90-97% token reduction for documentation queries.

## Changes

### 1. Minimal Mode (Default)
Returns metadata only (IPs, hostnames, URLs, file names) without full chunk content.

**Token savings**: 60k → 2k tokens (97% reduction)

### 2. Reduced Default Limit
Changed default from 10 → 5 results.

### 3. Optional Limit Parameter
Override default when deeper search needed.

## Usage

```python
# Default: minimal mode, 5 results (~2k tokens)
qdrant-find(query="server ip address")

# Full content mode (~3-6k tokens)
qdrant-find(query="configuration procedure", mode="full")

# More results when needed
qdrant-find(query="architecture", mode="full", limit=10)
```

## Token Comparison

| Mode | Results | Tokens | Use Case |
|------|---------|--------|----------|
| minimal (default) | 5 | ~2k | Factual lookups |
| full | 5 | ~3-6k | Procedures, configs |
| full | 10 | ~6-12k | Deep research |

## Configuration

```python
# settings.py
search_limit: int = Field(default=5)  # was 10

# mcp_server.py
mode: str = "minimal"  # was "full"
```

## Backward Compatible

- Explicit `mode='full'` behaves like original
- `QDRANT_SEARCH_LIMIT` env var can restore defaults
- All existing functionality preserved

## License

MIT (same as upstream)
