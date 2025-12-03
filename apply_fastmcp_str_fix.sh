#!/bin/bash
# Apply fix for FastMCP JSON parsing to handle Annotated[str, ...] parameters correctly
# This script patches the installed FastMCP package to prevent str parameters
# from being JSON-parsed when FASTMCP_TOOL_ATTEMPT_PARSE_JSON_ARGS=true
#
# The issue: FastMCP checks `annotation in (int, float, bool, str)` but pydantic
# uses Annotated[str, Field(...)] which doesn't match `str` directly.
# Fix: Unwrap Annotated types using get_origin/get_args before checking.

set -e

VENV_PATH="/home/cory/mcp-servers/qdrant-mcp/.venv"
FASTMCP_TOOL_FILE="$VENV_PATH/lib/python3.10/site-packages/fastmcp/tools/tool.py"

if [ ! -f "$FASTMCP_TOOL_FILE" ]; then
    echo "ERROR: FastMCP tool.py not found at: $FASTMCP_TOOL_FILE"
    echo "Make sure the virtual environment is set up correctly."
    exit 1
fi

echo "Applying FastMCP Annotated[str] fix..."
echo "File: $FASTMCP_TOOL_FILE"

# Check if already patched (look for get_origin which is part of our fix)
if grep -q "get_origin(annotation) is Annotated" "$FASTMCP_TOOL_FILE"; then
    echo "✓ Fix already applied"
    exit 0
fi

# Create backup
BACKUP_FILE="${FASTMCP_TOOL_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$FASTMCP_TOOL_FILE" "$BACKUP_FILE"
echo "Created backup: $BACKUP_FILE"

# Use Python for reliable patching
python3 << 'EOF'
import sys

file_path = "/home/cory/mcp-servers/qdrant-mcp/.venv/lib/python3.10/site-packages/fastmcp/tools/tool.py"

with open(file_path, 'r') as f:
    content = f.read()

# Step 1: Add imports if not present
old_import = "from typing import TYPE_CHECKING, Any"
new_import = "from typing import TYPE_CHECKING, Any, Annotated, get_origin, get_args"

if "get_origin, get_args" not in content:
    if old_import in content:
        content = content.replace(old_import, new_import)
        print("✓ Added get_origin, get_args imports")
    else:
        print("WARNING: Could not find import line to patch")

# Step 2: Fix the annotation check to unwrap Annotated types
old_check = """                # skip if the type is a simple type (int, float, bool)
                if signature.parameters[param_name].annotation in (
                    int,
                    float,
                    bool,
                ):
                    continue"""

new_check = """                # skip if the type is a simple type (int, float, bool, str)
                # Also handle Annotated[str, ...] by unwrapping
                annotation = signature.parameters[param_name].annotation
                base_type = annotation
                if get_origin(annotation) is Annotated:
                    base_type = get_args(annotation)[0]
                if base_type in (int, float, bool, str):
                    continue"""

# Also check for partially-applied old fix (just added str to tuple)
old_check_with_str = """                # skip if the type is a simple type (int, float, bool, str)
                if signature.parameters[param_name].annotation in (
                    int,
                    float,
                    bool,
                    str,
                ):
                    continue"""

if old_check in content:
    content = content.replace(old_check, new_check)
    print("✓ Applied Annotated unwrap fix (from original)")
elif old_check_with_str in content:
    content = content.replace(old_check_with_str, new_check)
    print("✓ Applied Annotated unwrap fix (from partial fix)")
elif new_check in content:
    print("✓ Fix already applied")
else:
    print("ERROR: Could not find the code to patch")
    print("The FastMCP version may have changed significantly.")
    sys.exit(1)

with open(file_path, 'w') as f:
    f.write(content)
print("✓ File saved")
sys.exit(0)
EOF

echo ""
echo "Fix applied successfully!"
echo ""
echo "IMPORTANT: Restart Claude Code or run /mcp to reload the MCP server"
echo ""
echo "To verify:"
echo "  Store JSON-like content in qdrant-store information parameter"
echo "  It should be stored as a string, not parsed as a dict"
