"""What an adapter's debug line says about a request: its shape, never its content.

Shared by all three adapters, so their request lines read alike and a key added
to any of them falls under the same rule.
"""

import json
from collections.abc import Mapping
from typing import Any

#: Request keys whose values are settings, logged as they are. Every other key
#: is conversation or declarations and is logged by size alone — so a key added
#: later stays private until someone decides it is a setting.
SETTINGS = frozenset(
    {
        "max_tokens",
        "max_output_tokens",
        "temperature",
        "thinking",
        "output_config",
        "reasoning",
        "reasoning_effort",
        "include",
        "store",
        "stream_options",
    }
)

#: Named elsewhere in the line, or the same on every request.
_OMITTED = frozenset({"model", "stream"})


def request_shape(request: Mapping[str, Any]) -> str:
    """`messages=12 system=1830 tools=9 max_tokens=8192 thinking={"type":"adaptive"}`.

    A size is items for a list and characters for a string.
    """
    parts = []
    for key, value in request.items():
        if key in _OMITTED:
            continue
        if key in SETTINGS:
            parts.append(f"{key}={json.dumps(value, separators=(',', ':'))}")
        elif isinstance(value, str | list | tuple | dict):
            parts.append(f"{key}={len(value)}")
        else:
            parts.append(f"{key}=<{type(value).__name__}>")
    return " ".join(parts)
