"""
deployment/mcp_server.py — FastMCP server exposing CrisisOpsEnv as an OpenEnv HTTP endpoint.

Spec: "FastMCP server exposing the environment as an OpenEnv HTTP endpoint"

Endpoints:
    POST /reset          — reset(seed?) → initial observation
    POST /step           — step(action) → (obs, reward, done, info)
    GET  /state          — state()      → full serialisable state
    GET  /health         — liveness check
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

# FastMCP import is guarded so the module compiles without the package installed
try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    FastMCP = None  # type: ignore[assignment,misc]

from env.environment import CrisisOpsEnv
from scenarios.level1 import get_random_level1_scenario

# ---------------------------------------------------------------------------
# Global environment singleton (one env per server process)
# ---------------------------------------------------------------------------
_env: Optional[CrisisOpsEnv] = None
_curriculum_level: int = int(os.environ.get("CRISISOPS_LEVEL", "1"))


def _get_env() -> CrisisOpsEnv:
    """Return the global environment, creating it if necessary."""
    global _env
    if _env is None:
        _env = CrisisOpsEnv(
            scenario_fn=get_random_level1_scenario(),
            curriculum_level=_curriculum_level,
        )
    return _env


# ---------------------------------------------------------------------------
# Build MCP server if available
# ---------------------------------------------------------------------------

if _MCP_AVAILABLE:
    mcp = FastMCP("CrisisOps")

    @mcp.tool()
    def reset(seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the CrisisOps environment.

        Returns the initial observation dict.
        Optionally accepts a seed for reproducible episodes.
        """
        env = _get_env()
        obs = env.reset(seed=seed)
        return obs

    @mcp.tool()
    def step(action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one action in the CrisisOps environment.

        Action format: {"action_type": "<type>", "params": {...}}

        Returns: {"observation": {...}, "reward": float, "done": bool, "info": {...}}
        """
        env = _get_env()
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs,
            "reward":      reward,
            "done":        done,
            "info":        info,
        }

    @mcp.tool()
    def get_state() -> Dict[str, Any]:
        """
        Return the full serialisable state of the environment.

        Includes true state (actual completions, candor levels) for debugging.
        """
        env = _get_env()
        return env.state()

    @mcp.tool()
    def health() -> Dict[str, str]:
        """Liveness check — always returns OK."""
        return {"status": "ok", "service": "CrisisOps MCP Server"}

else:
    mcp = None   # type: ignore[assignment]


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the FastMCP server.

    Raises ImportError if FastMCP is not installed.
    """
    if not _MCP_AVAILABLE or mcp is None:
        raise ImportError(
            "FastMCP not installed. Install with: pip install mcp"
        )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    run_server()
