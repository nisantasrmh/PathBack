"""
PathBack Enterprise Backend Server Launcher.
"""

from __future__ import annotations

import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PathBack Backend API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"🚀 Starting PathBack FastAPI Backend on http://{args.host}:{args.port}")
    print(f"📚 OpenAPI Documentation available at http://localhost:{args.port}/docs")

    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=args.reload)
