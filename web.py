#!/usr/bin/env python3
"""Launch the AI Subtitle Translator web UI.

Usage:
    python web.py                 # serve on http://127.0.0.1:8000
    python web.py --port 9000     # custom port
    python web.py --reload        # auto-reload on code changes (development)
    HOST=0.0.0.0 python web.py    # bind all interfaces (also via --host)
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the subtitle translator web UI")
    parser.add_argument(
        "--host", default=os.getenv("HOST", "127.0.0.1"),
        help="Interface to bind (default: 127.0.0.1, or $HOST)",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("PORT", "8000")),
        help="Port to listen on (default: 8000, or $PORT)",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Auto-reload on code changes (development only)",
    )
    args = parser.parse_args()

    import uvicorn

    print(f"AI Subtitle Translator web UI → http://{args.host}:{args.port}")
    uvicorn.run(
        "webapp.server:app", host=args.host, port=args.port, reload=args.reload,
    )


if __name__ == "__main__":
    main()
