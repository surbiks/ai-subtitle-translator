"""Web UI for the AI Subtitle Translator.

A thin async FastAPI layer over the existing translation pipeline. It reuses the
library functions unchanged (parse → chunk → translate → merge) and streams
per-chunk progress to the browser over Server-Sent Events. Run with::

    python web.py
"""
