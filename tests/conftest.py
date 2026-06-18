import os
import sys

# src-layout: make `ai_engine` importable without an editable install.
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Tests run in a dev context: with no INGEST_API_KEY configured the write
# endpoints are intentionally open. In prod the guard fails closed instead.
os.environ.setdefault("AI_ENGINE_DEV", "1")
