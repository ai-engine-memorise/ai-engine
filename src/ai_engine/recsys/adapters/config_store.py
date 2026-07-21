"""Runtime RecConfig override store.

The DURABLE baseline is `RecConfig` defaults + env (the cluster configmap). This
store holds an OPTIONAL runtime override applied on top, so recsys params can be
changed live (via PUT /config + the settings page) WITHOUT a redeploy.

recsys-redis is ephemeral: if it is flushed / restarts without persistence, the
override disappears and config reverts to the env/default baseline. So the
configmap stays the source of truth; Redis is the live override layer.
"""
from __future__ import annotations
import json
from typing import Optional

CONFIG_KEY = "recsys:config"


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge `patch` onto `base` (one level of nesting for fusion/
    engagement weights). Returns a new dict; tolerates partial overrides."""
    out = dict(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class RedisConfigStore:
    def __init__(self, client, *, key: str = CONFIG_KEY):
        self.client = client
        self.key = key                       # per-tenant config lives under a distinct key

    def get(self) -> Optional[dict]:
        raw = self.client.get(self.key)
        return json.loads(raw) if raw else None

    def set(self, data: dict) -> None:
        self.client.set(self.key, json.dumps(data))

    def clear(self) -> None:
        self.client.delete(self.key)


class FileConfigStore:
    """Durable override on the PVC (same volume as the event log / tenant registry).

    Survives pod restarts, image updates AND a redis wipe — an operator-set value
    like final_limit must not silently revert to baseline on the next deploy."""
    def __init__(self, path: str):
        import os
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def get(self) -> Optional[dict]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def set(self, data: dict) -> None:
        import os
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self.path)           # atomic: readers never see a torn file

    def clear(self) -> None:
        import os
        try:
            os.remove(self.path)
        except OSError:
            pass


class InMemoryConfigStore:
    """Dev / test fallback (no Redis)."""
    def __init__(self):
        self._data: Optional[dict] = None

    def get(self) -> Optional[dict]:
        return self._data

    def set(self, data: dict) -> None:
        self._data = data

    def clear(self) -> None:
        self._data = None
