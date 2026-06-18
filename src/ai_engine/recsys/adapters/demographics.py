"""DemographicsProvider implementations.

- NullDemographicsProvider: default, no demographics.
- StaticDemographicsProvider: in-memory map (tests / fixed configs).
- PostgresDemographicsProvider: reads the visitor table via the existing DB_Interface.
  NOTE: assumes the event user_id maps to visitor.id (integer). If your RudderStack
  userId is not the visitor PK, supply a mapping or use survey-event demographics.
"""
from __future__ import annotations
from typing import Optional


class NullDemographicsProvider:
    def get_demographics(self, user_id: str) -> dict:
        return {}


class StaticDemographicsProvider:
    def __init__(self, data: dict[str, dict]):
        self._data = data

    def get_demographics(self, user_id: str) -> dict:
        return dict(self._data.get(str(user_id), {}))


class PostgresDemographicsProvider:
    def __init__(self, db=None):
        if db is None:
            from ai_engine.db_interface import DB_Interface
            db = DB_Interface()
        self.db = db

    def get_demographics(self, user_id: str) -> dict:
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            return {}
        df = self.db.fetch_user(user_id=uid)
        if df is None or df.empty:
            return {}
        row = df.to_dict(orient="records")[0]
        return {
            "age": row.get("age"),
            "gender": row.get("gender"),
            "nationality": row.get("nationality"),
            "personal_connection": row.get("personal_connection"),
        }
