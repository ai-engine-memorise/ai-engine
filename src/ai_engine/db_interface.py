# %%
from loguru import logger
from typing import Dict, Any
from sqlalchemy import create_engine, text, URL
import pandas as pd

from ai_engine.common import User, Event
from ai_engine.config import SQL_DB_NAME


class DB_Interface():
    def __init__(self):
        # Create engine
        self.engine = create_engine(
            url=URL.create(
                database=SQL_DB_NAME,
                drivername='sqlite'),
            pool_pre_ping=True
        )

    ### Fetch
    ######################

    def fetch_events_raw(self, user_id: int) -> pd.DataFrame:
        sql_query = """
        SELECT *
        FROM user_events
        WHERE user_id = :user_id;
        """
        
        try:
            with self.engine.connect() as connection:
                return pd.read_sql(
                    text(sql_query), 
                    params={'user_id': user_id}, 
                    con=connection,
                    dtype_backend='numpy_nullable'
                )
        except Exception as e:
            logger.exception(e)
            return pd.DataFrame()
    
    def fetch_events(self, user_id: int) -> pd.DataFrame:
        sql_query = """
        SELECT
            user_id,
            item_id,
            start_ts,
            close_ts,
            CAST((JULIANDAY(close_ts) - JULIANDAY(start_ts)) * 86400.0 AS REAL) AS dwell_seconds
        FROM (
            SELECT
                user_id,
                item_id,
                ts AS start_ts,
                LEAD(ts) OVER (
                    PARTITION BY user_id, item_id
                    ORDER BY ts
                ) AS close_ts,
                event_type
            FROM user_events
            WHERE user_id = :user_id
        ) AS t
        WHERE event_type = 'start' AND close_ts IS NOT NULL;
        """

        try:
            with self.engine.connect() as connection:
                return pd.read_sql(
                    text(sql_query), 
                    params={'user_id': user_id}, 
                    con=connection,
                    dtype_backend='numpy_nullable'
                )
        except Exception as e:
            logger.exception(e)
            return pd.DataFrame()

    def fetch_user(self, user_id: int) -> pd.DataFrame:
            sql_query = """
            SELECT * 
            FROM users
            WHERE user_id = :user_id;
            """
            
            try:
                with self.engine.connect() as connection:
                    return pd.read_sql(
                        text(sql_query), 
                        params={'user_id': user_id}, 
                        con=connection,
                        dtype_backend='numpy_nullable'
                    )
            except Exception as e:
                logger.exception(e)
                return pd.DataFrame()

    ### Insert
    ######################

    def register_user(self, user: User) -> str:
        sql_query = """
        INSERT INTO users (
            user_id, age, nationality, personal_connection, payload
        ) VALUES (
            :id, :age, :nationality, :personal_connection, NULL
        )
        """
        new_id = None
        try:
            params = user.to_dict()
            with self.engine.connect() as connection:
                with connection.begin() as transaction:
                    result = connection.execute(text(sql_query), parameters=params)
                    if result.rowcount == 1:
                        new_id = result.lastrowid
                    transaction.commit()
            logger.info(f'Successful insert of user {user.id}')
            return f"{new_id}"
        
        except Exception as e:
            logger.exception(e)
            return "failed"

    def register_event(self, event: Event) -> str:
        sql_query = """
        INSERT INTO user_events (
            user_id, item_id, event_type, ts, payload
        ) VALUES (
            :user_id, :item_id, :event_type, :ts, NULL
        )
        """
        new_id = None
        try:
            params = event.to_dict()
            with self.engine.connect() as connection:
                with connection.begin() as transaction:
                    result = connection.execute(text(sql_query), parameters=params)
                    if result.rowcount == 1:
                        new_id = result.lastrowid
                    transaction.commit()
            logger.info(f'Successful insert of event {event.event_type} for {event.item_id} and user {event.user_id}')
            return f"{new_id}"
        
        except Exception as e:
            logger.exception(e)
            return "failed"

    ### Debug
    ######################

    def test_connection(self):
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Connection successful", result.scalar())
        except Exception as e:
            logger.exception("Connection failed:", e)


if __name__ == "__main__":
    from datetime import datetime
    client_db = DB_Interface()
    
    INSERT = True
    FETCH = True

    if INSERT:
        # Register User
        test_user = User(
            id = 10,
            age = 24,
            gender = 'male',
            nationality = 'spain',
            personal_connection = False
        )
        result = client_db.register_user(test_user)
        print(result)

        # Register Events
        test_event_open = Event(
            id = 1,
            user_id=test_user.id,
            item_id=1227,
            event_type='start',
            ts=datetime.now()
        )

        test_event_close = Event(
            id = 2,
            user_id=test_user.id,
            item_id=1227,
            event_type='end',
            ts=datetime.now()
        )

        result_event_open = client_db.register_event(test_event_open)
        result_event_close = client_db.register_event(test_event_close)
        print(result_event_open)
        print(result_event_close)

    if FETCH:
        # Fetch events
        result_fetch_events = client_db.fetch_events(user_id=10)
        print(result_fetch_events)

    

