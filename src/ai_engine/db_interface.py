# %%
import random
import secrets
from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy import create_engine, text, URL
from sqlalchemy.exc import IntegrityError
import pandas as pd


from ai_engine.common import User, Event, Session
from ai_engine.config import (
    DB_NAME, 
    DB_HOST, 
    DB_USER, 
    DB_PASSWORD, 
    DB_PORT, 
    DB_DRIVERNAME,
    TABLE_USER,
    TABLE_USER_EVENT
)


class DB_Interface():
    def __init__(self):
        # Create engine
        self.engine = create_engine(
            url=URL.create(
                database=DB_NAME,
                host=DB_HOST,
                username=DB_USER,
                password=DB_PASSWORD,
                port=DB_PORT,
                drivername=DB_DRIVERNAME),
            pool_pre_ping=True
        )

    ### Fetch
    ######################

    def fetch_events_raw(self, user_id: int) -> pd.DataFrame:
        sql_query = f"""
        SELECT *
        FROM {TABLE_USER_EVENT}
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
        sql_query = f"""
        SELECT
            user_id,
            item_id,
            start_ts,
            close_ts,
            EXTRACT(EPOCH FROM (close_ts - start_ts)) AS dwell_seconds
        FROM (
            SELECT
                user_id,
                -- If item_id is a column: item_id
                -- If item_id is inside JSON: CAST(event_payload->>'item_id' AS INTEGER) as item_id
                -- CAST(event_payload->>'item_id' AS INTEGER) as item_id,
                item_id,
                ts AS start_ts,
                LEAD(ts) OVER (
                    PARTITION BY user_id, event_payload->>'item_id'
                    ORDER BY ts
                ) AS close_ts,
                event_type
            FROM {TABLE_USER_EVENT}
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
            sql_query = f"""
            SELECT * 
            FROM {TABLE_USER}
            WHERE id = :user_id;
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

    def register_user(self, user: User) -> dict:
        sql_query = """
        INSERT INTO visitor (
            license_plate, email, password_hash, age, gender, nationality, personal_connection, payload
        ) VALUES (
            :license_plate, :email, :password_hash, :age, :gender, :nationality, :personal_connection, :payload
        ) RETURNING id;
        """
        new_id = None

        # We try up to 3 times to get a unique plate
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            current_plate = self._generate_user_plate()
            
            try:
                params = user.to_dict()
                params['license_plate'] = current_plate
                with self.engine.connect() as connection:
                    with connection.begin() as transaction:
                        result = connection.execute(text(sql_query), parameters=params)
                        new_id = result.scalar()
                        transaction.commit()
                logger.info(f'Successful insert of user {new_id}')
                return {
                    "status": "ok",
                    "id": new_id, 
                    "license_plate": current_plate
                }
        
            except IntegrityError as e:
                # specifically catch collision errors (duplicate plate)
                error_msg = str(e.orig)
                
                if "visitor_license_plate_key" in error_msg:
                    logger.warning(f"Real collision on plate {current_plate}, retrying...")
                    attempt += 1
                elif "visitor_email_key" in error_msg:
                    # If email exists (and isn't None), we shouldn't retry with a new plate
                    logger.error(f"Email already exists: {user.email}")
                    return {"status": "failed", "message": "Email already exists"}
                else:
                    # If it's a NOT NULL violation (like ID), fail immediately so you see it
                    logger.error(f"Database Integrity Error: {error_msg}")
                    return {"status": "failed", "message": error_msg}
                
            
            except Exception as e:
                # Catch other syntax/connection errors and fail hard
                logger.exception(f"Database error: {e}")
                return {"status": "failed", "message": str(e)}
            
        return {"status": "failed", "message": "Max retries exceeded"}

    def register_event(self, event: Event) -> dict:
        sql_query = f"""
        INSERT INTO {TABLE_USER_EVENT} (
            user_id, session_id, item_id, event_type, event_payload, ts
        ) VALUES (
            :user_id, :session_id, :item_id, :event_type, :event_payload, :ts
        ) RETURNING id;
        """
        new_id = None
        try:
            params = event.to_dict()
            with self.engine.connect() as connection:
                with connection.begin() as transaction:
                    result = connection.execute(text(sql_query), parameters=params)
                    new_id = result.scalar()
                    transaction.commit()
            logger.info(f'Successful insert of event {event.event_type} for {event.item_id} and user {event.user_id}')
            return {
                "status": "ok",
                "id": new_id
            }
        
        except Exception as e:
            logger.exception(e)
            return {
                "status": "failed",
                "id": None
            }

    def register_session(self, session: Session) -> dict:
        """
        Creates a new device row (if needed) and a new session row using the input schema.
        """
        new_sess_token = secrets.token_urlsafe(32) 
        expires_at = datetime.now() + timedelta(days=session.expires_in_days)

        try:
            with self.engine.begin() as conn: # Transaction start
                
                # --- A. Get or Create Device (Logic using data.device_token) ---
                sql_query = "SELECT id FROM device WHERE device_id_token = :t"
                dev_id = conn.execute(text(sql_query), {"t": session.device_token}).scalar()
                
                if not dev_id:
                    # Insert new device row
                    sql_create_dev = """
                    INSERT INTO device (user_id, device_id_token, user_agent, last_ip)
                    VALUES (:uid, :token, :ua, :ip) RETURNING id;
                    """
                    dev_id = conn.execute(text(sql_create_dev), {
                        "uid": session.user_id, 
                        "token": session.device_token, 
                        "ua": session.user_agent, 
                        "ip": session.ip
                    }).scalar()
                    logger.info(f"Created NEW device with UUID: {dev_id}")
                else:
                    # Update last seen info for known device
                    conn.execute(text("UPDATE device SET last_seen_at = NOW(), last_ip = :ip WHERE id = :id"), 
                                {"ip": session.ip, "id": dev_id})
                    logger.info(f"Found EXISTING device with UUID: {dev_id}. Updated last_seen_at.")

                # --- B. Insert Session (Logic using data.user_id, dev_id, etc.) ---
                sql_session = """
                INSERT INTO session (user_id, device_id, session_token, ip, user_agent, expires_at)
                VALUES (:uid, :did, :tok, :ip, :ua, :exp) 
                RETURNING id;
                """
                sess_uuid = conn.execute(text(sql_session), {
                    "uid": session.user_id, 
                    "did": dev_id, 
                    "tok": new_sess_token, 
                    "ip": session.ip, 
                    "ua": session.user_agent, 
                    "exp": expires_at
                }).scalar()

            logger.info(
                f"Successful insert of Session. User ID: {session.user_id or 'GUEST'} "
                f"| Session ID: {sess_uuid} | Expires: {expires_at.strftime('%Y-%m-%d %H:%M')}"
            )

            return {
                "status": "ok", 
                "session_id": sess_uuid,     
                "session_token": new_sess_token
            }

        except Exception as e:
            logger.exception(f"Session creation failed: {e}")
            return {"status": "error"}

    ### Utils
    ######################

    @staticmethod
    def _generate_user_plate():
        # Define readable characters (no O, 0, I, 1 to prevent confusion)
        # chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        numbers = "23456789"
        
        # Generate format XXX-XXX
        part1 = "".join(random.choices(letters, k=2))
        part2 = "".join(random.choices(numbers, k=4))
        
        return f"{part1}-{part2}"
    
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
    import uuid

    client_db = DB_Interface()
    
    INSERT = True
    FETCH = True
    user_id = 10
    known_device_token = "TEST_DESKTOP_001"

    if INSERT:
        # Register User
        test_user = User(
            id = user_id,
            age = 24,
            gender = 'male',
            nationality = 'spain',
            personal_connection = False,
            payload = {}
        )
        result = client_db.register_user(test_user)
        test_user_id = result.get('id', None)
        print(result)
        print(test_user_id)

        # Register Session
        test_session = Session(
            user_id=user_id,
            device_token=known_device_token,
            ip="203.0.113.45",
        user_agent="TestRunner/AuthPytest"
        )
        result = client_db.register_session(test_session)
        print(result)
        result2 = client_db.register_session(test_session)
        print(result2)

        session_id = uuid.uuid4()
        # Register Events
        test_event_open = Event(
            user_id=test_user_id,
            session_id=session_id,
            item_id="1227",
            event_type='start',
            event_payload={
                "query": "Bergen-Belsen"
            },
            ts=datetime.now()
        )

        test_event_close = Event(
            user_id=test_user_id,
            session_id=session_id,
            item_id="1227",
            event_type='end',
            event_payload={
                "query": "Bergen-Belsen"
            },
            ts=datetime.now()
        )

        result_event_open = client_db.register_event(test_event_open)
        result_event_close = client_db.register_event(test_event_close)
        print(result_event_open)
        print(result_event_close)

    if FETCH:
        # Fetch events
        result_fetch_events = client_db.fetch_events(user_id=user_id)
        print(result_fetch_events)

    
