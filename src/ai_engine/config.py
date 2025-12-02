import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# Load .env only if it exists (for local/dev/docker compose)
# In Kubernetes, we rely on env vars.
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

# ---------------------------------------------------------------------------
# Enviroment Variables
# ---------------------------------------------------------------------------

##################
####### LLM  #####
##################

# OpenRouter
OPENROUTER_NARRATIVE_MODEL = os.getenv("OPENROUTER_NARRATIVE_MODEL", "arliai/qwq-32b-arliai-rpr-v1:free")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.dev.memorise.sdu.dk")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.3:latest")

# Keycloak
KEYCLOAK_BASE_URL  = os.getenv("KEYCLOAK_BASE_URL", "https://keycloak.dev.memorise.sdu.dk")
KEYCLOAK_REALM     = os.getenv("KEYCLOAK_REALM", "oauth2-proxy")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "oauth2-proxy")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")
KEYCLOAK_USERNAME = os.getenv("KEYCLOAK_USERNAME")
KEYCLOAK_PASSWORD = os.getenv("KEYCLOAK_PASSWORD")
KEYCLOAK_SAFETY_MARGIN_SECONDS = int(os.getenv("KEYCLOAK_SAFETY_MARGIN_SECONDS", "30"))

##################
##### Qdrant #####
##################

QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "omeka-items")
FIELD_NAME_GEO  = os.getenv("FIELD_NAME_GEO", "locations")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "5"))

##################
######  DB  ######
##################

# SQLite (local event tracking)
SQL_DB_NAME = "events.db" 
SQL_DB_URL = os.getenv("SQL_DATABASE_URL")
SQL_DB_KEY = os.getenv("SQL_DATABASE_KEY")

# Postgress
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_DRIVERNAME = os.getenv("DB_DRIVERNAME", "postgresql+psycopg")

TABLE_USER = os.getenv("TABLE_USER", "visitor")
TABLE_USER_EVENT = os.getenv("TABLE_USER_EVENT", "visitor_event")

# ---------------------------------------------------------------------------
# Application Parameters
# ---------------------------------------------------------------------------

##################
### User State ###
##################

READING_SPEED_WPS = float(os.getenv("READING_SPEED_WPS", "4.2")) # 250 words per minute / 60 seconds
IMG_EXTRA_FIXED_TIME = float(os.getenv("IMG_EXTRA_FIXED_TIME", "1.3")) # Assumed fixed time to view an image, in seconds
