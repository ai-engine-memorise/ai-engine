import os
from dotenv import load_dotenv

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

load_dotenv()

##################
####### LLM  #####
##################

OPENROUTER_NARRATIVE_MODEL = 'arliai/qwq-32b-arliai-rpr-v1:free'
OPENROUTER_API_URL = os.environ.get("OPENROUTER_API_URL")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Ollama
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "https://ollama.dev.memorise.sdu.dk",  # sensible default for your setup
)
OLLAMA_MODEL = 'llama3.3:latest' # This MUST be a model pulled on your cluster

# Keycloak
KEYCLOAK_BASE_URL = os.environ.get(
    "KEYCLOAK_BASE_URL",
    "https://keycloak.dev.memorise.sdu.dk",
)
KEYCLOAK_REALM = os.environ.get(
    "KEYCLOAK_REALM",
    "oauth2-proxy",
)
KEYCLOAK_CLIENT_ID = os.environ.get(
    "KEYCLOAK_CLIENT_ID",
    "oauth2-proxy",
)
KEYCLOAK_CLIENT_SECRET = os.environ.get("KEYCLOAK_CLIENT_SECRET")
KEYCLOAK_USERNAME = os.environ.get("KEYCLOAK_USERNAME")
KEYCLOAK_PASSWORD = os.environ.get("KEYCLOAK_PASSWORD")
KEYCLOAK_SAFETY_MARGIN_SECONDS = int(
    os.environ.get("KEYCLOAK_SAFETY_MARGIN_SECONDS", "30")
)

##################
##### Qdrant #####
##################

QDRANT_API_URL = os.environ.get("QDRANT_API_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

COLLECTION_NAME = "omeka-items"
FIELD_NAME_GEO = "locations"

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

SEARCH_LIMIT = 5


##################
##### SQLite #####
##################

SQL_DB_NAME = "events.db" 
SQL_DB_URL = os.environ.get("SQL_DATABASE_URL")
SQL_DB_KEY = os.environ.get("SQL_DATABASE_KEY")

DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_DRIVERNAME = os.getenv("DB_DRIVERNAME", "postgresql+psycopg")

TABLE_USER = "visitor"
TABLE_USER_EVENT = "visitor_event"


##################
### User State ###
##################

READING_SPEED_WPS = 4.2       # 250 words per minute / 60 seconds
IMG_EXTRA_FIXED_TIME = 1.3   # Assumed fixed time to view an image, in seconds