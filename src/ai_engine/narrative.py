## This file should contain the class / methods for narrative generation
# Let's explore the concept of narrative as an LLM based sequence of content tailored to the user

### Given a List[Item] or IDs to retrieve from DB
### LLM + Qdrant Usage  --- What LLM framework supported in MEMORISE
### Output text with all the references to sources

import time
import requests
import json
from typing import Optional, List, Dict, Any
from loguru import logger
from openai import OpenAI
from ai_engine.common import NarrativeResult
from ai_engine.config import (
    OPENROUTER_API_URL, OPENROUTER_API_KEY, OPENROUTER_NARRATIVE_MODEL,
    KEYCLOAK_BASE_URL,
    KEYCLOAK_REALM,
    KEYCLOAK_CLIENT_ID,
    KEYCLOAK_CLIENT_SECRET,
    KEYCLOAK_USERNAME,
    KEYCLOAK_PASSWORD,
    KEYCLOAK_SAFETY_MARGIN_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
SYSTEM_PROMPT = """
You are a narrative designer for a collection browser.

Given a small set of related items (documents, photos, testimonies, etc.),
you must:

1. Detect high-level relationships between items:
   - chronology (earlier → later)
   - geography (same place, nearby places)
   - thematic similarity (shared topics/tags)
   - causality or influence (one item explains or causes another)
   - contrast (different perspectives on same theme)

2. Organize the items into a sequence of 3–7 narrative segments.
   Each segment should:
   - group 1–5 closely related items
   - have a short headline
   - have a short summary for the user
   - list which items belong to it (by item_id)
   - highlight a few key relationships between items in that segment

3. Design smooth transitions between segments so a user can “follow the story”.

4. Always preserve item_id references so the application can link back to the originals.

Output ONLY valid JSON matching this schema:

{SCHEMA}

Keep summaries concise and user-friendly (2–4 sentences per segment).
Use neutral, descriptive language.
"""

SCHEMA_EXAMPLE = {
    "narrative_title": "string",
    "overview": "string",
    "segments": [
        {
            "segment_id": "string",
            "headline": "string",
            "summary": "string",
            "item_ids": ["string"],
            "relationships": [
                {
                    "from": "string",
                    "to": "string",
                    "type": "string",
                    "explanation": "string"
                }
            ],
            "transition_to_next": "string"
        }
    ],
    "suggested_start_item_id": "string"
}


class NarrativeGenerator:

    def __init__(self, model: str = OLLAMA_MODEL, prompt: str = SYSTEM_PROMPT):
        # Ollama connection
        self.model = model
        token_client = KeycloakTokenClient()
        self.llm_client = OllamaClient(token_client=token_client)
        # self.llm_client = OpenRouterAdapter(
        #     model=self.model
        # )
        self.prompt = prompt
        logger.info(f"Model: {OLLAMA_MODEL} | {self.llm_client.base_url}")

    def generate_narrative(self, items: List[Dict[Any, str]]) -> NarrativeResult:
        
        schema_str = json.dumps(SCHEMA_EXAMPLE, indent=2)
        system_prompt = SYSTEM_PROMPT.replace("{SCHEMA}", schema_str)
        user_prompt = self.build_user_prompt(items)

        # Generation via LLM
        response = self.llm_client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

        raw = response["message"]["content"]

        print(raw)
        # data = json.loads(raw)
        # narrative = NarrativeResult.model_validate(data)  # Pydantic v2
        return None

    def build_user_prompt(self, items: List[Dict[Any, str]]) -> str:
        return f"""
        Here is the list of items you should work with (JSON):

        {json.dumps([item['payload'] for item in items], ensure_ascii=False, indent=2)}

        Now:

        - Identify the main themes and patterns.
        - Decide a good order to explore these items.
        - Fill the JSON you were instructed to output, using the item_id fields exactly as given.
        """


class OpenRouterAdapter:
    """
    Acts as a drop-in replacement for ollama.Client() for testing.
    Uses the OpenAI SDK pointed at the OpenRouter API.
    """
    def __init__(self, model: str):
        # The key is setting the base_url to the OpenRouter endpoint
        self.client = OpenAI(
            base_url=OPENROUTER_API_URL,
            api_key=OPENROUTER_API_KEY
        )
        # Use the chosen free model from OpenRouter's catalog
        self.model = model
        logger.info(f"Using OpenRouter Adapter (Model: {self.model}) as Ollama substitute.")

    def chat(self, model: str, messages: list) -> dict:
        """Simulates the ollama.Client().chat() method structure."""
        try:
            # The API call uses the OpenAI format
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            
            # Format the response to match the Ollama/OpenAI SDK output structure
            return {
                "model": model, 
                "created_at": "N/A (OpenRouter)",
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                },
                "done": True,
            }
        
        except Exception as e:
            error_message = f"OpenRouter API Error: {e}"
            logger.error(f"ERROR: {error_message}")
            return {
                "message": {"content": f"ERROR: Could not get a response. {error_message}"}
            }


class KeycloakTokenClient:
    """
    Fetches and refreshes Keycloak tokens as needed.
    """

    def __init__(
        self,
        base_url: str = KEYCLOAK_BASE_URL,
        realm: str = KEYCLOAK_REALM,
        client_id: str = KEYCLOAK_CLIENT_ID,
        client_secret: str = KEYCLOAK_CLIENT_SECRET,
        username: str = KEYCLOAK_USERNAME,
        password: str = KEYCLOAK_PASSWORD,
        safety_margin_seconds: int = 30,
    ):
        """
        base_url: e.g. "https://keycloak.dev.memorise.sdu.dk"
        realm:    e.g. "oauth2-proxy"
        """
        self.token_url = f"{base_url}/realms/{realm}/protocol/openid-connect/token"
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.safety_margin_seconds = safety_margin_seconds

        # runtime state
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._access_token_expires_at: float = 0.0

    def get_access_token(self) -> str:
        """
        Public API:
        Always returns a valid (or freshly refreshed) access token.
        """
        if not self._access_token or self._is_about_to_expire():
            self._ensure_token()
        return self._access_token

    # --------------- internal helpers ---------------

    def _is_about_to_expire(self) -> bool:
        now = time.time()
        return now >= (self._access_token_expires_at - self.safety_margin_seconds)

    def _ensure_token(self) -> None:
        """
        Ensure we have a valid token:
          - try refresh if we have a refresh_token
          - otherwise, or if refresh fails, do full login
        """
        if self._refresh_token:
            if self._refresh():
                return

        # either we have no refresh_token or refresh failed
        self._login()

    def _login(self) -> None:
        """
        Do password grant flow: username + password -> access_token + refresh_token
        """
        data = {
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": self.password,
        }

        resp = requests.post(self.token_url, data=data)
        resp.raise_for_status()
        token_data = resp.json()
        self._store_token_data(token_data)

    def _refresh(self) -> bool:
        """
        Try to refresh using refresh_token. Returns True on success.
        """
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self._refresh_token,
        }

        resp = requests.post(self.token_url, data=data)
        if not resp.ok:
            return False

        token_data = resp.json()
        self._store_token_data(token_data)
        return True

    def _store_token_data(self, token_data: Dict[str, Any]) -> None:
        """
        Store access_token, refresh_token, and compute expiry timestamp.
        """
        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 300)  # 5 min default if missing
        self._access_token_expires_at = time.time() + float(expires_in)


class OllamaClient:
    """
    Simple Ollama client that uses KeycloakTokenClient for auth.
    Can be used as a backend where you'd otherwise use OpenRouterAdapter.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL, *, token_client: KeycloakTokenClient):
        """
        base_url: e.g. "https://ollama.dev.memorise.sdu.dk"
        """
        self.base_url = base_url.rstrip("/")
        self.token_client = token_client
        self.session = requests.Session()
        self.model = model
        logger.info(f"Using OllamaClient with Keycloak auth (base_url={self.base_url})")

    # -------------------------------
    # Internal helpers
    # -------------------------------
    def _auth_headers(self) -> Dict[str, str]:
        token = self.token_client.get_access_token()
        return {"Authorization": f"Bearer {token}"}

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update(self._auth_headers())
        resp = self.session.request(method, url, headers=headers, **kwargs)
        resp.raise_for_status()
        return resp

    # -------------------------------
    # Model management / info
    # -------------------------------
    def list_models(self) -> Dict[str, Any]:
        """
        GET /api/tags
        Returns a dict of available models.
        """
        resp = self._request("GET", "/api/tags")
        return resp.json()

    def show_model(self, name: str) -> Dict[str, Any]:
        """
        POST /api/show
        Get detailed info for a model.
        """
        resp = self._request("POST", "/api/show", json={"name": name})
        return resp.json()

    # -------------------------------
    # Text generation (/api/)
    # -------------------------------
    def chat(self, model: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call POST /v1/chat/completions and normalize the result to:

        {
          "model": model,
          "created_at": "...",
          "message": {
            "role": "assistant",
            "content": "<string>"
          },
          "done": True
        }
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            # ask the model to return a JSON object (for your NarrativeResult)
            "response_format": {"type": "json_object"},
        }

        try:
            resp = self.session.post(
                url,
                headers=self._auth_headers(),
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            # OpenAI-style response:
            # {
            #   "id": "...",
            #   "object": "chat.completion",
            #   "created": 1234567890,
            #   "model": "...",
            #   "choices": [
            #       { "index": 0, "message": {"role": "assistant", "content": "..."} }
            #   ],
            #   ...
            # }
            choice = data["choices"][0]["message"]

            return {
                "model": data.get("model", model),
                "created_at": data.get("created", "N/A (on-prem OpenAI style)"),
                "message": {
                    "role": choice.get("role", "assistant"),
                    "content": choice.get("content", ""),
                },
                "done": True,
            }

        except Exception as e:
            error_message = f"On-prem OpenAI-style API Error: {e}"
            logger.error(error_message)
            return {
                "message": {
                    "content": f"ERROR: Could not get a response. {error_message}"
                }
            }
    
    # -------------------------------
    # Embeddings (/api/embeddings)
    # -------------------------------
    def embeddings(
        self,
        model: str,
        input_text: str,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call the Ollama /api/embeddings endpoint.

        Returns whatever Ollama returns, typically:
        {
          "embedding": [float, float, ...]
        }
        """
        payload = {
            "model": model,
            "prompt": input_text,
        }
        if extra_params:
            payload.update(extra_params)

        resp = self._request(
            "POST",
            "/api/embeddings",
            json=payload,
        )
        return resp.json()


if __name__ == '__main__':
    items = [
        {
            "id": 1251,
            "source": "vector",
            "score": 0.93264604,
            "payload": {
                "id": 1251,
                "title": "\"Bergen-Belsen\"",
                "public_url": "https://bb-g.futurememoryfoundation.org/items/show/1251",
                "text": None,
                "creator": "Valerij Talantow",
                "locations": [
                    {
                    "lat": 52.76277,
                    "lon": 9.90721
                    },
                    {
                    "lat": 52.76277,
                    "lon": 9.90721
                    }
                ],
                "geo_metadata": {
                    "horizon_height_estimate": None,
                    "label_elevation_correction": None,
                    "label_latitude": None,
                    "label_longitude": None,
                    "origin_place_altitude": None,
                    "origin_place_latitude": "52.76277",
                    "origin_place_longitude": "9.90721",
                    "reference_place_elevation": None,
                    "reference_place_latitude": None,
                    "reference_place_longitude": None,
                    "viewpoint_altitude": None,
                    "viewpoint_elevation": None,
                    "viewpoint_latitude": "52.76277",
                    "viewpoint_longitude": "9.90721"
                },
                "time_metadata": {
                    "date_available": None,
                    "dates_of_creation": "[xx/xx/1944]-[xx/xx/1944]",
                    "dates_of_reference": "[xx/07/1943]-[xx/01/1945]",
                    "earliest_start_of_reference_period": None,
                    "latest_end_of_reference_period": None
                },
                "files_url": [
                    "https://bb-g.futurememoryfoundation.org/files/original/a210111af25128618f57cb5fb2c69973.jpg"
                ],
                "image_url": "https://bb-g.futurememoryfoundation.org/files/original/a210111af25128618f57cb5fb2c69973.jpg"
                },
            "highlight": ""
        },
        {
            "id": 1159,
            "source": "vector",
            "score": 0.8502568,
            "payload": {
                "id": 1159,
                "title": "\"Bergen Belsen 10.C.\"",
                "public_url": "https://bb-g.futurememoryfoundation.org/items/show/1159",
                "text": None,
                "creator": "Istvan Irsai",
                "locations": [
                    {
                    "lat": 52.76222005,
                    "lon": 9.912284291
                    },
                    {
                    "lat": 52.76182911,
                    "lon": 9.91268
                    },
                    {
                    "lat": 52.762218,
                    "lon": 9.912289868
                    }
                ],
                "geo_metadata": {
                    "horizon_height_estimate": None,
                    "label_elevation_correction": None,
                    "label_latitude": None,
                    "label_longitude": None,
                    "origin_place_altitude": None,
                    "origin_place_latitude": "52.76222005",
                    "origin_place_longitude": "9.912284291",
                    "reference_place_elevation": None,
                    "reference_place_latitude": "52.76182911",
                    "reference_place_longitude": "9.91268",
                    "viewpoint_altitude": None,
                    "viewpoint_elevation": None,
                    "viewpoint_latitude": "52.762218",
                    "viewpoint_longitude": "9.912289868"
                },
                "time_metadata": {
                    "date_available": None,
                    "dates_of_creation": "09/07/1944 - 04/12/1944",
                    "dates_of_reference": "09/07/1944 - 04/12/1944",
                    "earliest_start_of_reference_period": "1944-07-09",
                    "latest_end_of_reference_period": "1944-12-04"
                },
                "files_url": [
                    "https://bb-g.futurememoryfoundation.org/files/original/1e8e360a93db3e80e82e3b59a25bc031.jpg"
                ],
                "image_url": "https://bb-g.futurememoryfoundation.org/files/original/1e8e360a93db3e80e82e3b59a25bc031.jpg"
                },
            "highlight": ""
        },
        {
            "id": 1152,
            "source": "vector",
            "score": 0.83762693,
            "payload": {
                "id": 1152,
                "title": "\"Bergen-Belsen, camp I\"",
                "public_url": "https://bb-g.futurememoryfoundation.org/items/show/1152",
                "text": None,
                "creator": "Georges Frejafón",
                "locations": [
                    {
                    "lat": 52.7629888,
                    "lon": 9.914174171
                    },
                    {
                    "lat": 52.76233433,
                    "lon": 9.913776063
                    },
                    {
                    "lat": 52.76295363,
                    "lon": 9.914380254
                    }
                ],
                "geo_metadata": {
                    "horizon_height_estimate": None,
                    "label_elevation_correction": None,
                    "label_latitude": None,
                    "label_longitude": None,
                    "origin_place_altitude": None,
                    "origin_place_latitude": "52.7629888",
                    "origin_place_longitude": "9.914174171",
                    "reference_place_elevation": None,
                    "reference_place_latitude": "52.76233433",
                    "reference_place_longitude": "9.913776063",
                    "viewpoint_altitude": None,
                    "viewpoint_elevation": None,
                    "viewpoint_latitude": "52.76295363",
                    "viewpoint_longitude": "9.914380254"
                },
                "time_metadata": None,
                "files_url": [
                    "https://bb-g.futurememoryfoundation.org/files/original/e5bc0606c40964cf1521f5af2316200c.jpg"
                ],
                "image_url": "https://bb-g.futurememoryfoundation.org/files/original/e5bc0606c40964cf1521f5af2316200c.jpg"
                },
            "highlight": ""
        },
        {
            "id": 1336,
            "source": "vector",
            "score": 0.7759917,
            "payload": {
                "id": 1336,
                "title": "Aerial view of Bergen-Belsen",
                "public_url": "https://bb-g.futurememoryfoundation.org/items/show/1336",
                "text": None,
                "creator": "Lt. Parfitt",
                "locations": [
                    {
                    "lat": 52.76342733899432,
                    "lon": 9.91646468639374
                    },
                    {
                    "lat": 52.76361560944056,
                    "lon": 9.914672970771791
                    },
                    {
                    "lat": 52.763265036232205,
                    "lon": 9.91696357727051
                    }
                ],
                "geo_metadata": {
                    "horizon_height_estimate": "0.92",
                    "label_elevation_correction": None,
                    "label_latitude": None,
                    "label_longitude": None,
                    "origin_place_altitude": "200",
                    "origin_place_latitude": "52.763265036232205",
                    "origin_place_longitude": "9.91696357727051",
                    "reference_place_elevation": None,
                    "reference_place_latitude": "52.763615609440556",
                    "reference_place_longitude": "9.914672970771791",
                    "viewpoint_altitude": None,
                    "viewpoint_elevation": "200",
                    "viewpoint_latitude": "52.76342733899432",
                    "viewpoint_longitude": "9.91646468639374"
                },
                "time_metadata": {
                    "date_available": None,
                    "dates_of_creation": "6/1945",
                    "dates_of_reference": "6/1945",
                    "earliest_start_of_reference_period": None,
                    "latest_end_of_reference_period": None
                },
                "files_url": [
                    "https://bb-g.futurememoryfoundation.org/files/original/4a99389554edf802ecd7c429e121a95f.jpg"
                ],
                "image_url": "https://bb-g.futurememoryfoundation.org/files/original/4a99389554edf802ecd7c429e121a95f.jpg"
                },
            "highlight": ""
        },
        {
            "id": 1136,
            "source": "vector",
            "score": 0.7731792,
            "payload": {
                "id": 1136,
                "title": "Part of the camp Bergen-Belsen 1945",
                "public_url": "https://bb-g.futurememoryfoundation.org/items/show/1136",
                "text": None,
                "creator": "Ervin Abádi",
                "locations": [
                    {
                    "lat": 52.75932269,
                    "lon": 9.909542484
                    },
                    {
                    "lat": 52.75962941,
                    "lon": 9.909369273
                    },
                    {
                    "lat": 52.75932269,
                    "lon": 9.909542484
                    }
                ],
                "geo_metadata": {
                    "horizon_height_estimate": None,
                    "label_elevation_correction": None,
                    "label_latitude": None,
                    "label_longitude": None,
                    "origin_place_altitude": None,
                    "origin_place_latitude": "52.75932269",
                    "origin_place_longitude": "9.909542484",
                    "reference_place_elevation": None,
                    "reference_place_latitude": "52.75962941",
                    "reference_place_longitude": "9.909369273",
                    "viewpoint_altitude": None,
                    "viewpoint_elevation": None,
                    "viewpoint_latitude": "52.75932269",
                    "viewpoint_longitude": "9.909542484"
                },
                "time_metadata": {
                    "date_available": None,
                    "dates_of_creation": "xx/04/1945 - xx/12/1945",
                    "dates_of_reference": "14/12/1944 - 10/04/1945",
                    "earliest_start_of_reference_period": "1944-12-14",
                    "latest_end_of_reference_period": "1945-04-07"
                },
                "files_url": [
                    "https://bb-g.futurememoryfoundation.org/files/original/4de192dbd5c79f60d786c0f6b00e63cf.jpg"
                ],
                "image_url": "https://bb-g.futurememoryfoundation.org/files/original/4de192dbd5c79f60d786c0f6b00e63cf.jpg"
                },
            "highlight": ""
        }
    ]
    
    # from loguru import logger

    # logger.info(f"KEYCLOAK_BASE_URL={KEYCLOAK_BASE_URL}")
    # logger.info(f"KEYCLOAK_REALM={KEYCLOAK_REALM}")
    # logger.info(f"KEYCLOAK_CLIENT_ID={KEYCLOAK_CLIENT_ID}")
    # logger.info(f"KEYCLOAK_USERNAME={KEYCLOAK_USERNAME}")
    # logger.info(f"KEYCLOAK_PASSWORD set? {'YES' if KEYCLOAK_PASSWORD else 'NO'}")
    # logger.info(f"KEYCLOAK_CLIENT_SECRET set? {'YES' if KEYCLOAK_CLIENT_SECRET else 'NO'}")

    # token_client = KeycloakTokenClient()
    # llm_client = OllamaClient(token_client=token_client)
    # print(llm_client.list_models())

    narrative_gen = NarrativeGenerator(OLLAMA_MODEL)
    logger.info(f"Items: {[item['id'] for item in items]}")
    narrative = narrative_gen.generate_narrative(items = items)
    logger.info(f"Narrative:\n{narrative}")