"""Evaluation tool: synthetic personas -> recommendation scenarios + metrics."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from ai_engine.recsys.api import create_app
from ai_engine.recsys.evaluation import match_tags, generate_persona, list_metrics


def test_match_tags_grounds_in_real_vocab():
    vocab = {"tags": ["theme_what:resistance", "theme_what:children", "location:Barrack 3"]}
    aff = match_tags("a young visitor drawn to resistance and children", vocab)
    assert "theme_what:resistance" in aff and "theme_what:children" in aff
    assert max(aff.values()) == 1.0                       # normalized
    assert match_tags("nothing matches xyzzy", vocab) == {}   # only real tags, never invented


def test_generate_persona_infers_demographics_and_tags():
    vocab = {"tags": ["theme_what:resistance", "person_who.age_group:age 18-24"]}
    p = generate_persona("young student interested in resistance", vocab)
    assert p["spec"]["demographics"].get("age_group") == "18_24"
    assert "theme_what:resistance" in p["spec"]["tag_affinity"]
    assert p["matched_tags"]                                # surfaced for scrutability


def test_list_metrics_shape():
    items = [{"id": "1", "role": "target", "relevance_score": 0.8, "content": {"tags": [{"facet": "theme_what", "label": "resistance"}]}},
             {"id": "2", "role": "distractor", "relevance_score": 0.1, "content": {"tags": []}}]
    m = list_metrics(items, {"1": [1.0, 0.0], "2": [0.0, 1.0]}, "warm")
    assert m["n_items"] == 2 and m["n_targets"] == 1
    assert m["distractor_present"] is True
    assert m["intra_list_diversity"] == 1.0                 # orthogonal vectors -> max diversity
    assert m["strategy"] == "warm"


def test_eval_endpoints_end_to_end():
    client = TestClient(create_app())                        # fakes (dev fixtures)
    personas = client.get("/api/eval/personas").json()["result"]
    assert personas and all("spec" in p for p in personas)

    gen = client.post("/api/eval/generate", json={"prompt": "resistance and children"}).json()["result"]
    run = client.post("/api/eval/run",
                      json={"spec": gen["spec"], "scenarios": [{"name": "open"}], "cold": True}).json()["result"]
    names = [s["name"] for s in run["scenarios"]]
    assert "open" in names and "cold-start" in names
    for s in run["scenarios"]:
        assert "metrics" in s and "items" in s and "strategy" in s
