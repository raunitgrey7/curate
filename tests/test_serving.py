import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from curate.config import Settings
from curate.data.synthetic import RetailConfig, generate_retail
from curate.features.store import FeatureStore
from curate.logging_utils import JsonFormatter, configure_logging, request_id_var
from curate.pipeline import STORE_FILE, ServingBundle, fit_for_serving
from curate.serving.app import create_app
from curate.serving.state import ServingState


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, monkeypatch_module):
    from curate import pipeline
    from curate.models import BPRRecommender, ItemKNNRecommender, PopularityRecommender, TwoTowerRecommender

    def fast_retrievers(seed, n_threads=2):
        return {
            "popularity": PopularityRecommender(),
            "itemknn": ItemKNNRecommender(n_neighbors=20),
            "bpr": BPRRecommender(dim=16, epochs=5, seed=seed, n_threads=n_threads),
            "two_tower": TwoTowerRecommender(
                dim=16, id_dim=8, hidden=32, epochs=4, seed=seed, n_threads=n_threads
            ),
        }

    monkeypatch_module.setattr(pipeline, "make_retrievers", fast_retrievers)
    path = tmp_path_factory.mktemp("artifacts")
    dataset, _ = generate_retail(
        RetailConfig(n_users=120, n_items=50, n_categories=4, latent_dim=8, mean_events_per_user=14.0, seed=2)
    )
    fit_for_serving(dataset, path, seed=2)
    return path


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


def _client(artifacts, api_key=None):
    settings = Settings(data_dir=artifacts, artifact_dir=artifacts, api_key=api_key)
    state = ServingState.from_bundle(
        ServingBundle.load(artifacts), FeatureStore(artifacts / STORE_FILE), settings
    )
    return TestClient(create_app(settings, state))


@pytest.fixture(scope="module")
def client(artifacts):
    with _client(artifacts) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["dataset"] == "retail_synthetic"
    assert set(body["models"]) == {"popularity", "itemknn", "bpr", "two_tower", "two_stage"}
    assert body["n_items"] == 50


def test_recommend_post_returns_ranked_items_with_explanations(client):
    response = client.post("/v1/recommend", json={"user_id": 1, "k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "two_stage"
    assert body["fallback"] == "none"
    assert [item["rank"] for item in body["items"]] == [1, 2, 3, 4, 5]
    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all(item["explanation"]["text"] for item in body["items"])
    assert body["request_id"] == response.headers["x-request-id"]
    assert float(response.headers["x-process-time-ms"]) >= 0


def test_recommend_excludes_history(client):
    history = {item["item_id"] for item in client.get("/v1/users/1/history?limit=1000").json()["items"]}
    recommended = {
        item["item_id"] for item in client.post("/v1/recommend", json={"user_id": 1, "k": 20}).json()["items"]
    }
    assert not history & recommended


def test_recommend_every_model(client):
    for model in ("popularity", "itemknn", "bpr", "two_tower", "two_stage"):
        body = client.post(
            "/v1/recommend", json={"user_id": 2, "k": 3, "model": model, "explain": False}
        ).json()
        assert body["model"] == model
        assert len(body["items"]) == 3
        assert body["items"][0].get("explanation") is None


def test_unknown_user_gets_popularity_fallback(client):
    body = client.post("/v1/recommend", json={"user_id": 10_000_000, "k": 3}).json()
    assert body["fallback"] == "popularity_cold_start"
    assert body["model"] == "popularity"
    assert body["items"][0]["explanation"]["kind"] == "popular"


@pytest.mark.parametrize(
    "payload",
    [
        {"user_id": -1, "k": 3},
        {"user_id": 1, "k": 0},
        {"user_id": 1, "k": 101},
        {"user_id": 1, "k": 3, "model": "oracle"},
        {"user_id": "one", "k": 3},
        {"user_id": 1, "k": 3, "unexpected": True},
        {},
    ],
)
def test_invalid_requests_are_rejected_with_422(client, payload):
    response = client.post("/v1/recommend", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert body["request_id"]


def test_malformed_json_body_is_rejected(client):
    response = client.post(
        "/v1/recommend", content=b'{"user_id": 1, "k": ', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


def test_get_recommend_route(client):
    response = client.get("/v1/recommend/3?k=4&model=itemknn")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 4
    assert client.get("/v1/recommend/3?k=0").status_code == 422


def test_similar_items_and_unknown_item(client):
    body = client.get("/v1/items/1000/similar?k=3").json()
    assert body["item"]["item_id"] == 1000
    assert 1 <= len(body["neighbours"]) <= 3
    assert all(n["similarity"] > 0 for n in body["neighbours"])
    assert client.get("/v1/items/1/similar").status_code == 404
    assert client.get("/v1/items/1000").json()["title"].startswith("SKU-0000")
    assert client.get("/v1/items/77777").status_code == 404


def test_history_endpoint(client):
    body = client.get("/v1/users/1/history?limit=3").json()
    assert body["n_interactions"] >= 5
    assert len(body["items"]) == 3
    assert client.get("/v1/users/424242/history").status_code == 404


def test_models_endpoint_and_root(client):
    body = client.get("/v1/models").json()
    assert body["default"] == "two_stage"
    assert "ranker" in body["train_seconds"]
    assert client.get("/").status_code == 200


def test_metrics_endpoint_exposes_prometheus_text(client):
    client.post("/v1/recommend", json={"user_id": 4, "k": 2})
    text = client.get("/metrics").text
    assert "curate_http_requests_total" in text
    assert 'curate_recommendations_total{model="two_stage"}' in text
    assert "curate_http_request_duration_seconds_bucket" in text
    assert 'route="/v1/recommend"' in text


def test_request_id_is_propagated_when_supplied(client):
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"


def test_api_key_enforced_when_configured(artifacts):
    with _client(artifacts, api_key="s3cret") as c:
        assert c.get("/health").status_code == 200
        assert c.get("/metrics").status_code == 200
        denied = c.post("/v1/recommend", json={"user_id": 1, "k": 2})
        assert denied.status_code == 401
        assert denied.json()["detail"] == "invalid or missing API key"
        assert (
            c.post("/v1/recommend", json={"user_id": 1, "k": 2}, headers={"X-API-Key": "wrong"}).status_code
            == 401
        )
        assert (
            c.post("/v1/recommend", json={"user_id": 1, "k": 2}, headers={"X-API-Key": "s3cret"}).status_code
            == 200
        )


def test_adversarial_inputs_do_not_crash(client):
    assert client.post("/v1/recommend", json={"user_id": 2**63, "k": 2}).status_code in (200, 422)
    assert client.get("/v1/items/%27%20OR%201=1").status_code == 422
    assert client.get("/v1/users/1/history?limit=-5").status_code == 422
    huge = "x" * 100_000
    assert client.post("/v1/recommend", json={"user_id": 1, "k": 2, "model": huge}).status_code == 422
    assert client.get("/health").status_code == 200


def test_recommendations_are_deterministic_across_calls(client):
    first = client.post("/v1/recommend", json={"user_id": 5, "k": 10}).json()["items"]
    second = client.post("/v1/recommend", json={"user_id": 5, "k": 10}).json()["items"]
    assert [i["item_id"] for i in first] == [i["item_id"] for i in second]


def test_json_formatter_emits_one_json_object_with_request_id():
    formatter = JsonFormatter()
    token = request_id_var.set("req-9")
    try:
        record = logging.LogRecord("curate.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        record.duration_ms = 1.5
        payload = json.loads(formatter.format(record))
    finally:
        request_id_var.reset(token)
    assert payload["msg"] == "hello world"
    assert payload["request_id"] == "req-9"
    assert payload["duration_ms"] == 1.5
    assert payload["level"] == "INFO"


def test_configure_logging_installs_json_handler():
    import io

    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    logging.getLogger("curate.x").info("structured", extra={"k": 1})
    line = stream.getvalue().strip().splitlines()[-1]
    assert json.loads(line)["k"] == 1


def test_serving_state_direct_calls(artifacts):
    settings = Settings(data_dir=artifacts, artifact_dir=artifacts)
    state = ServingState.load(settings)
    result = state.recommend(1, 3, "itemknn")
    assert len(result["items"]) == 3
    with pytest.raises(KeyError):
        state.recommend(1, 3, "nope")
    assert state.history(1, 2)["n_interactions"] > 0
    assert state.history(-5, 2) is None
    assert np.isfinite(result["model_latency_ms"])
