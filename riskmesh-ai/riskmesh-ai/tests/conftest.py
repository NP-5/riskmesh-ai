import os, tempfile
os.environ["RISKMESH_DATA_DIR"] = tempfile.mkdtemp()
os.environ["RISKMESH_AI_MODE"] = "mock"
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import cases as C

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture()
def fresh(client):
    client.post("/demo/generate", json={"seed": 7})
    return client, client.get("/demo/scenarios").json()
