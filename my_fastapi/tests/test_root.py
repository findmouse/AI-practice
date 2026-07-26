import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def unique_email():
    return f"test_{uuid.uuid4().hex}@example.com"


@pytest.fixture
def created_user():
    """テストユーザーを作成し、そのレスポンスデータを返す"""
    response = client.post(
        "/users/",
        json={"email": unique_email(), "password": "secret123"},
    )
    assert response.status_code == 200
    return response.json()


# ===== ユーザーAPIテスト =====

def test_create_user():
    email = unique_email()
    response = client.post(
        "/users/",
        json={"email": email, "password": "secret123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == email
    assert data["is_active"] is True
    assert "id" in data


def test_create_user_duplicate_email(created_user):
    response = client.post(
        "/users/",
        json={"email": created_user["email"], "password": "anotherpass"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "メールアドレスは既に登録されています"


def test_read_users(created_user):
    response = client.get("/users/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any(user["id"] == created_user["id"] for user in data)


def test_read_user(created_user):
    response = client.get(f"/users/{created_user['id']}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == created_user["id"]
    assert data["email"] == created_user["email"]


def test_read_user_not_found():
    response = client.get("/users/999999999")
    assert response.status_code == 404
    assert response.json()["detail"] == "ユーザーが存在しません"


# ===== 商品APIテスト =====

def test_create_item_for_user(created_user):
    response = client.post(
        f"/users/{created_user['id']}/items/",
        json={"title": "テスト商品", "description": "これはテスト商品です"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "テスト商品"
    assert data["description"] == "これはテスト商品です"
    assert data["owner_id"] == created_user["id"]
    assert "id" in data


def test_read_items(created_user):
    client.post(
        f"/users/{created_user['id']}/items/",
        json={"title": "テスト商品2", "description": None},
    )
    response = client.get("/items/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert any(item["owner_id"] == created_user["id"] for item in data)
