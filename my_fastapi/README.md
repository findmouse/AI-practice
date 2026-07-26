# my_fastapi

FastAPI + SQLAlchemy + MySQL によるユーザー/商品管理 API のデモプロジェクト。

## プロジェクト構成

```
my_fastapi/
├── app/                    # アプリケーション本体（Python パッケージ）
│   ├── main.py              # ルーティング定義（エントリポイント）
│   ├── database.py          # DB接続設定（SQLAlchemy engine / session）
│   ├── models.py            # ORM モデル（users, items）
│   ├── schemas.py           # Pydantic スキーマ（リクエスト/レスポンス）
│   ├── crud.py               # DB 操作関数
│   ├── static/               # 静的ファイル
│   └── templates/            # Jinja2 テンプレート
├── scripts/
│   └── init_db.py            # DB作成 + テストデータ投入スクリプト
├── tests/
│   └── test_root.py          # API テスト（pytest）
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```

## 動作環境

- Python 3.11.15
- MySQL 8.0
- Docker Desktop（Docker Engine 29系 / Compose v5系で動作確認済み）

依存パッケージは [requirements.txt](requirements.txt) を参照（FastAPI, SQLAlchemy, PyMySQL, uvicorn, pytest など）。

---

## 方法A：Docker Compose で一括起動（推奨）

コンテナだけで FastAPI + MySQL が両方立ち上がる、一番簡単な方法です。

### 1. コンテナをビルド・起動

プロジェクトのルートディレクトリで実行:

```bash
docker compose up -d --build
```

- `fastapi` サービス：FastAPI アプリ（ポート `8000`）
- `mysql` サービス：MySQL 8.0（ポート `3306`、データベース名 `sql_app` を自動作成）

初回起動時、`app/main.py` の `Base.metadata.create_all(bind=engine)` によって `users` / `items` テーブルが自動作成されます。

### 2. 起動確認

```bash
docker compose logs -f fastapi
```

`Application startup complete.` が出れば起動成功です。

### 3. テストデータを投入する

コンテナ起動後、コンテナの中で初期化スクリプトを実行します：

```bash
docker compose exec fastapi python -m scripts.init_db
```

alice / bob / carol の3ユーザーとサンプル商品データが投入されます（既に存在する場合はスキップされます）。

### 4. 動作確認

ブラウザまたは curl で確認できます：

```bash
curl http://localhost:8000/users/
curl http://localhost:8000/items/
```

Swagger UI（対話的な API ドキュメント）：`http://localhost:8000/docs`

### 5. 停止

```bash
docker compose down
```

データを残したまま停止する場合は上記でOK（`mysql-data` ボリュームにデータが永続化されます）。
データベースの中身も含めて完全に削除したい場合：

```bash
docker compose down -v
```

---

## 方法B：ローカル環境で直接実行

Docker を使わず、ローカルの Python 環境と MySQL で動かす場合。

### 1. MySQL を用意する

ローカルにインストール済みの MySQL、または Docker で MySQL だけ起動する方法があります：

```bash
docker run -d --name mysql-dev -p 3306:3306 -e MYSQL_ROOT_PASSWORD=123456 mysql:8.0
```

### 2. Python 仮想環境を作成し、依存パッケージをインストール

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 接続情報を設定する（必要な場合）

[app/database.py](app/database.py) はデフォルトで以下に接続します：

| 環境変数 | デフォルト値 | 説明 |
|---|---|---|
| `MYSQL_USER` | `root` | ユーザー名 |
| `MYSQL_PASSWORD` | `123456` | パスワード |
| `MYSQL_HOST` | `localhost` | ホスト（Docker Compose 内では `mysql`） |
| `MYSQL_PORT` | `3306` | ポート |
| `MYSQL_DB` | `sql_app` | データベース名 |

デフォルト値と異なる MySQL を使う場合は、上記の環境変数を実行前に設定してください。

### 4. データベースを作成し、テストデータを投入する

プロジェクトのルートディレクトリで実行（`sql_app` データベースが存在しない場合は自動作成されます）：

```bash
python -m scripts.init_db
```

### 5. アプリを起動する

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`http://localhost:8000/docs` にアクセスして動作確認できます。

---

## テストの実行

```bash
pytest
```

`tests/test_root.py` に users / items の主要エンドポイントのテストが含まれています。テストは実際に接続先の MySQL（上記の接続情報）に対してリクエストを行うため、事前にデータベースが起動している必要があります。

---

## 主なAPIエンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| POST | `/users/` | ユーザー作成 |
| GET | `/users/` | ユーザー一覧取得 |
| GET | `/users/{user_id}` | ユーザー詳細取得 |
| POST | `/users/{user_id}/items/` | 指定ユーザーの商品作成 |
| GET | `/items/` | 商品一覧取得 |

詳細なリクエスト/レスポンス形式は `http://localhost:8000/docs`（Swagger UI）を参照してください。
