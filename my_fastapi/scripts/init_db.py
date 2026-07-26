# coding:utf-8
"""データベース初期化：MYSQL_DB で指定されたデータベースを作成し、テーブルを作成してテストデータを挿入する。

使い方（プロジェクトのルートディレクトリで実行）：
    python -m scripts.init_db
"""
import pymysql

from app.database import (
    Base,
    SessionLocal,
    engine,
    MYSQL_DB,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)
from app import crud, schemas


def create_database_if_not_exists():
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=int(MYSQL_PORT),
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
        print(f"データベース `{MYSQL_DB}` の準備が完了しました")
    finally:
        conn.close()


def insert_test_data():
    db = SessionLocal()
    try:
        test_users = [
            {"email": "alice@example.com", "password": "alice123", "items": [
                {"title": "ノートパソコン", "description": "13インチ 薄型軽量モデル"},
                {"title": "メカニカルキーボード", "description": "青軸 87キー"},
            ]},
            {"email": "bob@example.com", "password": "bob123", "items": [
                {"title": "ワイヤレスマウス", "description": "人間工学デザイン"},
            ]},
            {"email": "carol@example.com", "password": "carol123", "items": []},
        ]

        for user_data in test_users:
            existing = crud.get_user_by_email(db, email=user_data["email"])
            if existing:
                print(f"ユーザー {user_data['email']} は既に存在するためスキップします")
                continue

            user_in = schemas.UserCreate(email=user_data["email"], password=user_data["password"])
            db_user = crud.create_user(db, user=user_in)
            print(f"ユーザーを作成しました: {db_user.email} (id={db_user.id})")

            for item_data in user_data["items"]:
                item_in = schemas.ItemCreate(**item_data)
                db_item = crud.create_item(db, item=item_in, user_id=db_user.id)
                print(f"  商品を作成しました: {db_item.title} (id={db_item.id})")
    finally:
        db.close()


if __name__ == "__main__":
    create_database_if_not_exists()
    Base.metadata.create_all(bind=engine)
    print("テーブルの準備が完了しました")
    insert_test_data()
    print("テストデータの挿入が完了しました")
