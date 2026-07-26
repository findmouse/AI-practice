import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# データベース設定パラメータ（環境変数で上書き可能、コンテナ内でサービス名接続に使用）
MYSQL_USER = os.environ.get("MYSQL_USER", "root")          # データベースユーザー名
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "123456")  # データベースパスワード
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")     # データベースホストアドレス（ローカルは localhost、docker-compose 内はサービス名 mysql）
MYSQL_PORT = os.environ.get("MYSQL_PORT", "3306")          # ポート番号、デフォルト 3306
MYSQL_DB = os.environ.get("MYSQL_DB", "sql_app")         # データベース名

# MySQL 接続文字列の形式：mysql+pymysql://ユーザー名:パスワード@ホスト:ポート/データベース名?charset=utf8mb4
SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
)

# データベースエンジンを作成
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=10,            # コネクションプールが保持する常駐接続数
    max_overflow=20,         # pool_size を超えた場合に一時的に作成できる最大接続数
    pool_recycle=3600,       # 接続を自動回収する時間（秒）、MySQL の8時間アイドル切断問題を防止
    pool_pre_ping=True       # 接続取得前に毎回 ping を行い、接続の有効性を確認
)

# セッションファクトリを作成
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ベースクラスを宣言
class Base(DeclarativeBase):
    pass