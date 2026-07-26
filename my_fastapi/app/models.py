# models.py
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from .database import Base


class User(Base):
    __tablename__ = "users"  # テーブル名

    id = Column(Integer, primary_key=True, index=True)  # 主キー、自動増分
    email = Column(String(255), unique=True, index=True)      # 一意、インデックス作成
    hashed_password = Column(String(255))                      # ハッシュ化パスワード
    is_active = Column(Boolean, default=True)             # 有効かどうか


class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True)                   # タイトル、インデックス作成
    description = Column(String(500))                          # 説明
    owner_id = Column(Integer, ForeignKey("users.id"))   # usersテーブルへの外部キー