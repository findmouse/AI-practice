# crud.py
from sqlalchemy.orm import Session
from . import models, schemas


def get_user(db: Session, user_id: int):
    """IDでユーザーを取得"""
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    """メールアドレスでユーザーを取得"""
    return db.query(models.User).filter(models.User.email == email).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    """ユーザー一覧を取得"""
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    """ユーザーを作成"""
    fake_hashed_password = user.password + "notreallyhashed"  # 実際には passlib でハッシュ化すべき
    db_user = models.User(
        email=user.email,
        hashed_password=fake_hashed_password
    )
    db.add(db_user)
    db.commit()        # トランザクションをコミット
    db.refresh(db_user)  # オブジェクトをリフレッシュし、DBが生成したidを取得
    return db_user


def get_items(db: Session, skip: int = 0, limit: int = 100):
    """商品一覧を取得"""
    return db.query(models.Item).offset(skip).limit(limit).all()


def create_item(db: Session, item: schemas.ItemCreate, user_id: int):
    """商品を作成"""
    db_item = models.Item(**item.model_dump(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item