from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import schemas, crud
from .database import engine, SessionLocal, Base

# データベーステーブルを作成
Base.metadata.create_all(bind=engine)

app = FastAPI()


# 依存性：データベースセッションを取得
def get_db():
    db = SessionLocal()
    try:
        yield db
        00000
    finally:
        db.close()


# ===== ユーザールート =====
@app.post("/users/", response_model=schemas.UserOut)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # メールアドレスが既に登録されているか確認
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="メールアドレスは既に登録されています")
    return crud.create_user(db=db, user=user)


@app.get("/users/", response_model=list[schemas.UserOut])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.get("/users/{user_id}", response_model=schemas.UserOut)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="ユーザーが存在しません")
    return db_user


# ===== 商品ルート =====
@app.post("/users/{user_id}/items/", response_model=schemas.ItemOut)
def create_item_for_user(
        user_id: int,
        item: schemas.ItemCreate,
        db: Session = Depends(get_db),
):
    return crud.create_item(db=db, item=item, user_id=user_id)


@app.get("/items/", response_model=list[schemas.ItemOut])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_items(db, skip=skip, limit=limit)
    return items
