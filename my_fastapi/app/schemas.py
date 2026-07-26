# schemas.py
from pydantic import BaseModel, ConfigDict


# ===== ユーザーモデル =====
class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str          # 作成時にパスワードが必要


class UserOut(UserBase):
    id: int
    is_active: bool

    model_config = ConfigDict(from_attributes=True)  # ORMオブジェクトに対応


# ===== 商品モデル =====
class ItemBase(BaseModel):
    title: str
    description: str | None = None


class ItemCreate(ItemBase):
    pass


class ItemOut(ItemBase):
    id: int
    owner_id: int

    model_config = ConfigDict(from_attributes=True)