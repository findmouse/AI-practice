from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

class OverAllState(TypedDict):
    username: str # ユーザー名
    age: int # 年齢
    gender: Literal["male", "female"] # 性別

def get_info_node(state: OverAllState) -> OverAllState:
    username = interrupt("ユーザー名を入力してください：")
    age = interrupt("年齢を入力してください：")
    gender = interrupt("性別を入力してください：(male/female)")

    return {
        "username": username,
        "age": age,
        "gender": gender
    }

builder = StateGraph(state_schema=OverAllState)
builder.add_node("get_info_node", get_info_node)
builder.add_edge(START, "get_info_node")
builder.add_edge("get_info_node", END)

graph = builder.compile()