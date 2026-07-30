"""
qdrant_manager.py

作者：ChatGPT
功能：
    对 Qdrant 常用操作进行封装。

适用于：
    - RAG
    - Agent
    - 企业知识库

"""

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)


class QdrantManager:

    def __init__(
            self,
            url: str = "http://localhost:6333",
            collection_name: str = "pku",
            vector_size: int = 1536,
            distance: Distance = Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance

        self.client = QdrantClient(url=url)

    # ====================================================
    # Collection
    # ====================================================

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def create_collection(self):

        if self.collection_exists():
            print("Collection 已存在")
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=self.distance,
            ),
        )

        print("Collection 创建成功")

    def recreate_collection(self):

        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

        self.create_collection()

    def delete_collection(self):

        if self.collection_exists():
            self.client.delete_collection(self.collection_name)
            print("Collection 已删除")

    def clear_collection(self):
        """
        清空Collection
        开发中最推荐的方法
        """

        self.recreate_collection()

    def get_collection_info(self):
        return self.client.get_collection(self.collection_name)

    # ====================================================
    # Point
    # ====================================================

    def upsert_points(
            self,
            points: List[PointStruct],
    ):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def retrieve_by_ids(
            self,
            ids: List[int],
    ):
        return self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
        )

    def search(
            self,
            query_vector: List[float],
            limit: int = 5,
    ):
        """
        新版SDK推荐使用 query_points()
        """

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )

        return result.points

    def scroll(
            self,
            limit: int = 20,
    ):

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return points

    def count(self) -> int:

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    # ====================================================
    # delete
    # ====================================================

    def delete_points_by_ids(
            self,
            ids: List[int],
    ):

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=ids,
            ),
        )

    def delete_by_payload(
            self,
            key: str,
            value: Any,
    ):
        """
        根据payload删除

        例如

        source="book.pdf"

        """

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(
                            value=value,
                        ),
                    )
                ]
            ),
        )

    # ====================================================
    # payload
    # ====================================================

    def set_payload(
            self,
            ids: List[int],
            payload: Dict,
    ):

        self.client.set_payload(
            collection_name=self.collection_name,
            payload=payload,
            points=ids,
        )

    # ====================================================
    # util
    # ====================================================

    @staticmethod
    def build_point(
            point_id: int,
            vector: List[float],
            payload: Dict,
    ) -> PointStruct:

        return PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )


# ============================================================
# test
# ============================================================

if __name__ == "__main__":

    import random

    VECTOR_SIZE = 8

    db = QdrantManager(
        collection_name="demo",
        vector_size=VECTOR_SIZE,
    )

    print("=" * 60)
    print("1. recreate collection")
    print("=" * 60)

    db.recreate_collection()

    print(db.get_collection_info())

    print()

    print("=" * 60)
    print("2. insert data")
    print("=" * 60)

    points = []

    for i in range(1, 6):
        vector = [random.random() for _ in range(VECTOR_SIZE)]

        payload = {
            "content": f"这是第{i}条数据",
            "source": "demo.pdf",
            "page": i,
        }

        points.append(
            db.build_point(
                point_id=i,
                vector=vector,
                payload=payload,
            )
        )

    db.upsert_points(points)

    print("插入完成")

    print()

    print("=" * 60)
    print("3. count")
    print("=" * 60)

    print(db.count())

    print()

    print("=" * 60)
    print("4. retrieve")
    print("=" * 60)

    result = db.retrieve_by_ids([1])

    print(result)

    print()

    print("=" * 60)
    print("5. scroll")
    print("=" * 60)

    for point in db.scroll():
        print(point)

    print()

    print("=" * 60)
    print("6. vector search")
    print("=" * 60)

    query = [random.random() for _ in range(VECTOR_SIZE)]

    result = db.search(query)

    for item in result:
        print(item)

    print()

    print("=" * 60)
    print("7. update payload")
    print("=" * 60)

    db.set_payload(
        ids=[1],
        payload={
            "author": "Tom",
        },
    )

    print(db.retrieve_by_ids([1]))

    print()

    print("=" * 60)
    print("8. delete id=2")
    print("=" * 60)

    db.delete_points_by_ids([2])

    print(db.count())

    print()

    print("=" * 60)
    print("9. delete payload")
    print("=" * 60)

    db.delete_by_payload(
        key="source",
        value="demo.pdf",
    )

    print(db.count())

    print()

    print("=" * 60)
    print("10. clear collection")
    print("=" * 60)

    db.clear_collection()

    print(db.count())

    print()

    print("=" * 60)
    print("11. delete collection")
    print("=" * 60)

    db.delete_collection()

    print("Done.")
