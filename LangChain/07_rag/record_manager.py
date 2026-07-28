import sqlite3
import threading
import time
from collections.abc import Sequence

from langchain_core.indexing.base import RecordManager


class SQLiteRecordManager(RecordManager):
    """langchain_core.indexing.index() 用の永続 RecordManager 実装。

    langchain-community の SQLRecordManager を使うと langchain-core のバージョンが
    衝突するため、標準ライブラリの sqlite3 だけで同じインターフェースを実装している。
    スクリプトの実行をまたいでファイルに状態が残るので、build_index.py を
    何度実行しても「変更されたチャンクだけ再埋め込みし、消えたチャンクは削除する」
    という増量インデックスが可能になる。
    """

    def __init__(self, namespace: str, db_path: str) -> None:
        super().__init__(namespace)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upsertion_record (
                key TEXT NOT NULL,
                namespace TEXT NOT NULL,
                group_id TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (key, namespace)
            )
            """
        )
        self._conn.commit()

    def create_schema(self) -> None:
        pass

    async def acreate_schema(self) -> None:
        self.create_schema()

    def get_time(self) -> float:
        return time.time()

    async def aget_time(self) -> float:
        return self.get_time()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        if group_ids and len(keys) != len(group_ids):
            raise ValueError("Length of keys must match length of group_ids")

        now = self.get_time()
        if time_at_least and now < time_at_least:
            raise ValueError("time_at_least must be in the past")

        group_ids = group_ids or [None] * len(keys)
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO upsertion_record (key, namespace, group_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (key, namespace) DO UPDATE SET
                    group_id = excluded.group_id,
                    updated_at = excluded.updated_at
                """,
                [(key, self.namespace, gid, now) for key, gid in zip(keys, group_ids)],
            )
            self._conn.commit()

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    def exists(self, keys: Sequence[str]) -> list[bool]:
        if not keys:
            return []
        placeholders = ",".join("?" * len(keys))
        rows = self._conn.execute(
            f"SELECT key FROM upsertion_record WHERE namespace = ? AND key IN ({placeholders})",
            [self.namespace, *keys],
        ).fetchall()
        found = {row[0] for row in rows}
        return [key in found for key in keys]

    async def aexists(self, keys: Sequence[str]) -> list[bool]:
        return self.exists(keys)

    def list_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        query = "SELECT key FROM upsertion_record WHERE namespace = ?"
        params: list = [self.namespace]
        if before is not None:
            query += " AND updated_at < ?"
            params.append(before)
        if after is not None:
            query += " AND updated_at > ?"
            params.append(after)
        if group_ids:
            placeholders = ",".join("?" * len(group_ids))
            query += f" AND group_id IN ({placeholders})"
            params.extend(group_ids)
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [row[0] for row in rows]

    async def alist_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        return self.list_keys(before=before, after=after, group_ids=group_ids, limit=limit)

    def delete_keys(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        placeholders = ",".join("?" * len(keys))
        with self._lock:
            self._conn.execute(
                f"DELETE FROM upsertion_record WHERE namespace = ? AND key IN ({placeholders})",
                [self.namespace, *keys],
            )
            self._conn.commit()

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        self.delete_keys(keys)
