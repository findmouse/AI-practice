import argparse
import json
from pathlib import Path

import faiss
import numpy as np

from hashing_embedding import embed_text


def parse_args():
    parser = argparse.ArgumentParser(description="Search a local FAISS news index.")
    parser.add_argument("query", help="Question or keywords to search for.")
    parser.add_argument(
        "--index-dir",
        default="local_faiss_news",
        help="Directory containing news.index, metadata.jsonl, and config.json.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))
    return records


def main():
    args = parse_args()
    index_dir = Path(args.index_dir)
    index_path = index_dir / "news.index"
    metadata_path = index_dir / "metadata.jsonl"
    config_path = index_dir / "config.json"

    if not index_path.exists() or not metadata_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            f"Missing FAISS files under {index_dir}. Run build_news_faiss.py first."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    dim = int(config["embedding_dim"])

    print(f"Loading FAISS index: {index_path}")
    index = faiss.read_index(str(index_path))
    metadata = load_jsonl(metadata_path)

    query_embedding = np.asarray([embed_text(args.query, dim)], dtype=np.float32)

    scores, ids = index.search(query_embedding, args.top_k)

    print(f"\nQuery: {args.query}\n")
    for rank, (score, doc_id) in enumerate(zip(scores[0], ids[0]), start=1):
        if doc_id == -1:
            continue
        record = metadata[int(doc_id)]
        print(f"{rank}. score={score:.4f} label={record['label']} id={record['id']}")
        print(record["text"])
        print()


if __name__ == "__main__":
    main()
