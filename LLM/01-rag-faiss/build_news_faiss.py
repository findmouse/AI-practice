import argparse
import json
from pathlib import Path

import faiss
from datasets import load_dataset
from tqdm import tqdm

from hashing_embedding import embed_texts


LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download news data and build a local FAISS index for RAG."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Number of AG News examples to download and index.",
    )
    parser.add_argument(
        "--dataset",
        default="fancyzhx/ag_news",
        help="Hugging Face dataset name used as the news corpus.",
    )
    parser.add_argument(
        "--output-dir",
        default="local_faiss_news",
        help="Directory where FAISS index and metadata will be saved.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=4096,
        help="Dimension of the local hashing embedding vectors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="FAISS add batch size.",
    )
    return parser.parse_args()


def write_metadata(records, metadata_path):
    with metadata_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading news dataset: {args.dataset} split=train[:{args.limit}]")
    dataset = load_dataset(args.dataset, split=f"train[:{args.limit}]")

    records = []
    texts = []
    for idx, row in enumerate(dataset):
        label = LABELS[row["label"]]
        text = row["text"].strip()
        records.append(
            {
                "id": idx,
                "source": "ag_news",
                "label": label,
                "text": text,
            }
        )
        texts.append(f"[{label}] {text}")

    print(f"Building local hashing embeddings: dim={args.dim}")
    embeddings = embed_texts(tqdm(texts, desc="Embedding news"), args.dim)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    for start in tqdm(range(0, len(embeddings), args.batch_size), desc="Adding to FAISS"):
        index.add(embeddings[start : start + args.batch_size])

    index_path = output_dir / "news.index"
    metadata_path = output_dir / "metadata.jsonl"
    config_path = output_dir / "config.json"

    faiss.write_index(index, str(index_path))
    write_metadata(records, metadata_path)
    config_path.write_text(
        json.dumps(
            {
                "embedding_backend": "local_hashing",
                "embedding_dim": int(embeddings.shape[1]),
                "index_metric": "cosine_similarity_via_inner_product",
                "document_count": len(records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved FAISS index: {index_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()
