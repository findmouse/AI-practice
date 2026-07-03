# Local FAISS News RAG

This example builds a local FAISS vector index from a downloaded news corpus. It is a good first step for RAG when you do not have your own training or knowledge-base documents yet.

## What This Creates

- `local_faiss_news/news.index`: local FAISS vector index.
- `local_faiss_news/metadata.jsonl`: original news text and labels.
- `local_faiss_news/config.json`: embedding model and index configuration.

The generated `local_faiss_news/` folder is ignored by Git.

## Install

From this directory:

```powershell
cd C:\workspace\AI-practice\LLM\01-rag-faiss
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `faiss-cpu` fails on Windows, use conda:

```powershell
conda create -n rag-faiss python=3.10 -y
conda activate rag-faiss
conda install -c pytorch -c conda-forge faiss-cpu -y
pip install datasets numpy tqdm
```

## Build The News Index

Use a small corpus first:

```powershell
python build_news_faiss.py --limit 200
```

Build a larger local index after the first run works:

```powershell
python build_news_faiss.py --limit 5000
```

The script downloads the `fancyzhx/ag_news` dataset from Hugging Face, embeds the news with a local hashing vectorizer, normalizes embeddings, and stores them in FAISS with inner-product search. With normalized vectors, inner product behaves like cosine similarity.

If you want to use another Hugging Face news dataset later, pass it explicitly:

```powershell
python build_news_faiss.py --dataset namespace/dataset_name --limit 200
```

## Query The Index

```powershell
python query_news_faiss.py "latest business news about company profits" --top-k 5
```

The default news corpus and hashing tokenizer are English-focused. English queries usually retrieve better results with this starter setup:

```powershell
python query_news_faiss.py "technology company and artificial intelligence news" --top-k 5
```

## RAG Flow

1. User asks a question.
2. `query_news_faiss.py` embeds the question.
3. FAISS returns the most similar news records.
4. Pass those records as context to an LLM prompt.

FAISS is not a database server. It is a local vector index saved as files, so this setup is simple, fast, and suitable for local experiments.
