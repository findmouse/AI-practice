import hashlib
import re
from collections import Counter

import numpy as np


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def tokenize(text):
    tokens = TOKEN_RE.findall(text.lower())
    bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


def _hash_feature(feature, dim):
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def embed_text(text, dim):
    vector = np.zeros(dim, dtype=np.float32)
    counts = Counter(tokenize(text))

    for feature, count in counts.items():
        index = _hash_feature(feature, dim)
        vector[index] += np.float32(1.0 + np.log1p(count))

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def embed_texts(texts, dim):
    return np.vstack([embed_text(text, dim) for text in texts]).astype(np.float32)
