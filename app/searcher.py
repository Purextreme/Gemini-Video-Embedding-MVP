from pathlib import Path

import numpy as np

from app.embedder import GeminiEmbedder
from app.index_store import load_index, load_vectors


def _cosine_scores(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    vector_norms = np.linalg.norm(vectors, axis=1)
    denom = vector_norms * query_norm
    denom[denom == 0] = 1e-12
    return vectors @ query_vector / denom


def search_assets(project_id: str, query: str, filter_type: str = "all", top_k: int = 30) -> list[dict]:
    records = load_index(project_id)
    vectors = load_vectors(project_id)
    if not query.strip() or vectors.size == 0:
        return []

    candidates = []
    candidate_vectors = []
    for record in records:
        vector_index = record.get("vector_index")
        if record.get("status") != "indexed" or vector_index is None:
            continue
        if filter_type != "all" and record.get("file_type") != filter_type:
            continue
        if not Path(record["path"]).exists():
            continue
        if vector_index >= len(vectors):
            continue

        candidates.append(record)
        candidate_vectors.append(vectors[vector_index])

    if not candidates:
        return []

    query_vector = GeminiEmbedder().embed_text(query)
    matrix = np.vstack(candidate_vectors).astype(np.float32)
    scores = _cosine_scores(query_vector, matrix)
    ranked = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked:
        record = candidates[int(idx)]
        results.append(
            {
                "path": record["path"],
                "file_type": record["file_type"],
                "score": float(scores[int(idx)]),
                "thumb_path": record.get("thumb_path"),
                "relative_path": record.get("relative_path"),
                "preview_path": record.get("preview_path"),
                "name": Path(record["path"]).name,
            }
        )
    return results

