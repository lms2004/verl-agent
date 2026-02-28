# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS FOR A PARTICULAR PURPOSE.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Information gain + redundancy penalty step reward for search:
#   R^t = Δ^t - λ * p^t
#   Δ^t = (1/n) * Σ_i (m_i^t - m_i^{t-1})  (coverage increment / uncertainty decrease)
#   p^t = (1/k) * Σ_j 𝟙(d_j^{r(t)} ∈ H^{t-1})  (redundancy penalty)
# Vectors are from E5 via the retrieval server's /embed endpoint.

import hashlib
from typing import List, Set, Tuple

import numpy as np


def _normalize_text(s: str) -> str:
    return " ".join(s.split()) if s else ""


def _doc_id_from_content(content: str) -> str:
    """Stable id for deduplication when API does not provide doc_id."""
    return hashlib.sha256(_normalize_text(content).encode("utf-8")).hexdigest()


def gold_docs_to_texts(gold_docs: List[dict]) -> List[str]:
    """Convert gold docs [{title, paragraph_text}, ...] to list of strings."""
    texts = []
    for d in gold_docs:
        title = str(d.get("title", "")).strip()
        para = str(d.get("paragraph_text", "")).strip()
        texts.append(_normalize_text(f"{title} {para}"))
    return texts


def compute_information_gain_reward(
    gold_embeddings: np.ndarray,
    retrieved_embeddings: np.ndarray,
    retrieved_docs: List[str],
    memory: List[float],
    history_ids: Set[str],
    lambda_: float = 0.0,
) -> Tuple[float, List[float], Set[str], float, float]:
    """
    Compute step reward R^t = Δ^t - λ * p^t using E5 (or other L2-normalized) embeddings.

    Args:
        gold_embeddings: (n_gold, dim) from /embed for gold docs.
        retrieved_embeddings: (k, dim) from /embed for this round's retrieved docs.
        retrieved_docs: Text list for doc_id hashing (redundancy penalty).
        memory: M^{t-1}. history_ids: H^{t-1}. lambda_: redundancy weight.

    Returns:
        reward, new_memory, new_history_ids, information_gain (delta_t), redundancy_penalty (p_t)
    """
    n = gold_embeddings.shape[0]
    if n == 0:
        return 0.0, [], set(), 0.0, 0.0

    retrieved_docs = [d for d in retrieved_docs if _normalize_text(d)]
    if not retrieved_docs or retrieved_embeddings.shape[0] == 0:
        return 0.0, list(memory), set(history_ids), 0.0, 0.0

    gold_embeddings = np.asarray(gold_embeddings, dtype=np.float64)
    retrieved_embeddings = np.asarray(retrieved_embeddings, dtype=np.float64)
    if gold_embeddings.shape[0] != n or retrieved_embeddings.shape[0] != len(retrieved_docs):
        return 0.0, list(memory), set(history_ids), 0.0, 0.0

    sim = np.dot(gold_embeddings, retrieved_embeddings.T)
    c_new = np.max(sim, axis=1)

    memory_arr = np.array(memory, dtype=np.float64)
    m_new = np.maximum(memory_arr, c_new)
    delta_t = float(np.mean(m_new - memory_arr))

    retrieved_ids = [_doc_id_from_content(d) for d in retrieved_docs]
    k = len(retrieved_ids)
    overlap = sum(1 for id_ in retrieved_ids if id_ in history_ids)
    p_t = overlap / k if k else 0.0

    reward = delta_t - lambda_ * p_t
    new_memory = m_new.tolist()
    new_history_ids = history_ids | set(retrieved_ids)

    return reward, new_memory, new_history_ids, delta_t, p_t


def fetch_embeddings_from_api(embed_url: str, texts: List[str], is_passage: bool = True, timeout: int = 30) -> np.ndarray:
    """
    Call the retrieval server's /embed endpoint. Returns (n, dim) numpy array.
    embed_url: e.g. http://127.0.0.1:8000/embed
    """
    import urllib.request
    import json as _json

    texts = [t.strip() for t in texts if t is not None and t.strip()]
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    data = _json.dumps({"texts": texts, "is_passage": is_passage}).encode("utf-8")
    req = urllib.request.Request(
        embed_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = _json.loads(resp.read().decode("utf-8"))
    embeddings = out.get("embeddings", [])
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32)
