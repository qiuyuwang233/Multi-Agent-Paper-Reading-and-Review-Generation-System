# -*- coding: utf-8 -*-
"""
向量检索工具（Chroma 持久化优先）。

策略：
1. 优先使用 API Embedding + Chroma 持久化向量库
2. API 或 Chroma 不可用时，按配置降级为本地 sentence-transformers / 哈希向量
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Sequence, Tuple

import numpy as np

from config.settings import Settings, get_settings
from src.schemas import Chunk

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    """Embedding 后端抽象：API 优先，本地与哈希兜底。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.embedding_model
        self._model = None
        self._use_hash_fallback = False
        self._api_failed = False
        self._init_local_backend()

    def _init_local_backend(self) -> None:
        if not self.settings.enable_local_embedding_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.settings.embedding_fallback_model)
        except Exception:
            # 模型下载失败/依赖缺失时自动降级，避免阻塞主流程。
            self._use_hash_fallback = True

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        clean_texts = [t or "" for t in texts]
        if not clean_texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self.settings.embedding_ready and not self._api_failed:
            try:
                return self._embed_with_api(clean_texts)
            except Exception as exc:
                self._api_failed = True
                logger.warning("Embedding API 调用失败，尝试降级到本地向量: %s", exc)

        if self._model is not None:
            vectors = self._model.encode(  # type: ignore[union-attr]
                clean_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=np.float32)

        if self.settings.enable_local_embedding_fallback:
            return np.vstack([self._hash_embed(t) for t in texts]).astype(np.float32)

        raise RuntimeError("Embedding API 不可用，且本地降级已关闭。")

    def _embed_with_api(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("缺少 openai 依赖，请先安装 requirements.txt。") from exc

        api_key = self.settings.embedding_api_key or self.settings.deepseek_api_key
        client = OpenAI(
            api_key=api_key,
            base_url=self.settings.embedding_base_url,
            timeout=self.settings.embedding_timeout,
        )
        resp = client.embeddings.create(model=self.settings.embedding_model, input=list(texts))
        vectors = [item.embedding for item in resp.data]
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
        """
        使用稳定哈希生成可复现向量，用于离线回退。
        """
        vec = np.zeros(dim, dtype=np.float32)
        tokens = [tok for tok in text.replace("\n", " ").split(" ") if tok]
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class LocalVectorStore:
    """Chroma 持久化向量库；不可用时自动退回内存检索。"""

    def __init__(self, collection_name: str | None = None) -> None:
        self.settings = get_settings()
        self.backend = EmbeddingBackend(self.settings)
        self.collection_name = _safe_collection_name(
            collection_name or self.settings.chroma_collection_prefix
        )
        self._client = None
        self._collection = None
        self._chunks: List[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._use_memory = False
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb  # type: ignore

            self.settings.vector_db_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.settings.vector_db_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "论文分块向量库"},
            )
        except Exception as exc:
            self._use_memory = True
            logger.warning("Chroma 初始化失败，降级为内存向量库: %s", exc)

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = np.zeros((0, 1), dtype=np.float32)
            return
        texts = [c.text for c in self._chunks]
        self._matrix = self.backend.embed_texts(texts)
        if self._use_memory or self._collection is None:
            return

        ids = [c.chunk_id for c in self._chunks]
        metadatas = [
            {"page": c.page, "start": c.start, "end": c.end}
            for c in self._chunks
        ]
        try:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=self._matrix.tolist(),
            )
        except Exception as exc:
            if "dimension" not in str(exc).lower():
                raise
            logger.warning("Chroma collection 向量维度不一致，重建 collection: %s", exc)
            self._reset_collection()
            if self._collection is None:
                self._use_memory = True
                return
            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=self._matrix.tolist(),
            )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self._matrix is None and self._collection is None:
            raise RuntimeError("向量库尚未构建，请先调用 build()。")
        if len(self._chunks) == 0:
            return []

        q = self.backend.embed_query(query)
        if not self._use_memory and self._collection is not None:
            result = self._collection.query(
                query_embeddings=[q.tolist()],
                n_results=min(max(top_k, 1), len(self._chunks)),
                include=["documents", "metadatas", "distances"],
            )
            return self._parse_chroma_result(result)

        if self._matrix is None:
            return []
        scores = np.dot(self._matrix, q)
        k = min(max(top_k, 1), len(self._chunks))
        idxs = np.argsort(-scores)[:k]
        return [(self._chunks[int(i)], float(scores[int(i)])) for i in idxs]

    def _parse_chroma_result(self, result: dict) -> List[Tuple[Chunk, float]]:
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: List[Tuple[Chunk, float]] = []
        for idx, chunk_id in enumerate(ids):
            metadata = metas[idx] or {}
            distance = float(distances[idx]) if idx < len(distances) else 0.0
            chunk = Chunk(
                chunk_id=str(chunk_id),
                page=int(metadata.get("page", 1) or 1),
                start=int(metadata.get("start", 0) or 0),
                end=int(metadata.get("end", 0) or 0),
                text=str(docs[idx] if idx < len(docs) else ""),
            )
            # Chroma 默认返回距离，转换成越大越相关的分数便于上层理解。
            chunks.append((chunk, 1.0 / (1.0 + max(distance, 0.0))))
        return chunks

    def _reset_collection(self) -> None:
        """重建当前 collection，用于处理 Embedding 维度变化。"""
        if self._client is None:
            self._collection = None
            return
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "论文分块向量库"},
        )


def _safe_collection_name(name: str) -> str:
    """将论文 ID / 前缀转换为 Chroma 可接受的 collection 名称。"""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}"
    return cleaned[:63]
