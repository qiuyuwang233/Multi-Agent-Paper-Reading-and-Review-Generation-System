# -*- coding: utf-8 -*-
"""
Retriever Agent：基于向量库检索相关证据片段。
"""

from __future__ import annotations

from typing import Dict, List

from config.settings import get_settings
from src.schemas import Chunk
from src.tools import LocalVectorStore


class RetrieverAgent:
    """执行 query -> top_k chunks 检索。"""

    def __init__(self) -> None:
        self.settings = get_settings()

    def retrieve(self, queries: List[str], vector_store: LocalVectorStore) -> Dict[str, List[Chunk]]:
        results: Dict[str, List[Chunk]] = {}
        for q in queries:
            hits = vector_store.search(q, top_k=self.settings.top_k)
            results[q] = [chunk for chunk, _score in hits]
        return results
