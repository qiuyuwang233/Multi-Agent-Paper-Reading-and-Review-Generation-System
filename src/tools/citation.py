# -*- coding: utf-8 -*-
"""
引用回溯工具。

用于将 Reader/Writer 产生的结论映射回原文片段，供 Critic 校验。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.schemas import Chunk, EvidenceItem, SourceRef


class CitationIndex:
    """chunk_id -> Chunk 的快速索引。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._index: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._index.get(chunk_id)

    def verify_source(self, source: SourceRef) -> tuple[bool, str]:
        """
        校验来源是否真实存在且片段大体匹配。
        """
        chunk = self.get_chunk(source.chunk_id)
        if chunk is None:
            return False, f"chunk_id 不存在: {source.chunk_id}"
        if source.page > 0 and source.page != chunk.page:
            return False, f"页码不匹配: source.page={source.page}, chunk.page={chunk.page}"
        if source.snippet and source.snippet not in chunk.text:
            # 允许前后截断，做宽松匹配
            short = source.snippet[:40]
            if short and short not in chunk.text:
                return False, "引用片段与原始 chunk 不匹配"
        return True, ""

    def attach_default_source(self, item: EvidenceItem, fallback_chunk: Optional[Chunk]) -> EvidenceItem:
        """
        若条目缺失来源，补一个默认来源，避免 Writer 输出无引用。
        """
        if item.sources or fallback_chunk is None:
            return item
        item.sources = [
            SourceRef(
                chunk_id=fallback_chunk.chunk_id,
                page=fallback_chunk.page,
                snippet=fallback_chunk.text[:160],
            )
        ]
        return item
