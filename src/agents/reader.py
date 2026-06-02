# -*- coding: utf-8 -*-
"""
Reader Agent：从检索片段抽取结构化信息（带证据来源）。
"""

from __future__ import annotations

import json
from typing import Dict, List

from src.llm import LLMError, get_provider
from src.prompts import READER_SYSTEM_PROMPT
from src.schemas import EvidenceItem, ExtractedInfo, SourceRef
from src.schemas.models import Chunk
from src.tools import CitationIndex


class ReaderAgent:
    """抽取 problem / method / experiments / innovations 四类字段。"""

    def __init__(self) -> None:
        self.provider = get_provider()

    def extract(self, retrieved: Dict[str, List[Chunk]], language: str = "zh") -> ExtractedInfo:
        # 先拉平片段，便于统一建引用索引与兜底逻辑。
        all_chunks = self._flatten_chunks(retrieved)
        citation_index = CitationIndex(all_chunks)

        try:
            return self._extract_with_llm(retrieved, citation_index, language=language)
        except LLMError:
            # 无可用 API Key 或网络异常时降级为规则抽取，保证链路可用。
            return self._extract_with_rules(retrieved, citation_index)
        except Exception:
            return self._extract_with_rules(retrieved, citation_index)

    def _extract_with_llm(
        self,
        retrieved: Dict[str, List[Chunk]],
        citation_index: CitationIndex,
        language: str = "zh",
    ) -> ExtractedInfo:
        chunks_payload = []
        for q, chunks in retrieved.items():
            for c in chunks:
                chunks_payload.append(
                    {
                        "query": q,
                        "chunk_id": c.chunk_id,
                        "page": c.page,
                        "text": c.text[:1500],
                    }
                )

        language_name = _language_name(language)
        system = (
            READER_SYSTEM_PROMPT
            + f"\n请将 content 字段用{language_name}表达；sources.snippet 必须保持原文片段，不要翻译。"
        )
        user = (
            "请输出 JSON，结构如下：\n"
            "{\n"
            '  "problem":[{"content":"...","sources":[{"chunk_id":"...","page":1,"snippet":"..."}]}],\n'
            '  "method":[...],\n'
            '  "experiments":[...],\n'
            '  "innovations":[...]\n'
            "}\n\n"
            "片段列表如下：\n"
            f"{json.dumps(chunks_payload, ensure_ascii=False)}"
        )
        data = self.provider.chat_json(system=system, user=user, temperature=0.1)

        extracted = ExtractedInfo(
            problem=self._parse_items(data.get("problem", []), citation_index, retrieved),
            method=self._parse_items(data.get("method", []), citation_index, retrieved),
            experiments=self._parse_items(data.get("experiments", []), citation_index, retrieved),
            innovations=self._parse_items(data.get("innovations", []), citation_index, retrieved),
        )
        return extracted

    def _extract_with_rules(self, retrieved: Dict[str, List[Chunk]], citation_index: CitationIndex) -> ExtractedInfo:
        """
        规则兜底：按 query 粗分四个类别，每类取 top 证据构造条目。
        """
        problem_items: List[EvidenceItem] = []
        method_items: List[EvidenceItem] = []
        exp_items: List[EvidenceItem] = []
        innovation_items: List[EvidenceItem] = []

        for query, chunks in retrieved.items():
            for chunk in chunks[:2]:
                item = EvidenceItem(
                    content=chunk.text[:180].strip().replace("\n", " "),
                    sources=[
                        SourceRef(
                            chunk_id=chunk.chunk_id,
                            page=chunk.page,
                            snippet=chunk.text[:160],
                        )
                    ],
                )
                q = query.lower()
                if "实验" in query or "指标" in query or "结果" in query or "dataset" in q:
                    exp_items.append(item)
                elif "创新" in query or "贡献" in query or "差异" in query:
                    innovation_items.append(item)
                elif "方法" in query or "模块" in query or "算法" in query:
                    method_items.append(item)
                else:
                    problem_items.append(item)

        return ExtractedInfo(
            problem=problem_items[:5],
            method=method_items[:5],
            experiments=exp_items[:5],
            innovations=innovation_items[:5],
        )

    @staticmethod
    def _flatten_chunks(retrieved: Dict[str, List[Chunk]]) -> List[Chunk]:
        seen = set()
        result: List[Chunk] = []
        for chunks in retrieved.values():
            for c in chunks:
                if c.chunk_id in seen:
                    continue
                seen.add(c.chunk_id)
                result.append(c)
        return result

    @staticmethod
    def _parse_items(
        raw_items: list,
        citation_index: CitationIndex,
        retrieved: Dict[str, List[Chunk]],
    ) -> List[EvidenceItem]:
        parsed: List[EvidenceItem] = []
        fallback_chunk = None
        for chunks in retrieved.values():
            if chunks:
                fallback_chunk = chunks[0]
                break

        for obj in raw_items:
            if not isinstance(obj, dict):
                continue
            content = str(obj.get("content", "")).strip()
            if not content:
                continue
            srcs_raw = obj.get("sources", [])
            sources = []
            if isinstance(srcs_raw, list):
                for s in srcs_raw:
                    if not isinstance(s, dict):
                        continue
                    try:
                        source = SourceRef(
                            chunk_id=str(s.get("chunk_id", "")),
                            page=int(s.get("page", 0) or 0),
                            snippet=str(s.get("snippet", "")),
                        )
                        ok, _ = citation_index.verify_source(source)
                        if ok:
                            sources.append(source)
                    except Exception:
                        continue
            item = EvidenceItem(content=content, sources=sources)
            parsed.append(citation_index.attach_default_source(item, fallback_chunk))
        return parsed


def _language_name(language: str) -> str:
    """将语言代码转换为提示词中的可读名称。"""
    normalized = (language or "zh").lower()
    mapping = {
        "zh": "中文",
        "zh-cn": "中文",
        "chinese": "中文",
        "en": "英文",
        "english": "英文",
        "ja": "日文",
        "japanese": "日文",
    }
    return mapping.get(normalized, language or "中文")

